import gymnasium as gym
import ale_py
import time
import cv2
import os
import re
import numpy as np
import math
from ultralytics import YOLO
from collections import deque
import dashscope
from dashscope import MultiModalConversation

# pacman相关导入
from detect_all import detect_all_in_one, crop_image, find_label, detect_score, detect_HP
from utils_all.game_utils import create_pacman_environment

# 注册Atari环境
gym.register_envs(ale_py)

# VLM API配置
dashscope.api_key = "sk-361f43ece66a49e299a35ef26ac687d7"#wangjun

# VLM对话历史（用于连续对话）
vlm_conversation_history = []


# ==================== VLM调用函数 ====================
def call_qwen_vl(image_path=None, prompt="", vlm_model='qwen-vl-plus', use_history=False, reset_history=False):
    """
    调用 Qwen-VL 多模态模型分析图像
    
    :param image_path: 图片文件路径
    :param prompt: 文字提示
    :param vlm_model: VLM模型名称
    :param use_history: 是否使用对话历史
    :param reset_history: 是否重置对话历史
    :return: VLM返回的文本
    """
    global vlm_conversation_history
    
    if reset_history:
        vlm_conversation_history = []
    
    # 构建消息内容
    if image_path:
        user_content = [
            {"image": f"file://{image_path}"},
            {"text": prompt}
        ]
    else:
        user_content = [{"text": prompt}]
    
    # 添加用户消息到历史
    if use_history:
        vlm_conversation_history.append({
            "role": "user",
            "content": user_content
        })
        messages = vlm_conversation_history
    else:
        messages = [{"role": "user", "content": user_content}]
    
    try:
        response = MultiModalConversation.call(
            model=vlm_model,
            messages=messages
        )
        
        if response.status_code == 200:
            assistant_response = response.output.choices[0].message.content[0]['text']
            
            if use_history:
                vlm_conversation_history.append({
                    "role": "assistant",
                    "content": [{"text": assistant_response}]
                })
            
            return assistant_response
        else:
            return f"Error: {response.code} - {response.message}"
    except Exception as e:
        return f"Exception: {str(e)}"


def parse_vlm_action(vlm_response):
    """
    从VLM返回的文本中解析动作
    
    :param vlm_response: VLM返回的文本
    :return: 动作编号 (0-4)，解析失败返回0
    """
    # 尝试从返回文本中提取数字
    numbers = re.findall(r'\b\d+\.?\d*\b', vlm_response)
    
    if numbers:
        action = int(float(numbers[0]))
        # 确保动作在有效范围内
        if 0 <= action <= 4:
            return action
    
    return 0  # 默认返回NOOP


# ==================== 配置参数类 ====================
class GameArgs:
    """游戏配置参数"""
    def __init__(self):
        self.size = 256  # 图片大小（不要修改）
        self.visualize_save = False  # 是否保存可视化结果（关闭以提高流畅度）
        self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"  # YOLO模型路径
        self.your_mission_name = "BFS_Mission"  # 任务名称，用于保存结果
        self.game_name = 'MsPacmanNoFrameskip-v4'  # 游戏名称
        self.vlm = 'qwen-vl-plus'  # VLM模型名称
        self.ghost_danger_threshold = 20  # 鬼危险距离阈值（曼哈顿距离）
        
        # 大力丸追逐策略参数
        self.superpill_chase_threshold = 50  # 大力丸追逐距离阈值：Pacman离大力丸的距离小于此值时考虑追逐
        self.superpill_safe_margin = 20      # 安全边际：最近Ghost距离需要比大力丸距离多出这个值才会追逐
        
        # 追击Ghost策略参数（吃掉大力丸后）
        self.ghost_chase_threshold = 80  # 追击Ghost的距离阈值：Ghost距离小于此值时主动追击


# ==================== 游戏信息存储类 ====================
class GameState:
    """
    存储从detect_all_in_one获取的游戏状态信息
    用于BFS路径规划和决策
    """
    
    # 反方向映射：用于排除回头路
    OPPOSITE_DIRECTION = {
        'up': 'down',
        'down': 'up',
        'left': 'right',
        'right': 'left'
    }
    
    # 动作编号到方向名的映射
    ACTION_TO_DIRECTION = {
        0: None,      # NOOP
        1: 'up',
        2: 'right',
        3: 'left',
        4: 'down'
    }
    
    def __init__(self):
        # Pacman信息
        self.pacman_boxes = []          # pacman边界框 [[x1, y1, x2, y2], ...]
        self.pacman_centers = []        # pacman中心点 [[x, y], ...]
        self.pacman_position = None     # 当前pacman位置 (x, y)
        
        # Ghost信息
        self.ghosts_boxes = []          # 所有ghost边界框
        self.ghosts_centers = []        # 所有ghost中心点
        self.four_ghosts_boxes = []     # 4个ghost的边界框（用于算法）
        self.four_ghosts_centers = []   # 4个ghost的中心点（用于算法）
        self.ghost_num = 0              # ghost数量
        
        # Pill（豆子）信息
        self.pill_centers = []          # 所有豆子中心点 [[x, y], ...]
        self.pill_num = 0               # 豆子数量
        
        # SuperPill（大力丸）信息
        self.superpill_boxes = []       # 大力丸边界框
        self.superpill_centers = []     # 大力丸中心点
        self.superpill_info = None      # 大力丸完整信息
        
        # Door（传送门）信息
        self.door_centers = []          # 传送门中心点
        
        # 障碍物信息
        self.obstacles_mask = None      # 障碍物掩码（二值图像，用于路径规划）
        
        # 决策信息
        self.pacman_decision = {}       # 可行动方向 {'up': 1/0, 'down': 1/0, 'left': 1/0, 'right': 1/0}
        self.legal_action_num = 0       # 可行动方向数量
        
        # 上一步动作记录（用于防止兜圈）
        self.last_action = None         # 上一步执行的动作编号
        self.last_direction = None      # 上一步的方向名 ('up', 'down', 'left', 'right')
        
        # 游戏状态
        self.score = 0                  # 当前得分
        self.HP = 0                     # 当前生命值
        self.state = 'init'             # 游戏状态: 'init'(初始化), 'run'(逃跑), 'chase'(追击)
        
        # 帧信息
        self.frame = 0                  # 当前帧数
        self.epoch = 0                  # 游戏轮次（死亡/吃完豆子后重置）
        
        # 位置停留检测（用于调试卡住问题）
        self.stuck_position = None       # 记录可能卡住的位置
        self.stuck_frames = 0            # 在该位置停留的帧数
        self.stuck_threshold = 30        # 判定为卡住的帧数阈值
        self.stuck_distance = 5          # 判定为同一位置的距离阈值
        
        # 卡住时的VLM决策历史（避免重复选择相同方向）
        self.stuck_vlm_history = {}      # {(x, y): [已尝试的动作列表]}
        self.stuck_history_distance = 10  # 判定为同一卡住位置的距离阈值
    
    def set_last_action(self, action):
        """
        记录上一步执行的动作
        
        :param action: 动作编号 (0:NOOP, 1:UP, 2:RIGHT, 3:LEFT, 4:DOWN)
        """
        self.last_action = action
        self.last_direction = self.ACTION_TO_DIRECTION.get(action, None)
    
    def get_opposite_direction(self):
        """
        获取上一步动作的反方向（需要被排除的方向）
        
        :return: 反方向名，如果上一步是None则返回None
        """
        if self.last_direction is None:
            return None
        return self.OPPOSITE_DIRECTION.get(self.last_direction, None)
    
    def get_legal_actions_no_backtrack(self):
        """
        获取排除回头路后的合法动作列表
        
        :return: 动作列表，如 ['up', 'left', 'right']（排除了上一步的反方向）
        """
        legal = self.get_legal_actions()
        opposite = self.get_opposite_direction()
        
        # 如果有反方向且合法动作数量大于1，则排除反方向
        if opposite and opposite in legal and len(legal) > 1:
            legal = [a for a in legal if a != opposite]
        
        return legal
    
    def update_from_detect_all(self, all_game_info, frame, epoch):
        """
        从detect_all_in_one的返回值更新游戏状态
        
        :param all_game_info: detect_all_in_one返回的字典
        :param frame: 当前帧数
        :param epoch: 当前轮次
        """
        # 更新帧信息
        self.frame = frame
        self.epoch = epoch
        
        # 更新Pacman信息
        self.pacman_boxes = all_game_info.get('pacman_boxes', [])
        self.pacman_centers = all_game_info.get('pacman_centers', [])
        # 检查pacman_centers是否为空，为空时保持上一帧的位置
        if self.pacman_centers and len(self.pacman_centers) > 0:
            self.pacman_position = tuple(self.pacman_centers[0])  # (x, y)
        # 如果为空，pacman_position保持不变（使用上一帧的位置）
        
        # 更新Ghost信息
        self.ghosts_boxes = all_game_info.get('ghosts_boxes', [])
        self.ghosts_centers = all_game_info.get('ghosts_centers', [])
        self.four_ghosts_boxes = all_game_info.get('4ghosts_boxes', [])
        self.four_ghosts_centers = all_game_info.get('4ghosts_centers', [])
        self.ghost_num = all_game_info.get('ghost_num', 0)
        
        # 更新Pill信息
        self.pill_centers = all_game_info.get('pill_centers', [])
        pill_num_list = all_game_info.get('pill_num', [0])
        self.pill_num = pill_num_list[0] if isinstance(pill_num_list, list) else pill_num_list
        
        # 更新SuperPill信息
        self.superpill_boxes = all_game_info.get('superpill_boxes', [])
        self.superpill_centers = all_game_info.get('superpill_centers', [])
        self.superpill_info = all_game_info.get('superpill_info', None)
        
        # 更新Door信息
        self.door_centers = all_game_info.get('door_centers', [])
        
        # 更新障碍物信息
        self.obstacles_mask = all_game_info.get('obstacles_mask', None)
        
        # 更新决策信息
        self.pacman_decision = all_game_info.get('pacman_decision', {})
        self.legal_action_num = all_game_info.get('legal_action_num', 0)
        
        # 更新游戏状态
        self.score = all_game_info.get('score', 0)
        self.HP = all_game_info.get('HP', 0)
        self.state = all_game_info.get('state', 'init')
    
    def get_pacman_pos(self):
        """获取Pacman当前位置"""
        return self.pacman_position
    
    def get_ghost_positions(self):
        """获取所有有效Ghost的位置列表"""
        positions = []
        for center in self.four_ghosts_centers:
            if center is not None and len(center) == 2:
                positions.append(tuple(center))
        return positions
    
    def get_pill_positions(self):
        """获取所有豆子位置列表"""
        return [tuple(center) for center in self.pill_centers if center and len(center) == 2]
    
    def get_superpill_positions(self):
        """获取所有大力丸位置列表"""
        return [tuple(center) for center in self.superpill_centers if center and len(center) == 2]
    
    def get_legal_actions(self):
        """
        获取当前可执行的动作列表
        返回: 动作列表，如 ['up', 'down', 'left', 'right']
        """
        legal = []
        for direction, is_legal in self.pacman_decision.items():
            if is_legal == 1:
                legal.append(direction)
        return legal
    
    def is_in_danger(self, threshold=30):
        """
        判断Pacman是否处于危险状态（离Ghost太近）
        
        :param threshold: 曼哈顿距离阈值
        :return: (是否危险, 最近Ghost距离, 最近Ghost位置)
        """
        if self.pacman_position is None:
            return False, float('inf'), None
        
        pacman_x, pacman_y = self.pacman_position
        min_distance = float('inf')
        nearest_ghost = None
        
        for ghost_pos in self.get_ghost_positions():
            ghost_x, ghost_y = ghost_pos
            # 计算曼哈顿距离
            distance = abs(pacman_x - ghost_x) + abs(pacman_y - ghost_y)
            if distance < min_distance:
                min_distance = distance
                nearest_ghost = ghost_pos
        
        return min_distance < threshold, min_distance, nearest_ghost
    
    def count_nearby_ghosts(self, threshold=30):
        """
        统计在危险阈值范围内的Ghost数量
        
        :param threshold: 曼哈顿距离阈值
        :return: (nearby_count, ghost_distances) - 附近Ghost数量, 各Ghost距离列表[(pos, dist), ...]
        """
        if self.pacman_position is None:
            return 0, []
        
        pacman_x, pacman_y = self.pacman_position
        ghost_distances = []
        
        for ghost_pos in self.get_ghost_positions():
            ghost_x, ghost_y = ghost_pos
            distance = abs(pacman_x - ghost_x) + abs(pacman_y - ghost_y)
            if distance < threshold:
                ghost_distances.append((ghost_pos, distance))
        
        # 按距离排序
        ghost_distances.sort(key=lambda x: x[1])
        
        return len(ghost_distances), ghost_distances
    
    def is_multi_ghost_danger(self, threshold=30, min_ghost_count=2):
        """
        判断是否面临多Ghost威胁（需要VLM介入）
        
        :param threshold: 曼哈顿距离阈值
        :param min_ghost_count: 最少Ghost数量阈值
        :return: (is_multi_danger, nearby_count, ghost_distances)
        """
        nearby_count, ghost_distances = self.count_nearby_ghosts(threshold)
        is_multi_danger = nearby_count >= min_ghost_count
        return is_multi_danger, nearby_count, ghost_distances
    
    def should_chase_superpill(self, chase_threshold=50, safe_margin=20):
        """
        判断是否应该追逐大力丸
        
        条件：
        1. Pacman离最近的大力丸距离小于 chase_threshold
        2. 最近的Ghost距离 > 大力丸距离 + safe_margin
        
        :param chase_threshold: Pacman距离大力丸的阈值
        :param safe_margin: Ghost需要比大力丸距离多出的安全边际
        :return: (should_chase, nearest_superpill_pos, superpill_dist, ghost_dist)
        """
        superpill_positions = self.get_superpill_positions()
        
        # 如果没有大力丸，不追逐
        if not superpill_positions:
            return False, None, float('inf'), float('inf')
        
        pacman_pos = self.get_pacman_pos()
        if pacman_pos is None:
            return False, None, float('inf'), float('inf')
        
        # 找到最近的大力丸
        min_superpill_dist = float('inf')
        nearest_superpill = None
        for sp_pos in superpill_positions:
            distance = manhattan_distance(pacman_pos, sp_pos)
            if distance < min_superpill_dist:
                min_superpill_dist = distance
                nearest_superpill = sp_pos
        
        # 获取最近Ghost的距离
        _, ghost_dist, _ = self.is_in_danger(threshold=float('inf'))
        
        # 判断是否应该追逐大力丸
        should_chase = (
            min_superpill_dist < chase_threshold and
            ghost_dist > min_superpill_dist + safe_margin
        )
        
        return should_chase, nearest_superpill, min_superpill_dist, ghost_dist
    
    def should_chase_ghost(self, chase_threshold=60):
        """
        判断是否应该追击Ghost（在chase状态下，即吃掉大力丸后）
        
        条件：
        1. 当前处于chase状态（吃掉大力丸后）
        2. 最近的Ghost距离小于 chase_threshold
        
        :param chase_threshold: 追击Ghost的距离阈值
        :return: (should_chase, nearest_ghost_pos, ghost_dist)
        """
        # 必须处于chase状态
        if self.state != 'chase':
            return False, None, float('inf')
        
        pacman_pos = self.get_pacman_pos()
        if pacman_pos is None:
            return False, None, float('inf')
        
        ghost_positions = self.get_ghost_positions()
        if not ghost_positions:
            return False, None, float('inf')
        
        # 找到最近的Ghost
        min_ghost_dist = float('inf')
        nearest_ghost = None
        for ghost_pos in ghost_positions:
            distance = manhattan_distance(pacman_pos, ghost_pos)
            if distance < min_ghost_dist:
                min_ghost_dist = distance
                nearest_ghost = ghost_pos
        
        # 判断是否应该追击Ghost
        should_chase = min_ghost_dist < chase_threshold
        
        return should_chase, nearest_ghost, min_ghost_dist
    
    def print_state(self):
        """打印当前游戏状态（用于调试）"""
        print("=" * 50)
        print(f"[Frame {self.frame}, Epoch {self.epoch}] Game State:")
        print(f"  Pacman Position: {self.pacman_position}")
        print(f"  Ghost Positions: {self.get_ghost_positions()}")
        print(f"  Pill Count: {self.pill_num}")
        print(f"  SuperPill Count: {len(self.superpill_centers)}")
        print(f"  Legal Actions: {self.get_legal_actions()}")
        print(f"  Score: {self.score}, HP: {self.HP}")
        print(f"  State: {self.state}")
        is_danger, dist, nearest = self.is_in_danger()
        print(f"  In Danger: {is_danger}, Nearest Ghost Distance: {dist}")
        print("=" * 50)
    
    def check_stuck(self):
        """
        检测Pacman是否卡住（在同一位置停留过长时间）
        
        :return: (is_stuck, stuck_frames) - 是否卡住，已停留帧数
        """
        current_pos = self.get_pacman_pos()
        
        if current_pos is None:
            return False, 0
        
        # 如果没有记录位置，初始化
        if self.stuck_position is None:
            self.stuck_position = current_pos
            self.stuck_frames = 1
            return False, 1
        
        # 计算与记录位置的距离
        distance = manhattan_distance(current_pos, self.stuck_position)
        
        if distance <= self.stuck_distance:
            # 仍在同一位置附近，增加计数
            self.stuck_frames += 1
        else:
            # 已移动，重置记录
            self.stuck_position = current_pos
            self.stuck_frames = 1
        
        # 判断是否卡住
        is_stuck = self.stuck_frames >= self.stuck_threshold
        return is_stuck, self.stuck_frames
    
    def reset_stuck_detection(self):
        """重置卡住检测状态"""
        self.stuck_position = None
        self.stuck_frames = 0
    
    def get_stuck_history_key(self, pos):
        """
        获取卡住位置对应的历史记录键
        将位置量化到网格，便于匹配相近位置
        
        :param pos: 位置 (x, y)
        :return: 量化后的位置键 (qx, qy)
        """
        if pos is None:
            return None
        # 将位置量化到10像素的网格
        grid_size = self.stuck_history_distance
        qx = int(pos[0] // grid_size) * grid_size
        qy = int(pos[1] // grid_size) * grid_size
        return (qx, qy)
    
    def get_stuck_tried_actions(self, pos):
        """
        获取在某位置已经尝试过的VLM动作
        
        :param pos: 当前位置 (x, y)
        :return: 已尝试的动作列表
        """
        key = self.get_stuck_history_key(pos)
        if key is None:
            return []
        return self.stuck_vlm_history.get(key, [])
    
    def add_stuck_tried_action(self, pos, action):
        """
        记录在某位置尝试的VLM动作
        
        :param pos: 当前位置 (x, y)
        :param action: 执行的动作编号
        """
        key = self.get_stuck_history_key(pos)
        if key is None:
            return
        if key not in self.stuck_vlm_history:
            self.stuck_vlm_history[key] = []
        if action not in self.stuck_vlm_history[key]:
            self.stuck_vlm_history[key].append(action)
    
    def clear_stuck_history(self):
        """清空卡住历史记录（游戏重置时调用）"""
        self.stuck_vlm_history = {}


# ==================== 辅助函数 ====================
def manhattan_distance(pos1, pos2):
    """
    计算两点之间的曼哈顿距离
    
    :param pos1: 点1坐标 (x1, y1)
    :param pos2: 点2坐标 (x2, y2)
    :return: 曼哈顿距离
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1, pos2):
    """
    计算两点之间的欧几里得距离
    
    :param pos1: 点1坐标 (x1, y1)
    :param pos2: 点2坐标 (x2, y2)
    :return: 欧几里得距离
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


# ==================== VLM Prompt 生成函数 ====================
def generate_low_pill_guide_prompt(game_state):
    """
    生成豆子数量较少时的VLM引导Prompt
    
    :param game_state: GameState对象
    :return: prompt字符串
    """
    pacman_pos = game_state.get_pacman_pos()
    ghost_positions = game_state.get_ghost_positions()
    legal_actions = game_state.get_legal_actions()
    pill_positions = game_state.get_pill_positions()
    superpill_positions = game_state.get_superpill_positions()
    
    # 构建豆子位置信息
    pill_info = []
    if pacman_pos and pill_positions:
        for pill in pill_positions:
            dist = manhattan_distance(pacman_pos, pill)
            pill_info.append((pill, dist))
        pill_info.sort(key=lambda x: x[1])  # 按距离排序
    
    # 取最近的5个豆子
    nearest_pills = pill_info[:5] if pill_info else []

    prompt = f"""你是Pac-Man游戏AI助手。当前豆子数量很少（只剩{game_state.pill_num}个），需要你帮助找到并吃掉剩余的豆子来通关。

【当前状态】
- Pacman位置: {pacman_pos}
- 剩余豆子数: {game_state.pill_num}
- 可执行动作: {legal_actions}
- Ghost位置: {ghost_positions}
- 大力丸位置: {superpill_positions}

【最近的豆子位置和距离】
{chr(10).join([f"  - 位置{p[0]}, 距离{p[1]:.1f}" for p in nearest_pills]) if nearest_pills else "  无法检测到豆子位置"}

【动作编号】
0=静止, 1=上, 2=右, 3=左, 4=下

【决策要求】
1. 优先选择朝向最近豆子的方向
2. 避开Ghost所在方向
3. 如果豆子在左右两侧，优先水平移动
4. 如果豆子在上下方向，优先垂直移动

请只输出一个数字（0-4）表示建议的动作，不要输出其他内容！"""

    return prompt


def generate_stuck_prompt(game_state, tried_actions=None):
    """
    生成Pacman卡住时的VLM Prompt
    
    :param game_state: GameState对象
    :param tried_actions: 在该位置已尝试过的动作列表
    :return: prompt字符串
    """
    pacman_pos = game_state.get_pacman_pos()
    ghost_positions = game_state.get_ghost_positions()
    legal_actions = game_state.get_legal_actions()
    pill_positions = game_state.get_pill_positions()
    superpill_positions = game_state.get_superpill_positions()
    
    # 找最近的豆子
    nearest_pill = None
    min_pill_dist = float('inf')
    if pacman_pos and pill_positions:
        for pill in pill_positions:
            dist = manhattan_distance(pacman_pos, pill)
            if dist < min_pill_dist:
                min_pill_dist = dist
                nearest_pill = pill
    
    # 将已尝试的动作转换为方向名
    action_to_name = {0: '静止', 1: '上', 2: '右', 3: '左', 4: '下'}
    tried_actions_str = ""
    if tried_actions and len(tried_actions) > 0:
        tried_names = [action_to_name.get(a, str(a)) for a in tried_actions]
        tried_actions_str = f"\n- ⚠️ 在此位置已尝试过的方向: {tried_names}（请不要再选择这些方向！）"

    prompt = f"""你是Pac-Man游戏AI助手。Pacman已经卡住了，在同一位置停留了{game_state.stuck_frames}帧,你应该决策一个方向，保证逃离这个区域，防止继续被卡住，朝着豆子最近的方向走。

【当前状态】
- Pacman位置: {pacman_pos}
- 可执行动作: {legal_actions}
- 上一步动作: {game_state.last_direction}（可能导致卡住）{tried_actions_str}
- Ghost位置: {ghost_positions}
- 最近豆子位置: {nearest_pill}，距离: {min_pill_dist:.1f}
- 大力丸位置: {superpill_positions}
- 剩余豆子数: {game_state.pill_num}

【动作编号】
0=静止, 1=上, 2=右, 3=左, 4=下

【分析要求】
1. Pacman卡住通常是因为重复往返或撞墙
2. 请选择一个与上一步不同的方向
3. {"⚠️ 重要：必须避开已尝试过的方向！" if tried_actions else ""}
4. 优先选择朝向最近豆子的方向
5. 避开Ghost所在方向

请只输出一个数字（0-4）表示建议的动作，不要输出其他内容！"""

    return prompt


def generate_multi_ghost_danger_prompt(game_state, ghost_distances):
    """
    生成多Ghost围堵时的VLM Prompt
    
    :param game_state: GameState对象
    :param ghost_distances: Ghost距离列表 [(pos, dist), ...]
    :return: prompt字符串
    """
    pacman_pos = game_state.get_pacman_pos()
    legal_actions = game_state.get_legal_actions()
    superpill_positions = game_state.get_superpill_positions()
    
    # 分析各方向的Ghost威胁
    direction_threats = {'up': [], 'down': [], 'left': [], 'right': []}
    
    if pacman_pos:
        px, py = pacman_pos
        for ghost_pos, dist in ghost_distances:
            gx, gy = ghost_pos
            # 判断Ghost在Pacman的哪个方向
            if gy < py:  # Ghost在上方
                direction_threats['up'].append((ghost_pos, dist))
            if gy > py:  # Ghost在下方
                direction_threats['down'].append((ghost_pos, dist))
            if gx < px:  # Ghost在左边
                direction_threats['left'].append((ghost_pos, dist))
            if gx > px:  # Ghost在右边
                direction_threats['right'].append((ghost_pos, dist))
    
    # 找最近的大力丸
    nearest_superpill = None
    superpill_dist = float('inf')
    if pacman_pos and superpill_positions:
        for sp in superpill_positions:
            dist = manhattan_distance(pacman_pos, sp)
            if dist < superpill_dist:
                superpill_dist = dist
                nearest_superpill = sp
    
    prompt = f"""你是Pac-Man游戏AI助手。Pacman正面临{len(ghost_distances)}个Ghost的围堵，情况危急！

【当前危机】
- Pacman位置: {pacman_pos}
- 附近Ghost数量: {len(ghost_distances)}
- Ghost详细位置和距离: {[(pos, f'{dist:.1f}') for pos, dist in ghost_distances]}

【各方向威胁分析】
- 上方威胁: {len(direction_threats['up'])}个Ghost
- 下方威胁: {len(direction_threats['down'])}个Ghost  
- 左边威胁: {len(direction_threats['left'])}个Ghost
- 右边威胁: {len(direction_threats['right'])}个Ghost

【可用资源】
- 可执行动作: {legal_actions}
- 最近大力丸: {nearest_superpill}，距离: {superpill_dist:.1f}

【动作编号】
0=静止, 1=上, 2=右, 3=左, 4=下

【逃脱策略建议】
1. 选择Ghost威胁最少的方向逃跑
2. 如果大力丸距离较近且路上Ghost较少，可以冲向大力丸
3. 尽量不要静止不动
4. 避免走向死角

请只输出一个数字（0-4）表示最佳逃脱方向，不要输出其他内容！"""

    return prompt


def vlm_decide_action(game_state, env_img, scenario, args, ghost_distances=None, save_dir="vlm_debug", tried_actions=None, vlm_image_path=None):
    """
    使用VLM进行决策
    
    :param game_state: GameState对象
    :param env_img: 当前帧图像 (BGR格式)
    :param scenario: 场景类型 ('stuck', 'multi_ghost', 'low_pill')
    :param args: 配置参数
    :param ghost_distances: Ghost距离列表（仅multi_ghost场景需要）
    :param save_dir: 调试图片保存目录
    :param tried_actions: 在卡住位置已尝试过的动作列表（仅stuck场景需要）
    :param vlm_image_path: detect_all_in_one生成的4vlm标注图片路径（如果提供则使用此图片）
    :return: 动作编号 (0-4)
    """
    # 确定使用的图片路径
    if vlm_image_path and os.path.exists(vlm_image_path):
        # 使用detect_all_in_one生成的4vlm标注图片
        image_path = vlm_image_path
        print(f"   使用4VLM标注图片: {image_path}")
    else:
        # 创建保存目录并保存原始图片
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(save_dir, f"vlm_{scenario}_frame{game_state.frame}_{timestamp}.png")
        cv2.imwrite(image_path, env_img)
    
    # 根据场景生成不同的Prompt
    if scenario == 'stuck':
        prompt = generate_stuck_prompt(game_state, tried_actions)
        print(f"🤖 VLM介入：Pacman卡住，请求AI决策...")
        if tried_actions:
            print(f"   已尝试过的动作: {tried_actions}")
    elif scenario == 'multi_ghost':
        prompt = generate_multi_ghost_danger_prompt(game_state, ghost_distances)
        print(f"🤖 VLM介入：{len(ghost_distances)}个Ghost围堵，请求AI决策...")
    elif scenario == 'low_pill':
        prompt = generate_low_pill_guide_prompt(game_state)
        print(f"🤖 VLM介入：豆子数量少({game_state.pill_num}个)，请求AI引导...")
    else:
        return 0
    
    # 调用VLM
    response = call_qwen_vl(
        image_path=image_path,
        prompt=prompt,
        vlm_model=args.vlm,
        use_history=False
    )
    
    print(f"   VLM响应: {response}")
    
    # 解析动作
    action = parse_vlm_action(response)
    
    # 对于stuck场景，如果VLM返回的动作在已尝试列表中，强制选择其他合法动作
    if scenario == 'stuck' and tried_actions and action in tried_actions:
        legal_actions = game_state.get_legal_actions()
        reverse_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
        # 找一个没尝试过的合法动作
        for legal in legal_actions:
            if legal in reverse_map:
                alt_action = reverse_map[legal]
                if alt_action not in tried_actions:
                    print(f"   VLM选择了已尝试过的动作{action}，强制改为: {alt_action} ({legal})")
                    action = alt_action
                    break
    
    # 验证动作是否合法
    legal_actions = game_state.get_legal_actions()
    action_map = {1: 'up', 2: 'right', 3: 'left', 4: 'down'}
    
    if action in action_map and action_map[action] not in legal_actions:
        # VLM建议的动作不合法，选择一个合法动作
        for legal in legal_actions:
            reverse_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
            if legal in reverse_map:
                action = reverse_map[legal]
                print(f"   VLM建议动作不合法，改为: {action} ({legal})")
                break
    
    print(f"   最终决策: {action}")
    return action


def get_4vlm_image_path(args, epoch, frame):
    """
    获取detect_all_in_one生成的4vlm标注图片路径
    
    :param args: 配置参数（包含your_mission_name）
    :param epoch: 当前轮次
    :param frame: 当前帧数
    :return: 图片路径
    """
    file_name = args.your_mission_name + "_4vlm"
    frame_str = str(frame).zfill(6)
    image_path = os.path.join("detection_results", file_name, f"{epoch}_4vlm_frame_{frame_str}.png")
    return image_path


def save_stuck_detection_image(env_img, all_game_info, game_state, frame, epoch, save_dir="stuck_detection"):
    """
    保存Pacman卡住时的检测图片，包含legal action箭头
    
    :param env_img: 当前帧图像 (BGR格式)
    :param all_game_info: detect_all_in_one返回的游戏信息
    :param game_state: GameState对象
    :param frame: 当前帧数
    :param epoch: 当前轮次
    :param save_dir: 保存目录
    """
    import matplotlib.pyplot as plt
    
    # 创建显示图像
    display_img = env_img.copy()
    
    # 绘制Ghost边界框和中心点（红色）
    ghost_boxes = all_game_info.get('4ghosts_boxes', [])
    ghost_centers = all_game_info.get('4ghosts_centers', [])
    for bbox in ghost_boxes:
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for center in ghost_centers:
        if center and len(center) == 2:
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    
    # 绘制Pacman边界框和中心点（绿色）
    pacman_boxes = all_game_info.get('pacman_boxes', [])
    pacman_centers = all_game_info.get('pacman_centers', [])
    for bbox in pacman_boxes:
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for center in pacman_centers:
        if len(center) == 2:
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 4, (0, 255, 0), -1)
    
    # 绘制大力丸（青色）
    superpill_centers = all_game_info.get('superpill_centers', [])
    for center in superpill_centers:
        if len(center) == 2:
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 6, (255, 255, 0), -1)
    
    # 绘制豆子（黄色小点）
    pill_centers = all_game_info.get('pill_centers', [])
    for center in pill_centers:
        if len(center) == 2:
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 2, (0, 255, 255), -1)
    
    # 绘制legal action箭头
    legal_action = all_game_info.get('pacman_decision', {})
    if pacman_centers and legal_action:
        pacman_center = pacman_centers[0]
        cx, cy = pacman_center
        arrow_length = 25
        arrow_color = (0, 255, 0)  # 绿色箭头
        
        if legal_action.get('up', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx), int(cy) - arrow_length), arrow_color, 2, tipLength=0.3)
        if legal_action.get('down', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx), int(cy) + arrow_length), arrow_color, 2, tipLength=0.3)
        if legal_action.get('left', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx) - arrow_length, int(cy)), arrow_color, 2, tipLength=0.3)
        if legal_action.get('right', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx) + arrow_length, int(cy)), arrow_color, 2, tipLength=0.3)
    
    # 添加文字信息
    info_text = [
        f"Frame: {frame}, Epoch: {epoch}",
        f"Stuck Frames: {game_state.stuck_frames}",
        f"Pacman Pos: {game_state.pacman_position}",
        f"Legal Actions: {game_state.get_legal_actions()}",
        f"State: {game_state.state}",
        f"Score: {game_state.score}"
    ]
    
    y_offset = 15
    for i, text in enumerate(info_text):
        cv2.putText(display_img, text, (5, y_offset + i * 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存图片
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"stuck_epoch{epoch}_frame{frame}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    
    # 转换BGR到RGB保存
    cv2.imwrite(filepath, display_img)
    print(f"🚨 Pacman卡住！已保存检测图片: {filepath}")
    print(f"   位置: {game_state.pacman_position}, 停留帧数: {game_state.stuck_frames}")
    print(f"   合法动作: {game_state.get_legal_actions()}")


# ==================== BFS路径规划类 ====================
class PathFinder:
    """
    路径规划器：根据豆子数量选择不同的寻路策略
    - 豆子数量 <= 15: 使用BFS精确搜索
    - 豆子数量 > 15: 使用曼哈顿距离 + 障碍物感知的启发式算法
    """
    
    # 方向定义：(dx, dy, 方向名, 动作编号)
    # 动作编号: 0:NOOP, 1:UP, 2:RIGHT, 3:LEFT, 4:DOWN
    DIRECTIONS = [
        (0, -1, 'up', 1),      # 上：y减小
        (0, 1, 'down', 4),     # 下：y增大
        (-1, 0, 'left', 3),    # 左：x减小
        (1, 0, 'right', 2),    # 右：x增大
    ]
    
    # 豆子数量阈值
    PILL_THRESHOLD = 150
    GLOBAL_SEARCH_PILL_THRESHOLD = 20  # 当豆子数量小于此值时，扩大搜索半径至全局
    
    def __init__(self, game_state, search_radius=5):
        """
        初始化路径规划器
        
        :param game_state: GameState对象
        :param search_radius: BFS搜索时的像素搜索半径（用于判断是否到达目标点）
        """
        self.game_state = game_state
        self.search_radius = search_radius
    
    def find_next_action(self):
        """
        根据当前游戏状态，决定下一步动作
        
        :return: (动作编号, 目标豆子位置, 使用的策略)
                 动作编号: 0:NOOP, 1:UP, 2:RIGHT, 3:LEFT, 4:DOWN
        """
        pacman_pos = self.game_state.get_pacman_pos()
        pill_positions = self.game_state.get_pill_positions()
        superpill_positions = self.game_state.get_superpill_positions()
        
        # 合并所有目标（豆子 + 大力丸）
        all_targets = pill_positions + superpill_positions
        
        # if pacman_pos is None or len(all_targets) == 0:
        #     return 0, None, 'none'  # 无有效目标，保持静止
        
        pill_count = len(pill_positions)
        
        
        
        # 根据豆子数量选择策略
        if pill_count <= self.PILL_THRESHOLD:
            # 豆子较少，使用BFS精确搜索
            return self._bfs_find_path(pacman_pos, all_targets)
        else:
            # 豆子较多，使用启发式算法
            return self._heuristic_find_path(pacman_pos, all_targets)
    
    def _bfs_find_path(self, start_pos, target_positions):
        """
        BFS搜索最短路径到最近的豆子
        
        :param start_pos: 起始位置 (x, y)
        :param target_positions: 目标位置列表 [(x, y), ...]
        :return: (动作编号, 目标位置, 策略名)
        """
        obstacles_mask = self.game_state.obstacles_mask
        if obstacles_mask is None:
            return self._heuristic_find_path(start_pos, target_positions)
        
        height, width = obstacles_mask.shape[:2]
        
        # 获取当前豆子数量，决定搜索策略
        pill_count = len(self.game_state.get_pill_positions())
        use_global_search = pill_count < self.GLOBAL_SEARCH_PILL_THRESHOLD
        
        # 将目标位置转换为集合（使用固定的小半径作为到达判定）
        target_set = set()
        hit_radius = self.search_radius  # 判定到达目标的半径
        for pos in target_positions:
            x, y = int(pos[0]), int(pos[1])
            for dx in range(-hit_radius, hit_radius + 1):
                for dy in range(-hit_radius, hit_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        target_set.add((nx, ny))
        
        # BFS搜索
        start = (int(start_pos[0]), int(start_pos[1]))
        
        # 检查起点是否已经在目标区域
        if start in target_set:
            nearest = self._find_nearest_target(start_pos, target_positions)
            if nearest and manhattan_distance(start_pos, nearest) > hit_radius:
                target_positions = [t for t in target_positions if t != nearest]
                if target_positions:
                    return self._bfs_find_path(start_pos, target_positions)
            return 0, start_pos, 'bfs_at_target'
        
        # BFS队列: (当前位置, 第一步方向, 第一步动作, 路径长度)
        queue = deque()
        visited = set()
        visited.add(start)
        
        # 获取需要排除的回头方向
        opposite_direction = self.game_state.get_opposite_direction()
        legal_actions_no_backtrack = self.game_state.get_legal_actions_no_backtrack()
        
        # 初始化：将起点的有效方向加入队列
        for dx, dy, direction, action in self.DIRECTIONS:
            if direction == opposite_direction and len(legal_actions_no_backtrack) > 0:
                if direction not in legal_actions_no_backtrack:
                    continue
            
            nx, ny = start[0] + dx, start[1] + dy
            if self._is_valid_position(nx, ny, obstacles_mask):
                queue.append(((nx, ny), direction, action, 1))
                visited.add((nx, ny))
        
        # 设置搜索深度限制
        # 豆子数量少时，不限制搜索深度（全局搜索）
        # 豆子数量多时，限制搜索深度以提高效率
        max_search_depth = float('inf') if use_global_search else 200
        
        # BFS搜索
        while queue:
            (cx, cy), first_direction, first_action, dist = queue.popleft()
            
            # 检查是否到达目标
            if (cx, cy) in target_set:
                target_pos = self._find_nearest_target((cx, cy), target_positions)
                return first_action, target_pos, 'bfs_global' if use_global_search else 'bfs'
            
            # 检查搜索深度限制
            if dist >= max_search_depth:
                continue
            
            # 扩展邻居节点
            for dx, dy, _, _ in self.DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and self._is_valid_position(nx, ny, obstacles_mask):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), first_direction, first_action, dist + 1))
        
        # BFS找不到路径，退化为启发式
        return self._heuristic_find_path(start_pos, target_positions)
    
    def _heuristic_find_path(self, start_pos, target_positions):
        """
        启发式算法：曼哈顿距离 + 障碍物感知 + 方向连续性
        
        策略：
        1. 计算到所有豆子的曼哈顿距离
        2. 根据障碍物情况对距离进行惩罚
        3. 选择"有效距离"最小的豆子作为目标
        4. 根据目标方向和合法动作选择最佳动作（排除回头路）
        
        :param start_pos: 起始位置 (x, y)
        :param target_positions: 目标位置列表 [(x, y), ...]
        :return: (动作编号, 目标位置, 策略名)
        """
        obstacles_mask = self.game_state.obstacles_mask
        
        # 使用排除回头路后的合法动作
        legal_actions = self.game_state.get_legal_actions_no_backtrack()
        
        if not legal_actions:
            # 如果排除回头路后没有合法动作，使用原始合法动作
            legal_actions = self.game_state.get_legal_actions()
        
        if not legal_actions:
            return 0, None, 'heuristic_no_action'
        
        # 计算每个豆子的有效距离（曼哈顿距离 + 障碍物惩罚）
        best_target = None
        best_score = float('inf')
        
        for target in target_positions:
            # 基础曼哈顿距离
            base_dist = manhattan_distance(start_pos, target)
            
            # 障碍物惩罚：检查直线路径上的障碍物
            obstacle_penalty = self._calculate_obstacle_penalty(start_pos, target, obstacles_mask)
            
            # Ghost惩罚：如果路径靠近Ghost，增加惩罚
            ghost_penalty = self._calculate_ghost_penalty(target)
            
            # 综合评分（距离 + 惩罚）
            total_score = base_dist + obstacle_penalty * 2 + ghost_penalty * 3
            
            if total_score < best_score:
                best_score = total_score
                best_target = target
        
        if best_target is None:
            return 0, None, 'heuristic_no_target'
        
        # 根据目标位置选择最佳动作（使用排除回头路后的合法动作）
        best_action = self._select_action_towards_target(start_pos, best_target, legal_actions)
        
        return best_action, best_target, 'heuristic'
    
    def _is_valid_position(self, x, y, obstacles_mask):
        """
        检查位置是否有效（在边界内且不是障碍物）
        
        :param x: x坐标
        :param y: y坐标
        :param obstacles_mask: 障碍物掩码
        :return: 是否有效
        """
        height, width = obstacles_mask.shape[:2]
        
        # 检查边界
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        
        # 检查障碍物（障碍物掩码中非零值表示障碍物）
        # 注意：obstacles_mask的索引是[y, x]
        if obstacles_mask[int(y), int(x)] > 0:
            return False
        
        return True
    
    def _calculate_obstacle_penalty(self, start, target, obstacles_mask):
        """
        计算从起点到目标的直线路径上的障碍物惩罚
        
        :param start: 起点 (x, y)
        :param target: 目标点 (x, y)
        :param obstacles_mask: 障碍物掩码
        :return: 惩罚值
        """
        if obstacles_mask is None:
            return 0
        
        # 使用Bresenham算法采样直线路径上的点
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(target[0]), int(target[1])
        
        height, width = obstacles_mask.shape[:2]
        
        # 简化：只检查几个关键点
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            return 0
        
        penalty = 0
        sample_count = min(10, steps)  # 最多采样10个点
        
        for i in range(1, sample_count + 1):
            t = i / (sample_count + 1)
            check_x = int(x0 + (x1 - x0) * t)
            check_y = int(y0 + (y1 - y0) * t)
            
            if 0 <= check_x < width and 0 <= check_y < height:
                if obstacles_mask[check_y, check_x] > 0:
                    penalty += 10  # 每遇到一个障碍物点，增加惩罚
        
        return penalty
    
    def _calculate_ghost_penalty(self, target):
        """
        计算目标点附近Ghost带来的惩罚
        
        :param target: 目标点 (x, y)
        :return: 惩罚值
        """
        ghost_positions = self.game_state.get_ghost_positions()
        penalty = 0
        
        for ghost_pos in ghost_positions:
            dist = manhattan_distance(target, ghost_pos)
            # 如果Ghost在目标附近，增加惩罚
            if dist < 30:
                penalty += (30 - dist)  # 距离越近惩罚越大
        
        return penalty
    
    def _select_action_towards_target(self, start, target, legal_actions):
        """
        选择朝向目标的最佳合法动作
        
        :param start: 起点 (x, y)
        :param target: 目标点 (x, y)
        :param legal_actions: 合法动作列表 ['up', 'down', 'left', 'right']
        :return: 动作编号
        """
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        
        # 根据目标方向确定优先动作
        preferred_actions = []
        
        # 水平方向
        if dx > 0:
            preferred_actions.append(('right', 2))
        elif dx < 0:
            preferred_actions.append(('left', 3))
        
        # 垂直方向
        if dy > 0:
            preferred_actions.append(('down', 4))
        elif dy < 0:
            preferred_actions.append(('up', 1))
        
        # 按照距离差的绝对值排序，优先选择差距大的方向
        if abs(dx) >= abs(dy):
            # 水平距离更大，优先水平移动
            preferred_actions.sort(key=lambda x: 0 if x[0] in ['left', 'right'] else 1)
        else:
            # 垂直距离更大，优先垂直移动
            preferred_actions.sort(key=lambda x: 0 if x[0] in ['up', 'down'] else 1)
        
        # 选择第一个合法的优先动作
        for action_name, action_num in preferred_actions:
            if action_name in legal_actions:
                return action_num
        
        # 如果没有优先动作可用，随机选择一个合法动作
        action_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
        for action_name in legal_actions:
            if action_name in action_map:
                return action_map[action_name]
        
        return 0  # 无合法动作，保持静止
    
    def _find_nearest_target(self, pos, targets):
        """
        找到最近的目标点
        
        :param pos: 当前位置 (x, y)
        :param targets: 目标位置列表
        :return: 最近的目标位置
        """
        if not targets:
            return None
        
        nearest = None
        min_dist = float('inf')
        
        for target in targets:
            dist = manhattan_distance(pos, target)
            if dist < min_dist:
                min_dist = dist
                nearest = target
        
        return nearest
    
    def get_strategy_name(self, pill_count):
        """
        根据豆子数量返回将使用的策略名称
        
        :param pill_count: 豆子数量
        :return: 策略名称
        """
        if pill_count < self.GLOBAL_SEARCH_PILL_THRESHOLD:
            return "BFS全局搜索（无深度限制）"
        elif pill_count <= self.PILL_THRESHOLD:
            return "BFS精确搜索（深度限制200）"
        else:
            return "启发式搜索（曼哈顿距离+障碍物感知）"


# ==================== 动作决策函数 ====================
def decide_next_action(game_state, args, env_img=None, frame=0, epoch=0):
    """
    决定下一步动作的主函数
    
    :param game_state: GameState对象
    :param args: 配置参数
    :param env_img: 当前帧图像（用于VLM调用，可选）
    :param frame: 当前帧数
    :param epoch: 当前轮次
    :return: (动作编号, 目标位置, 策略, 是否危险)
    """
    # 首先检查是否处于chase状态（吃掉大力丸后可以追击Ghost）
    should_chase_ghost, ghost_pos, ghost_dist = game_state.should_chase_ghost(args.ghost_chase_threshold)
    
    if should_chase_ghost and ghost_pos is not None:
        # 处于chase状态且Ghost距离足够近，主动追击Ghost
        # print(f"👻 追击Ghost！距离: {ghost_dist}")
        path_finder = PathFinder(game_state)
        # 将Ghost作为目标，使用BFS搜索
        action, target, strategy = path_finder._bfs_find_path(
            game_state.get_pacman_pos(),
            [ghost_pos]
        )
        if action != 0:
            return action, ghost_pos, 'chase_ghost', False
    
    # 检查是否面临多Ghost威胁（废弃VLM，改用自己的逃跑逻辑）
    is_multi_danger, nearby_count, ghost_distances = game_state.is_multi_ghost_danger(
        threshold=args.ghost_danger_threshold,
        min_ghost_count=2
    )
    
    if is_multi_danger and game_state.state != 'chase':
        # 多Ghost围堵，使用逃跑逻辑（选择远离所有Ghost的方向）
        escape_action = _get_escape_action_multi_ghost(game_state, ghost_distances)
        return escape_action, None, 'escape_multi_ghost', True
    
    # 检查是否处于危险状态（非chase状态下Ghost太近，但只有一个Ghost）
    is_danger, ghost_dist, nearest_ghost = game_state.is_in_danger(args.ghost_danger_threshold)
    
    if is_danger and game_state.state != 'chase':
        # 单Ghost危险，使用简单逃跑逻辑
        escape_action = _get_escape_action(game_state, nearest_ghost)
        return escape_action, None, 'escape', True
    
    # 检查是否应该追逐大力丸
    should_chase, superpill_pos, sp_dist, gh_dist = game_state.should_chase_superpill(
        args.superpill_chase_threshold,
        args.superpill_safe_margin
    )
    
    if should_chase and superpill_pos is not None:
        # 应该追逐大力丸，使用BFS路径规划直接找到大力丸
        # print(f"🔥 追逐大力丸！距离: {sp_dist}, Ghost距离: {gh_dist}")
        path_finder = PathFinder(game_state)
        # 只将大力丸作为目标，使用BFS搜索
        action, target, strategy = path_finder._bfs_find_path(
            game_state.get_pacman_pos(), 
            [superpill_pos]
        )
        if action != 0:
            return action, superpill_pos, 'chase_superpill', False
    
    # 检查豆子数量是否少于20，如果是则调用VLM引导找豆子
    # 条件：豆子数量 <= 20，帧数 >= 300，有图像，且有豆子
    LOW_PILL_THRESHOLD = 20  # 触发VLM引导的豆子数量阈值
    LOW_PILL_VLM_INTERVAL = 30  # VLM引导的帧间隔（避免过于频繁调用）
    
    if (game_state.pill_num <= LOW_PILL_THRESHOLD and 
        game_state.pill_num > 0 and 
        env_img is not None and 
        frame >= 300 and
        frame % LOW_PILL_VLM_INTERVAL == 0):  # 每30帧调用一次VLM
        # 获取4vlm标注图片路径
        vlm_image_path = get_4vlm_image_path(args, epoch, frame)
        
        # 豆子数量少，调用VLM引导找豆子
        action = vlm_decide_action(
            game_state, env_img,
            scenario='low_pill',
            args=args,
            save_dir="vlm_debug",
            vlm_image_path=vlm_image_path
        )
        if action != 0:
            return action, None, 'vlm_low_pill', False
    
    # 非危险状态，使用路径规划寻找豆子
    path_finder = PathFinder(game_state)
    action, target, strategy = path_finder.find_next_action()
    
    return action, target, strategy, False


def _get_escape_action(game_state, ghost_pos):
    """
    获取逃跑动作（远离Ghost的方向）
    
    :param game_state: GameState对象
    :param ghost_pos: 最近Ghost的位置
    :return: 动作编号
    """
    pacman_pos = game_state.get_pacman_pos()
    legal_actions = game_state.get_legal_actions()
    
    if pacman_pos is None or ghost_pos is None or not legal_actions:
        return 0
    
    px, py = pacman_pos
    gx, gy = ghost_pos
    
    # 计算每个合法方向移动后与Ghost的距离
    action_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
    direction_delta = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }
    
    best_action = 0
    max_distance = -1
    
    for direction in legal_actions:
        if direction not in action_map:
            continue
        
        dx, dy = direction_delta[direction]
        # 计算移动后的新位置
        new_px, new_py = px + dx * 5, py + dy * 5  # 假设移动5个像素
        # 计算新位置与Ghost的曼哈顿距离
        new_distance = abs(new_px - gx) + abs(new_py - gy)
        
        if new_distance > max_distance:
            max_distance = new_distance
            best_action = action_map[direction]
    
    return best_action if best_action != 0 else 0


def _get_escape_action_multi_ghost(game_state, ghost_distances):
    """
    获取逃跑动作（远离多个Ghost的方向）
    
    策略：选择移动后与所有Ghost距离之和最大的方向
    
    :param game_state: GameState对象
    :param ghost_distances: Ghost距离列表 [(pos, dist), ...]
    :return: 动作编号
    """
    pacman_pos = game_state.get_pacman_pos()
    legal_actions = game_state.get_legal_actions()
    
    if pacman_pos is None or not ghost_distances or not legal_actions:
        return 0
    
    px, py = pacman_pos
    
    action_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
    direction_delta = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }
    
    best_action = 0
    max_total_distance = -1
    
    for direction in legal_actions:
        if direction not in action_map:
            continue
        
        dx, dy = direction_delta[direction]
        # 计算移动后的新位置
        new_px, new_py = px + dx * 5, py + dy * 5
        
        # 计算新位置与所有Ghost的距离之和
        total_distance = 0
        min_ghost_dist = float('inf')  # 同时跟踪最近Ghost的距离
        
        for ghost_pos, _ in ghost_distances:
            gx, gy = ghost_pos
            dist = abs(new_px - gx) + abs(new_py - gy)
            total_distance += dist
            min_ghost_dist = min(min_ghost_dist, dist)
        
        # 综合评分：总距离 + 最近Ghost距离的权重
        # 这样可以避免选择虽然总距离大但离某个Ghost很近的方向
        score = total_distance + min_ghost_dist * 2
        
        if score > max_total_distance:
            max_total_distance = score
            best_action = action_map[direction]
    
    return best_action if best_action != 0 else 0


def single_action(env, action_num, duration):
    """
    执行单个动作持续一定帧数
    
    :param env: 游戏环境
    :param action_num: 动作编号 (0:NOOP, 1:UP, 2:RIGHT, 3:LEFT, 4:DOWN)
    :param duration: 持续帧数
    :return: observation, reward, terminated, truncated, info
    """
    obs = None
    total_reward = 0
    for _ in range(duration):
        obs, reward, terminated, truncated, info = env.step(action_num)
        total_reward += reward
        if terminated or truncated:
            break
    return obs, total_reward, terminated, truncated, info


# ==================== 主游戏循环（示例） ====================
def initialize_game():
    """
    初始化游戏环境和相关变量
    
    :return: env, args, model, game_state
    """
    # 创建配置
    args = GameArgs()
    
    # 创建游戏环境
    env = gym.make(args.game_name, render_mode='human')
    
    # 加载YOLO模型（关闭verbose输出）
    model = YOLO(args.path, verbose=False)
    
    # 创建游戏状态对象
    game_state = GameState()
    
    print(f"游戏环境 {args.game_name} 初始化完成")
    print(f"YOLO模型加载自: {args.path}")
    
    return env, args, model, game_state


def update_game_state(env_img, args, epoch, frame, former_all_game_info, model, game_state, if_4vlm=False):
    """
    更新游戏状态
    
    :param env_img: 当前帧图像
    :param args: 配置参数
    :param epoch: 当前轮次
    :param frame: 当前帧数
    :param former_all_game_info: 上一帧的游戏信息
    :param model: YOLO模型
    :param game_state: 游戏状态对象
    :param if_4vlm: 是否生成VLM专用的标注图片
    :return: 更新后的all_game_info字典
    """
    # 调用detect_all_in_one获取所有游戏信息
    all_game_info = detect_all_in_one(
        env_img,
        args,
        epoch,
        frame,
        former_all_game_info,
        model=model,
        if_4vlm=if_4vlm
    )
    
    # 更新GameState对象
    game_state.update_from_detect_all(all_game_info, frame, epoch)
    
    return all_game_info


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 初始化游戏
    env, args, model, game_state = initialize_game()
    
    # 重置环境
    observation, info = env.reset()
    
    # 游戏循环变量
    frame = 0
    epoch = 0
    former_all_game_info = None
    last_HP = 3                        # 上一帧的HP值，用于检测掉命（初始为3条命）
    
    # ========== 决策间隔控制 ==========
    DECISION_INTERVAL = 4          # 决策间隔：每隔多少帧重新调用一次decide_next_action
    current_action = 0             # 当前执行的动作
    current_target = None           # 当前目标
    current_strategy = 'none'       # 当前策略
    frames_since_decision = 0       # 距离上次决策的帧数
    # ==================================
    
    print("开始游戏循环测试...")
    print("=" * 60)
    print("策略说明:")
    print(f"  - 豆子数量 < {PathFinder.GLOBAL_SEARCH_PILL_THRESHOLD}: BFS全局搜索（无深度限制，搜索整个地图）")
    print(f"  - 豆子数量 <= {PathFinder.PILL_THRESHOLD}: BFS精确搜索（深度限制200）")
    print(f"  - 豆子数量 > {PathFinder.PILL_THRESHOLD}: 启发式搜索")
    print(f"  - Ghost距离 < {args.ghost_danger_threshold}: 触发危险模式/逃跑")
    print(f"  - 决策间隔: 每 {DECISION_INTERVAL} 帧重新决策一次")
    print("=" * 60)
    
    try:
        # 先执行一个空动作让游戏开始
        observation, _, terminated, truncated, _ = single_action(env, 0, 10)
        
        # VLM触发阈值常量
        LOW_PILL_THRESHOLD = 20  # 触发VLM引导的豆子数量阈值
        LOW_PILL_VLM_INTERVAL = 30  # VLM引导的帧间隔
        
        # 游戏主循环
        while True:
            # 转换图像格式
            image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            
            # 判断是否需要生成VLM标注图片
            # 条件1: 卡住检测可能触发VLM（基于上一帧的stuck状态）
            # 条件2: 豆子数量少于阈值且满足VLM调用间隔
            need_vlm_image = False
            if frame >= 300:
                # 检查是否可能卡住（使用上一帧的stuck状态判断）
                if game_state.stuck_frames >= game_state.stuck_threshold - 1:
                    need_vlm_image = True
                # 检查是否豆子数量少
                if (game_state.pill_num <= LOW_PILL_THRESHOLD and 
                    game_state.pill_num > 0 and
                    frame % LOW_PILL_VLM_INTERVAL == 0):
                    need_vlm_image = True
            
            # 更新游戏状态（根据需要生成VLM图片）
            all_game_info = update_game_state(
                image_bgr, args, epoch, frame,
                former_all_game_info, model, game_state,
                if_4vlm=need_vlm_image
            )
            
            # 输出剩余豆子数量（每50帧输出一次，减少刷屏）
            if frame % 50 == 0:
                print(f"[Frame {frame}] 剩余豆子: {game_state.pill_num}")

            # ========== 掉命检测：HP减少时重置frame ==========
            current_HP = game_state.HP
            if current_HP < last_HP:
                print(f"💀 Pacman掉命！HP: {last_HP} -> {current_HP}，重置frame为0")
                frame = 0
                # 重置卡住检测
                game_state.reset_stuck_detection()
                # 重置决策间隔计数器
                frames_since_decision = DECISION_INTERVAL
            last_HP = current_HP
            # ================================================

            # ========== 卡住检测与VLM介入 ==========
            is_stuck, stuck_frames = game_state.check_stuck()
            vlm_stuck_action = None
            
            if is_stuck and stuck_frames >= game_state.stuck_threshold and frame >= 300:
                # 达到卡住阈值且帧数>=300，保存检测图片并调用VLM
                # 获取在该位置已尝试过的动作
                pacman_pos = game_state.get_pacman_pos()
                tried_actions = game_state.get_stuck_tried_actions(pacman_pos)
                
                # 每次达到阈值时保存图片（只在刚达到阈值时保存，避免重复）
                if stuck_frames == game_state.stuck_threshold:
                    save_stuck_detection_image(
                        image_bgr, all_game_info, game_state, 
                        frame, epoch, save_dir="stuck_detection"
                    )
                
                # 获取4vlm标注图片路径
                vlm_image_path = get_4vlm_image_path(args, epoch, frame)
                
                # 调用VLM决策，传入已尝试过的动作和4vlm图片路径
                vlm_stuck_action = vlm_decide_action(
                    game_state, image_bgr,
                    scenario='stuck',
                    args=args,
                    save_dir="vlm_debug",
                    tried_actions=tried_actions,
                    vlm_image_path=vlm_image_path
                )
                
                # 记录本次VLM决策的动作
                if vlm_stuck_action is not None and vlm_stuck_action != 0:
                    game_state.add_stuck_tried_action(pacman_pos, vlm_stuck_action)
            # ================================
            
            # ========== 决策间隔控制逻辑 ==========
            # 检查是否需要重新决策
            need_new_decision = (
                frames_since_decision >= DECISION_INTERVAL or  # 达到间隔
                frame == 0 or                                  # 第一帧
                vlm_stuck_action is not None                   # VLM介入（卡住）
            )
            
            if need_new_decision:
                if vlm_stuck_action is not None:
                    # 使用VLM的卡住决策
                    action = vlm_stuck_action
                    target = None
                    strategy = 'vlm_stuck'
                    is_danger = False
                    # 重置卡住检测，给VLM决策一些执行时间
                    game_state.reset_stuck_detection()
                else:
                    # 正常决策（传入图像、帧数和轮次以支持VLM调用）
                    action, target, strategy, is_danger = decide_next_action(game_state, args, env_img=image_bgr, frame=frame, epoch=epoch)
                
                current_action = action
                current_target = target
                current_strategy = strategy
                frames_since_decision = 0  # 重置计数器
                
                # 打印决策信息（只在重新决策时打印，且每10次决策打印一次）
                if frame % (DECISION_INTERVAL * 10) == 0 or strategy.startswith('vlm'):
                    pill_count = game_state.pill_num
                    print(f"[Frame {frame}] 策略: {strategy}, 动作: {action}, 豆子: {pill_count}")
            else:
                # 继续使用上次的决策
                action = current_action
                frames_since_decision += 1
            # =====================================
            
            # 更新历史信息
            former_all_game_info = all_game_info
            frame += 1
            
            # 记录本次执行的动作（用于下一帧排除回头路）
            game_state.set_last_action(action)
            
            # 执行决策的动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 只在获得较大奖励时打印
            if reward >= 10:
                print(f"  🎉 获得奖励: {reward}")
            
            # 检查游戏是否结束
            if terminated or truncated:
                print("=" * 60)
                print(f"游戏结束！最终得分: {game_state.score}")
                print("重新开始...")
                print("=" * 60)
                observation, info = env.reset()
                epoch += 1
                frame = 0
                former_all_game_info = None
                last_HP = 3  # 重置HP记录
                # 重置上一步动作记录
                game_state.set_last_action(None)
                # 重置决策间隔计数器
                frames_since_decision = DECISION_INTERVAL  # 确保下一帧立即决策
                # 重置卡住检测
                game_state.reset_stuck_detection()
                # 清空卡住历史记录（新游戏开始）
                game_state.clear_stuck_history()
            
            # 移除sleep，让画面尽可能流畅
            # time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n用户中断游戏")
    finally:
        env.close()
        print("游戏环境已关闭")