import gymnasium as gym
import ale_py
import time
import cv2
import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import dashscope
from dashscope import MultiModalConversation
import re
import numpy as np
import random
import math
from ultralytics import YOLO

from detect_all import detect_all_in_one, crop_image, find_label, detect_score, detect_HP
from utils_all.game_utils import create_pacman_environment

from sklearn.neighbors import KNeighborsClassifier as Agent
import warnings
warnings.filterwarnings('ignore')

class PacmanState:
    def __init__(self, pacman_x, pacman_y, ghost_distance, superpill_distance, pill_number, decision, scared_ghost, score):
        self.pacman_x = pacman_x
        self.pacman_y = pacman_y
        self.ghost_distance = ghost_distance
        self.superpill_distance = superpill_distance
        self.pill_number = pill_number
        self.decision = decision
        self.scared_ghost = scared_ghost
        self.score = score
        self.next_action = 0

class PacmanEvaluator:
    def __init__(self):
        self.weight = np.load('weight.npy')
        self.agent = Agent(n_neighbors=1).fit(self.weight[:,:8], self.weight[:, 8:])
        
    def evaluate_state(self, state: PacmanState) -> float:
        value_list = self.agent.predict(np.array([[state.pacman_x, state.pacman_y, state.ghost_distance, state.superpill_distance, state.pill_number, state.decision, state.scared_ghost, state.score]]))
        v = 1
        if (int(value_list[0, state.next_action-1]) >= 0.25):
            T = 1
        elif ((state.decision >> (state.next_action-1)) & 0b1 == 0):
            T = 1e-2
        else:
            T = 1e-1
        
        return T*int(value_list[0, 4])

class PacmanRLReasoner:
    def __init__(self, evaluator: PacmanEvaluator = None):
        self.evaluator = evaluator or PacmanEvaluator()
        self.action_space = [1, 4, 3, 2]
    
    def choose_action(self, state: PacmanState):
        best_action = 0
        best_value = -1000
        
        for action in self.action_space:
            state.next_action = action
            
            state_value = self.evaluator.evaluate_state(state)
            
            if state_value > best_value:
                best_action = action
                best_value = state_value
        
        return [int(best_action), int(best_value)]

def create_state(obs, former_all_game_info):
    if len(former_all_game_info["pacman_centers"]) == 0:
        pacman_x = 0
        pacman_y = 0
    else:
        pacman_x, pacman_y = former_all_game_info["pacman_centers"][0]

    ghost_distance = 1000
    for i in range(1, 5):
        distance = abs(former_all_game_info["ghosts_centers"][i][0]-pacman_x) + abs(former_all_game_info["ghosts_centers"][i][1]-pacman_y)
        if (distance < ghost_distance):
            ghost_distance = distance

    superpill_distance = 1000
    for i in range(len(former_all_game_info["superpill_centers"])):
        distance = abs(former_all_game_info["superpill_centers"][i][0]-pacman_x) + abs(former_all_game_info["superpill_centers"][i][1]-pacman_y)
        if (distance < superpill_distance):
            superpill_distance = distance

    pill_number = former_all_game_info["pill_num"][0]

    decision = 0
    decision += former_all_game_info["pacman_decision"]["up"]
    decision <<= 2
    decision += former_all_game_info["pacman_decision"]["right"]
    decision <<= 2
    decision += former_all_game_info["pacman_decision"]["left"]
    decision <<= 2
    decision += former_all_game_info["pacman_decision"]["down"]
        
    image_rgb = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    mask = (image_rgb[:, :, 0] == 66) & (image_rgb[:, :, 1] == 114) & (image_rgb[:, :, 2] == 194)
    scared_ghost = int(np.sum(mask))

    score = former_all_game_info["score"] % 3000

    state = PacmanState(pacman_x, pacman_y, ghost_distance, superpill_distance, pill_number, decision, scared_ghost, score)

    return state

def right_situation(all_game_info):
    return not((1000+all_game_info["score"]) >> 13)

gym.register_envs(ale_py)

dashscope.api_key = "sk-361f43ece66a49e299a35ef26ac687d7"#wangjun

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
    
    if image_path:
        user_content = [
            {"image": f"file://{image_path}"},
            {"text": prompt}
        ]
    else:
        user_content = [{"text": prompt}]
    
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
    numbers = re.findall(r'\b\d+\.?\d*\b', vlm_response)
    
    if numbers:
        action = int(float(numbers[0]))
        if 0 <= action <= 4:
            return action
    
    return 0 

class GameArgs:
    def __init__(self):
        self.size = 256  
        self.visualize_save = False  
        self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"  
        self.your_mission_name = "BFS_Mission"  
        self.game_name = 'MsPacmanNoFrameskip-v4'  
        self.vlm = 'qwen-vl-plus' 
        self.ghost_danger_threshold = 20  
        self.superpill_chase_threshold = 50  
        self.superpill_safe_margin = 50      
        self.ghost_chase_threshold = 100  

# ==================== 游戏信息存储类 ====================
class GameState:
    OPPOSITE_DIRECTION = {
        'up': 'down',
        'down': 'up',
        'left': 'right',
        'right': 'left'
    }
    
    ACTION_TO_DIRECTION = {
        0: None,     
        1: 'up',
        2: 'right',
        3: 'left',
        4: 'down'
    }
    
    def __init__(self):
        self.pacman_boxes = []          # pacman边界框 [[x1, y1, x2, y2], ...]
        self.pacman_centers = []        # pacman中心点 [[x, y], ...]
        self.pacman_position = None     # 当前pacman位置 (x, y)
        self.ghosts_boxes = []          # 所有ghost边界框
        self.ghosts_centers = []        # 所有ghost中心点
        self.four_ghosts_boxes = []     # 4个ghost的边界框（用于算法）
        self.four_ghosts_centers = []   # 4个ghost的中心点（用于算法）
        self.ghost_num = 0              # ghost数量
        self.pill_centers = []          # 所有豆子中心点 [[x, y], ...]
        self.pill_num = 0               # 豆子数量
        self.superpill_boxes = []       # 大力丸边界框
        self.superpill_centers = []     # 大力丸中心点
        self.superpill_info = None      # 大力丸完整信息
        self.door_centers = []          # 传送门中心点
        self.obstacles_mask = None      # 障碍物掩码（二值图像，用于路径规划）
        self.pacman_decision = {}       # 可行动方向 {'up': 1/0, 'down': 1/0, 'left': 1/0, 'right': 1/0}
        self.legal_action_num = 0       # 可行动方向数量
        self.last_action = None         # 上一步执行的动作编号
        self.last_direction = None      # 上一步的方向名 ('up', 'down', 'left', 'right')
        self.score = 0                  # 当前得分
        self.HP = 0                     # 当前生命值
        self.state = 'init'             # 游戏状态: 'init'(初始化), 'run'(逃跑), 'chase'(追击)
        self.frame = 0                  # 当前帧数
        self.epoch = 0                  # 游戏轮次（死亡/吃完豆子后重置）
        self.stuck_position = None       # 记录可能卡住的位置
        self.stuck_frames = 0            # 在该位置停留的帧数
        self.stuck_threshold = 30        # 判定为卡住的帧数阈值
        self.stuck_distance = 5          # 判定为同一位置的距离阈值
        self.stuck_vlm_history = {}      # {(x, y): [已尝试的动作列表]}
        self.stuck_history_distance = 10  # 判定为同一卡住位置的距离阈值
    
    def set_last_action(self, action):
        self.last_action = action
        self.last_direction = self.ACTION_TO_DIRECTION.get(action, None)
    
    def get_opposite_direction(self):
        if self.last_direction is None:
            return None
        return self.OPPOSITE_DIRECTION.get(self.last_direction, None)
    
    def get_legal_actions_no_backtrack(self):
        legal = self.get_legal_actions()
        opposite = self.get_opposite_direction()
        
        if opposite and opposite in legal and len(legal) > 1:
            legal = [a for a in legal if a != opposite]
        
        return legal
    
    def update_from_detect_all(self, all_game_info, frame, epoch):
        self.frame = frame
        self.epoch = epoch
        self.pacman_boxes = all_game_info.get('pacman_boxes', [])
        self.pacman_centers = all_game_info.get('pacman_centers', [])
        if self.pacman_centers and len(self.pacman_centers) > 0:
            self.pacman_position = tuple(self.pacman_centers[0])  
        self.ghosts_boxes = all_game_info.get('ghosts_boxes', [])
        self.ghosts_centers = all_game_info.get('ghosts_centers', [])
        self.four_ghosts_boxes = all_game_info.get('4ghosts_boxes', [])
        self.four_ghosts_centers = all_game_info.get('4ghosts_centers', [])
        self.ghost_num = all_game_info.get('ghost_num', 0)
        self.pill_centers = all_game_info.get('pill_centers', [])
        pill_num_list = all_game_info.get('pill_num', [0])
        self.pill_num = pill_num_list[0] if isinstance(pill_num_list, list) else pill_num_list
        self.superpill_boxes = all_game_info.get('superpill_boxes', [])
        self.superpill_centers = all_game_info.get('superpill_centers', [])
        self.superpill_info = all_game_info.get('superpill_info', None)
        self.door_centers = all_game_info.get('door_centers', [])
        self.obstacles_mask = all_game_info.get('obstacles_mask', None)
        self.pacman_decision = all_game_info.get('pacman_decision', {})
        self.legal_action_num = all_game_info.get('legal_action_num', 0)
        self.score = all_game_info.get('score', 0)
        self.HP = all_game_info.get('HP', 0)
        self.state = all_game_info.get('state', 'init')
    
    def get_pacman_pos(self):
        return self.pacman_position
    
    def get_ghost_positions(self):
        positions = []
        for center in self.four_ghosts_centers:
            if center is not None and len(center) == 2:
                positions.append(tuple(center))
        return positions
    
    def get_pill_positions(self):
        return [tuple(center) for center in self.pill_centers if center and len(center) == 2]
    
    def get_superpill_positions(self):
        return [tuple(center) for center in self.superpill_centers if center and len(center) == 2]
    
    def get_legal_actions(self):
        legal = []
        for direction, is_legal in self.pacman_decision.items():
            if is_legal == 1:
                legal.append(direction)
        return legal
    
    def is_in_danger(self, threshold=30):
        if self.pacman_position is None:
            return False, float('inf'), None
        
        pacman_x, pacman_y = self.pacman_position
        min_distance = float('inf')
        nearest_ghost = None
        
        for ghost_pos in self.get_ghost_positions():
            ghost_x, ghost_y = ghost_pos
            distance = abs(pacman_x - ghost_x) + abs(pacman_y - ghost_y)
            if distance < min_distance:
                min_distance = distance
                nearest_ghost = ghost_pos
        
        return min_distance < threshold, min_distance, nearest_ghost
    
    def count_nearby_ghosts(self, threshold=30):
        if self.pacman_position is None:
            return 0, []
        
        pacman_x, pacman_y = self.pacman_position
        ghost_distances = []
        
        for ghost_pos in self.get_ghost_positions():
            ghost_x, ghost_y = ghost_pos
            distance = abs(pacman_x - ghost_x) + abs(pacman_y - ghost_y)
            if distance < threshold:
                ghost_distances.append((ghost_pos, distance))
        
        ghost_distances.sort(key=lambda x: x[1])
        
        return len(ghost_distances), ghost_distances
    
    def is_multi_ghost_danger(self, threshold=30, min_ghost_count=2):
        nearby_count, ghost_distances = self.count_nearby_ghosts(threshold)
        is_multi_danger = nearby_count >= min_ghost_count
        return is_multi_danger, nearby_count, ghost_distances
    
    def should_chase_superpill(self, chase_threshold=50, safe_margin=20):
        superpill_positions = self.get_superpill_positions()
        
        if not superpill_positions:
            return False, None, float('inf'), float('inf')
        
        pacman_pos = self.get_pacman_pos()
        if pacman_pos is None:
            return False, None, float('inf'), float('inf')
        
        min_superpill_dist = float('inf')
        nearest_superpill = None
        for sp_pos in superpill_positions:
            distance = manhattan_distance(pacman_pos, sp_pos)
            if distance < min_superpill_dist:
                min_superpill_dist = distance
                nearest_superpill = sp_pos
        
        _, ghost_dist, _ = self.is_in_danger(threshold=float('inf'))
        
        should_chase = (
            min_superpill_dist < chase_threshold and
            ghost_dist > min_superpill_dist + safe_margin
        )
        
        return should_chase, nearest_superpill, min_superpill_dist, ghost_dist
    
    def should_chase_ghost(self, chase_threshold=60):
        if self.state != 'chase':
            return False, None, float('inf')
        
        pacman_pos = self.get_pacman_pos()
        if pacman_pos is None:
            return False, None, float('inf')
        
        ghost_positions = self.get_ghost_positions()
        if not ghost_positions:
            return False, None, float('inf')
        
        min_ghost_dist = float('inf')
        nearest_ghost = None
        for ghost_pos in ghost_positions:
            distance = manhattan_distance(pacman_pos, ghost_pos)
            if distance < min_ghost_dist:
                min_ghost_dist = distance
                nearest_ghost = ghost_pos
        
        should_chase = min_ghost_dist < chase_threshold
        
        return should_chase, nearest_ghost, min_ghost_dist
    
    def print_state(self):
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
        current_pos = self.get_pacman_pos()
        
        if current_pos is None:
            return False, 0
        
        if self.stuck_position is None:
            self.stuck_position = current_pos
            self.stuck_frames = 1
            return False, 1
        
        distance = manhattan_distance(current_pos, self.stuck_position)
        
        if distance <= self.stuck_distance:
            self.stuck_frames += 1
        else:
            self.stuck_position = current_pos
            self.stuck_frames = 1
        
        is_stuck = self.stuck_frames >= self.stuck_threshold
        return is_stuck, self.stuck_frames
    
    def reset_stuck_detection(self):
        self.stuck_position = None
        self.stuck_frames = 0
    
    def get_stuck_history_key(self, pos):
        if pos is None:
            return None
        # 将位置量化到10像素的网格
        grid_size = self.stuck_history_distance
        qx = int(pos[0] // grid_size) * grid_size
        qy = int(pos[1] // grid_size) * grid_size
        return (qx, qy)
    
    def get_stuck_tried_actions(self, pos):
        key = self.get_stuck_history_key(pos)
        if key is None:
            return []
        return self.stuck_vlm_history.get(key, [])
    
    def add_stuck_tried_action(self, pos, action):
        key = self.get_stuck_history_key(pos)
        if key is None:
            return
        if key not in self.stuck_vlm_history:
            self.stuck_vlm_history[key] = []
        if action not in self.stuck_vlm_history[key]:
            self.stuck_vlm_history[key].append(action)
    
    def clear_stuck_history(self):
        self.stuck_vlm_history = {}

# ==================== 辅助函数 ====================
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# ==================== VLM Prompt 生成函数 ====================
def generate_low_pill_guide_prompt(game_state):
    pacman_pos = game_state.get_pacman_pos()
    ghost_positions = game_state.get_ghost_positions()
    legal_actions = game_state.get_legal_actions()
    pill_positions = game_state.get_pill_positions()
    superpill_positions = game_state.get_superpill_positions()
    
    pill_info = []
    if pacman_pos and pill_positions:
        for pill in pill_positions:
            dist = manhattan_distance(pacman_pos, pill)
            pill_info.append((pill, dist))
        pill_info.sort(key=lambda x: x[1])
    
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
    pacman_pos = game_state.get_pacman_pos()
    ghost_positions = game_state.get_ghost_positions()
    legal_actions = game_state.get_legal_actions()
    pill_positions = game_state.get_pill_positions()
    superpill_positions = game_state.get_superpill_positions()
    
    nearest_pill = None
    min_pill_dist = float('inf')
    if pacman_pos and pill_positions:
        for pill in pill_positions:
            dist = manhattan_distance(pacman_pos, pill)
            if dist < min_pill_dist:
                min_pill_dist = dist
                nearest_pill = pill
    
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
    pacman_pos = game_state.get_pacman_pos()
    legal_actions = game_state.get_legal_actions()
    superpill_positions = game_state.get_superpill_positions()
    
    direction_threats = {'up': [], 'down': [], 'left': [], 'right': []}
    
    if pacman_pos:
        px, py = pacman_pos
        for ghost_pos, dist in ghost_distances:
            gx, gy = ghost_pos
            if gy < py:  
                direction_threats['up'].append((ghost_pos, dist))
            if gy > py:  
                direction_threats['down'].append((ghost_pos, dist))
            if gx < px:  
                direction_threats['left'].append((ghost_pos, dist))
            if gx > px:  
                direction_threats['right'].append((ghost_pos, dist))
    
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


def vlm_decide_action(game_state, env_img, scenario, args, ghost_distances=None, save_dir="vlm_debug", tried_actions=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(save_dir, f"vlm_{scenario}_frame{game_state.frame}_{timestamp}.png")
    cv2.imwrite(image_path, env_img)
    
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
    
    response = call_qwen_vl(
        image_path=image_path,
        prompt=prompt,
        vlm_model=args.vlm,
        use_history=False
    )
    
    print(f"   VLM响应: {response}")
    
    action = parse_vlm_action(response)
    
    if scenario == 'stuck' and tried_actions and action in tried_actions:
        legal_actions = game_state.get_legal_actions()
        reverse_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
        for legal in legal_actions:
            if legal in reverse_map:
                alt_action = reverse_map[legal]
                if alt_action not in tried_actions:
                    print(f"   VLM选择了已尝试过的动作{action}，强制改为: {alt_action} ({legal})")
                    action = alt_action
                    break
    
    legal_actions = game_state.get_legal_actions()
    action_map = {1: 'up', 2: 'right', 3: 'left', 4: 'down'}
    
    if action in action_map and action_map[action] not in legal_actions:
        for legal in legal_actions:
            reverse_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
            if legal in reverse_map:
                action = reverse_map[legal]
                print(f"   VLM建议动作不合法，改为: {action} ({legal})")
                break
    
    print(f"   最终决策: {action}")
    return action


def save_stuck_detection_image(env_img, all_game_info, game_state, frame, epoch, save_dir="stuck_detection"):
    import matplotlib.pyplot as plt
    
    display_img = env_img.copy()
    
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
    
    superpill_centers = all_game_info.get('superpill_centers', [])
    for center in superpill_centers:
        if len(center) == 2:
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 6, (255, 255, 0), -1)
    
    pill_centers = all_game_info.get('pill_centers', [])
    for center in pill_centers:
        if len(center) == 2:
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 2, (0, 255, 255), -1)
    
    legal_action = all_game_info.get('pacman_decision', {})
    if pacman_centers and legal_action:
        pacman_center = pacman_centers[0]
        cx, cy = pacman_center
        arrow_length = 25
        arrow_color = (0, 255, 0)  
        
        if legal_action.get('up', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx), int(cy) - arrow_length), arrow_color, 2, tipLength=0.3)
        if legal_action.get('down', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx), int(cy) + arrow_length), arrow_color, 2, tipLength=0.3)
        if legal_action.get('left', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx) - arrow_length, int(cy)), arrow_color, 2, tipLength=0.3)
        if legal_action.get('right', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx) + arrow_length, int(cy)), arrow_color, 2, tipLength=0.3)
    
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
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"stuck_epoch{epoch}_frame{frame}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    
    cv2.imwrite(filepath, display_img)
    print(f"🚨 Pacman卡住！已保存检测图片: {filepath}")
    print(f"   位置: {game_state.pacman_position}, 停留帧数: {game_state.stuck_frames}")
    print(f"   合法动作: {game_state.get_legal_actions()}")

# ==================== BFS路径规划类 ====================
class PathFinder:
    DIRECTIONS = [
        (0, -1, 'up', 1),      # 上：y减小
        (0, 1, 'down', 4),     # 下：y增大
        (-1, 0, 'left', 3),    # 左：x减小
        (1, 0, 'right', 2),    # 右：x增大
    ]
    
    PILL_THRESHOLD = 100
    
    def __init__(self, game_state, search_radius=5):
        self.game_state = game_state
        self.search_radius = search_radius
    
    def find_next_action(self):
        pacman_pos = self.game_state.get_pacman_pos()
        pill_positions = self.game_state.get_pill_positions()
        superpill_positions = self.game_state.get_superpill_positions()
        
        all_targets = pill_positions + superpill_positions
        
        pill_count = len(pill_positions)
        
        if pill_count <= self.PILL_THRESHOLD:
            return self._bfs_find_path(pacman_pos, all_targets)
        else:
            return self._heuristic_find_path(pacman_pos, all_targets)
    
    def _bfs_find_path(self, start_pos, target_positions):
        from collections import deque
        
        obstacles_mask = self.game_state.obstacles_mask
        if obstacles_mask is None:
            return self._heuristic_find_path(start_pos, target_positions)
        
        height, width = obstacles_mask.shape[:2]
        
        target_set = set()
        for pos in target_positions:
            x, y = int(pos[0]), int(pos[1])
            for dx in range(-self.search_radius, self.search_radius + 1):
                for dy in range(-self.search_radius, self.search_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        target_set.add((nx, ny))
        
        start = (int(start_pos[0]), int(start_pos[1]))
        
        if start in target_set:
            nearest = self._find_nearest_target(start_pos, target_positions)
            if nearest and manhattan_distance(start_pos, nearest) > self.search_radius:
                target_positions = [t for t in target_positions if t != nearest]
                if target_positions:
                    return self._bfs_find_path(start_pos, target_positions)
            return 0, start_pos, 'bfs_at_target'
        
        queue = deque()
        visited = set()
        visited.add(start)
        
        opposite_direction = self.game_state.get_opposite_direction()
        
        legal_actions_no_backtrack = self.game_state.get_legal_actions_no_backtrack()
        
        for dx, dy, direction, action in self.DIRECTIONS:
            if direction == opposite_direction and len(legal_actions_no_backtrack) > 0:
                if direction not in legal_actions_no_backtrack:
                    continue
            
            nx, ny = start[0] + dx, start[1] + dy
            if self._is_valid_position(nx, ny, obstacles_mask):
                queue.append(((nx, ny), direction, action, 1))
                visited.add((nx, ny))
        
        while queue:
            (cx, cy), first_direction, first_action, dist = queue.popleft()
            
            if (cx, cy) in target_set:
                target_pos = self._find_nearest_target((cx, cy), target_positions)
                return first_action, target_pos, 'bfs'
    
            for dx, dy, _, _ in self.DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and self._is_valid_position(nx, ny, obstacles_mask):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), first_direction, first_action, dist + 1))
        
        return self._heuristic_find_path(start_pos, target_positions)
    
    def _heuristic_find_path(self, start_pos, target_positions):
        obstacles_mask = self.game_state.obstacles_mask
        
        legal_actions = self.game_state.get_legal_actions_no_backtrack()
        
        if not legal_actions:
            legal_actions = self.game_state.get_legal_actions()
        
        if not legal_actions:
            return 0, None, 'heuristic_no_action'
        
        best_target = None
        best_score = float('inf')
        
        for target in target_positions:
            base_dist = manhattan_distance(start_pos, target)
            
            obstacle_penalty = self._calculate_obstacle_penalty(start_pos, target, obstacles_mask)
            
            ghost_penalty = self._calculate_ghost_penalty(target)
            
            total_score = base_dist + obstacle_penalty * 2 + ghost_penalty * 3
            
            if total_score < best_score:
                best_score = total_score
                best_target = target
        
        if best_target is None:
            return 0, None, 'heuristic_no_target'
        
        best_action = self._select_action_towards_target(start_pos, best_target, legal_actions)
        
        return best_action, best_target, 'heuristic'
    
    def _is_valid_position(self, x, y, obstacles_mask):
        height, width = obstacles_mask.shape[:2]
        
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        
        if obstacles_mask[int(y), int(x)] > 0:
            return False
        
        return True
    
    def _calculate_obstacle_penalty(self, start, target, obstacles_mask):
        if obstacles_mask is None:
            return 0
        
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(target[0]), int(target[1])
        
        height, width = obstacles_mask.shape[:2]
        
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            return 0
        
        penalty = 0
        sample_count = min(10, steps) 
        
        for i in range(1, sample_count + 1):
            t = i / (sample_count + 1)
            check_x = int(x0 + (x1 - x0) * t)
            check_y = int(y0 + (y1 - y0) * t)
            
            if 0 <= check_x < width and 0 <= check_y < height:
                if obstacles_mask[check_y, check_x] > 0:
                    penalty += 10  
        
        return penalty
    
    def _calculate_ghost_penalty(self, target):
        ghost_positions = self.game_state.get_ghost_positions()
        penalty = 0
        
        for ghost_pos in ghost_positions:
            dist = manhattan_distance(target, ghost_pos)
            if dist < 30:
                penalty += (30 - dist) 
        
        return penalty
    
    def _select_action_towards_target(self, start, target, legal_actions):
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        
        preferred_actions = []
        
        if dx > 0:
            preferred_actions.append(('right', 2))
        elif dx < 0:
            preferred_actions.append(('left', 3))
        
        if dy > 0:
            preferred_actions.append(('down', 4))
        elif dy < 0:
            preferred_actions.append(('up', 1))
        
        if abs(dx) >= abs(dy):
            preferred_actions.sort(key=lambda x: 0 if x[0] in ['left', 'right'] else 1)
        else:
            preferred_actions.sort(key=lambda x: 0 if x[0] in ['up', 'down'] else 1)
        
        for action_name, action_num in preferred_actions:
            if action_name in legal_actions:
                return action_num
        
        action_map = {'up': 1, 'right': 2, 'left': 3, 'down': 4}
        for action_name in legal_actions:
            if action_name in action_map:
                return action_map[action_name]
        
        return 0
    
    def _find_nearest_target(self, pos, targets):
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
        if pill_count <= self.PILL_THRESHOLD:
            return "BFS精确搜索"
        else:
            return "启发式搜索（曼哈顿距离+障碍物感知）"

# ==================== 动作决策函数 ====================
def decide_next_action(game_state, args, env_img=None, frame=0):
    should_chase_ghost, ghost_pos, ghost_dist = game_state.should_chase_ghost(args.ghost_chase_threshold)
    
    if should_chase_ghost and ghost_pos is not None:
        path_finder = PathFinder(game_state)
        action, target, strategy = path_finder._bfs_find_path(
            game_state.get_pacman_pos(),
            [ghost_pos]
        )
        if action != 0:
            return action, ghost_pos, 'chase_ghost', False
    
    is_multi_danger, nearby_count, ghost_distances = game_state.is_multi_ghost_danger(
        threshold=args.ghost_danger_threshold,
        min_ghost_count=2
    )
    
    if is_multi_danger and game_state.state != 'chase' and env_img is not None and frame >= 300:
        action = vlm_decide_action(
            game_state, env_img, 
            scenario='multi_ghost', 
            args=args,
            ghost_distances=ghost_distances,
            save_dir="vlm_debug"
        )
        return action, None, 'vlm_multi_ghost', True
    
    is_danger, ghost_dist, nearest_ghost = game_state.is_in_danger(args.ghost_danger_threshold)
    
    if is_danger and game_state.state != 'chase':
        escape_action = _get_escape_action(game_state, nearest_ghost)
        return escape_action, None, 'escape', True
    
    should_chase, superpill_pos, sp_dist, gh_dist = game_state.should_chase_superpill(
        args.superpill_chase_threshold,
        args.superpill_safe_margin
    )
    
    if should_chase and superpill_pos is not None:
        path_finder = PathFinder(game_state)
        action, target, strategy = path_finder._bfs_find_path(
            game_state.get_pacman_pos(), 
            [superpill_pos]
        )
        if action != 0:
            return action, superpill_pos, 'chase_superpill', False
    
    LOW_PILL_THRESHOLD = 20  
    LOW_PILL_VLM_INTERVAL = 30  
    
    if (game_state.pill_num <= LOW_PILL_THRESHOLD and 
        game_state.pill_num > 0 and 
        env_img is not None and 
        frame >= 300 and
        frame % LOW_PILL_VLM_INTERVAL == 0):  
        action = vlm_decide_action(
            game_state, env_img,
            scenario='low_pill',
            args=args,
            save_dir="vlm_debug"
        )
        if action != 0:
            return action, None, 'vlm_low_pill', False
    
    path_finder = PathFinder(game_state)
    action, target, strategy = path_finder.find_next_action()
    
    return action, target, strategy, False

def _get_escape_action(game_state, ghost_pos):
    pacman_pos = game_state.get_pacman_pos()
    legal_actions = game_state.get_legal_actions()
    
    if pacman_pos is None or ghost_pos is None or not legal_actions:
        return 0
    
    px, py = pacman_pos
    gx, gy = ghost_pos
    
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
        new_px, new_py = px + dx * 5, py + dy * 5 
        new_distance = abs(new_px - gx) + abs(new_py - gy)
        
        if new_distance > max_distance:
            max_distance = new_distance
            best_action = action_map[direction]
    
    return best_action if best_action != 0 else 0

def single_action(env, action_num, duration):
    obs = None
    total_reward = 0
    for _ in range(duration):
        obs, reward, terminated, truncated, info = env.step(action_num)
        total_reward += reward
        if terminated or truncated:
            break
    cv2.imwrite(f'MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    return obs, total_reward, terminated, truncated, info

# ==================== 主游戏循环（示例） ====================
def initialize_game():
    args = GameArgs()
    env = gym.make(args.game_name, render_mode='human')
    model = YOLO(args.path)
    game_state = GameState()
    
    print(f"游戏环境 {args.game_name} 初始化完成")
    print(f"YOLO模型加载自: {args.path}")
    
    return env, args, model, game_state

def update_game_state(env_img, args, epoch, frame, former_all_game_info, model, game_state):
    all_game_info = detect_all_in_one(env_img, args, epoch, frame, former_all_game_info, model=model)
    
    game_state.update_from_detect_all(all_game_info, frame, epoch)
    
    return all_game_info

# ==================== 测试代码 ====================
if __name__ == "__main__":
    env, args, model, game_state = initialize_game()
    observation, info = env.reset(seed=42)
    evaluator = PacmanEvaluator()
    reasoner = PacmanRLReasoner(evaluator)
    
    frame = 0
    epoch = 0
    former_all_game_info = None
    last_HP = 3                   
    
    # ========== 决策间隔控制 ==========
    DECISION_INTERVAL = 6          # 决策间隔：每隔多少帧重新调用一次decide_next_action
    current_action = 0             # 当前执行的动作
    current_target = None           # 当前目标
    current_strategy = 'none'       # 当前策略
    frames_since_decision = 0       # 距离上次决策的帧数
    # ==================================
    
    print("开始游戏循环测试...")
    print("=" * 60)
    print("策略说明:")
    print(f"  - 豆子数量 <= {PathFinder.PILL_THRESHOLD}: 使用BFS精确搜索")
    print(f"  - 豆子数量 > {PathFinder.PILL_THRESHOLD}: 使用启发式搜索")
    print(f"  - Ghost距离 < {args.ghost_danger_threshold}: 触发危险模式/逃跑")
    print(f"  - 决策间隔: 每 {DECISION_INTERVAL} 帧重新决策一次")
    print("=" * 60)
    
    try:
        while True:
            image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'MsPacman/cut.png', image_bgr)
            
            all_game_info = update_game_state(image_bgr, args, epoch, frame, former_all_game_info, model, game_state)
            
            value = 5

            # ========== 掉命检测：HP减少时重置frame ==========
            current_HP = game_state.HP
            if current_HP < last_HP:
                print(f"💀 Pacman掉命！HP: {last_HP} -> {current_HP}，重置frame为0")
                frame = 0
                game_state.reset_stuck_detection()
                frames_since_decision = DECISION_INTERVAL
            last_HP = current_HP
            # ================================================

            # ========== 卡住检测与VLM介入 ==========
            is_stuck, stuck_frames = game_state.check_stuck()
            vlm_stuck_action = None
            
            if is_stuck and stuck_frames >= game_state.stuck_threshold and frame >= 300:
                pacman_pos = game_state.get_pacman_pos()
                tried_actions = game_state.get_stuck_tried_actions(pacman_pos)
                
                if stuck_frames == game_state.stuck_threshold:
                    save_stuck_detection_image(
                        image_bgr, all_game_info, game_state, 
                        frame, epoch, save_dir="stuck_detection"
                    )
                
                vlm_stuck_action = vlm_decide_action(
                    game_state, image_bgr,
                    scenario='stuck',
                    args=args,
                    save_dir="vlm_debug",
                    tried_actions=tried_actions
                )
                
                if vlm_stuck_action is not None and vlm_stuck_action != 0:
                    game_state.add_stuck_tried_action(pacman_pos, vlm_stuck_action)
            # ================================

            if frame % 50 == 0:
                game_state.print_state()
            
            # ========== 决策间隔控制逻辑 ==========
            need_new_decision = (
                frames_since_decision >= DECISION_INTERVAL or  
                frame == 0 or                                  
                vlm_stuck_action is not None                   
            )
            
            if need_new_decision:
                if vlm_stuck_action is not None:
                    action = vlm_stuck_action
                    target = None
                    strategy = 'vlm_stuck'
                    is_danger = False
                    game_state.reset_stuck_detection()
                else:
                    action, target, strategy, is_danger = decide_next_action(game_state, args, env_img=image_bgr, frame=frame)
                
                current_action = action
                current_target = target
                current_strategy = strategy
                frames_since_decision = 0  
                
                if frame % (DECISION_INTERVAL * 10) == 0 or strategy.startswith('vlm'):
                    pill_count = game_state.pill_num
                    print(f"[Frame {frame}] 策略: {strategy}, 动作: {action}, 豆子: {pill_count}")
            else:
                action = current_action
                frames_since_decision += 1
            
            if right_situation(all_game_info):
                action, value = reasoner.choose_action(create_state(observation, all_game_info))

            observation, reward, terminated, truncated, info = single_action(env, action, value)
                
            former_all_game_info = all_game_info

            if right_situation(former_all_game_info):
                continue

            frame += 1

            game_state.set_last_action(action)

            if reward >= 10:
                print(f"  🎉 获得奖励: {reward}")
            
            if terminated or truncated:
                print("=" * 60)
                print(f"游戏结束！最终得分: {game_state.score}")
                print("重新开始...")
                print("=" * 60)
                break
            
    except KeyboardInterrupt:
        print("\n用户中断游戏")
    finally:
        env.close()
        print("游戏环境已关闭")