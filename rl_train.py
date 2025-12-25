import gymnasium as gym
import ale_py
import time
import cv2
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ultralytics import YOLO
from detect_all import detect_all_in_one
import os

# Register ALE environments
gym.register_envs(ale_py)

class Args:
    def __init__(self):
        self.size = 256
        self.visualize_save = False
        self.mission_name = 'rl_train'
        self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"

from DQNAgent import DQNAgent

def is_valid_detection(game_info):
    """
    检查检测结果是否有效（Pacman是否被检测到）。
    如果Pacman无法被找到或检测失败，返回False。
    """
    if not game_info:
        return False
    pacman_centers = game_info.get('pacman_centers', [])
    if not pacman_centers or len(pacman_centers) == 0:
        return False
    return True

def encode_state(game_info):
    """
    将游戏信息字典编码为固定大小的状态向量。
    状态向量组成（归一化到[0, 1]）：
    - Pacman中心点 (x, y)
    - 4个Ghost中心点 (x, y)（不足4个用-1填充）
    - 最近pill的相对位置 (relative x, relative y)
    - 最近superpill的相对位置 (relative x, relative y)
    - 分数（归一化）
    - 生命值（归一化）
    """
    state = []
    
    # Image dimensions for normalization (assuming 210x160 for MsPacmanNoFrameskip-v4, 
    # but detect_all_in_one might return coordinates based on 256x256 resize? 
    # Let's assume coordinates are in 256x256 as per detect_all.py resize)
    W, H = 210.0, 160.0 

    # Pacman位置
    if game_info['pacman_centers']:
        px, py = game_info['pacman_centers'][0]
        state.extend([px / W, py / H])
    else:
        state.extend([-1.0, -1.0]) # Pacman未找到
        px, py = -1, -1

    # Ghost位置（最多4个）
    ghosts = game_info.get('ghosts_centers', [])
    for i in range(4):
        if i < len(ghosts):
            gx, gy = ghosts[i]
            state.extend([gx / W, gy / H])
        else:
            state.extend([-1.0, -1.0])

    # 最近的Pill
    pills = game_info.get('pill_centers', [])
    if pills and px != -1:
        # 找最近的
        dists = [((p[0]-px)**2 + (p[1]-py)**2) for p in pills]
        min_idx = np.argmin(dists)
        nx, ny = pills[min_idx]
        state.extend([(nx - px) / W, (ny - py) / H])
    else:
        state.extend([0.0, 0.0])

    # 最近的Superpill
    superpills = game_info.get('superpill_centers', [])
    if superpills and px != -1:
        dists = [((p[0]-px)**2 + (p[1]-py)**2) for p in superpills]
        min_idx = np.argmin(dists)
        nx, ny = superpills[min_idx]
        state.extend([(nx - px) / W, (ny - py) / H])
    else:
        state.extend([0.0, 0.0])

    # 分数（缩放）
    state.append(game_info.get('score', 0) / 10000.0)

    # 生命值（缩放）
    state.append(game_info.get('HP', 3) / 3.0)

    return np.array(state, dtype=np.float32)

def get_legal_actions_mask(game_info):
    """
    返回合法动作的掩码 [NOOP, UP, RIGHT, LEFT, DOWN]
    """
    mask = np.zeros(5, dtype=np.float32)
    mask[0] = 1.0 # NOOP（静止）始终合法

    decision = game_info.get('pacman_decision', {})
    if decision == 'Caught':
        return mask # 被抓住时只能NOOP
    
    if isinstance(decision, dict):
        if decision.get('up', 0): mask[1] = 1.0
        if decision.get('right', 0): mask[2] = 1.0
        if decision.get('left', 0): mask[3] = 1.0
        if decision.get('down', 0): mask[4] = 1.0
    
    return mask

def play_game(env_name, render=True, verbose=True):
    """
    DQN智能体的主训练循环。
    
    参数:
        env_name: gymnasium环境名称
        render: 如果为True，显示游戏窗口（人类模式）。如果为False，无头运行（训练更快）
        verbose: 如果为True，每步打印详细训练信息（reward, loss等）
    """
    args = Args()
    
    # Load YOLO model once
    print(f"Loading YOLO model from {args.path}...")
    try:
        model = YOLO(args.path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    env = gym.make(env_name, render_mode='human' if render else 'rgb_array')
    
    # 初始化智能体
    # 状态维度: 2 (Pac) + 8 (Ghosts) + 2 (Pill) + 2 (Super) + 1 (Score) + 1 (HP) = 16
    state_size = 16
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    
    num_episodes = 1000
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # 初始检测
        image_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # 调整到256x256以确保与detect_score兼容（需要高度>215）
        image_bgr = cv2.resize(image_bgr, (256, 256))
        former_all_game_info = {} # 第一帧初始化为空
        
        # 检测第一帧
        all_game_info = detect_all_in_one(
            image_bgr, 
            args, 
            episode, 
            0, # frame 0
            former_all_game_info,
            model=model
        )
        
        state = encode_state(all_game_info)
        total_reward = 0
        frame_count = 0
        done = False
        
        while not done:
            frame_count += 1
            
            # 选择动作
            legal_mask = get_legal_actions_mask(all_game_info)
            action = agent.choose_action(state, legal_mask)
            
            # 执行动作
            # 重复动作4帧（帧跳跃）
            total_step_reward = 0
            for _ in range(2):
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_step_reward += reward
                if terminated or truncated:
                    break
            
            # 检测下一状态
            next_image_bgr = cv2.cvtColor(next_obs, cv2.COLOR_RGB2BGR)
            # 调整到256x256以确保与detect_score兼容
            next_image_bgr = cv2.resize(next_image_bgr, (256, 256))
            next_all_game_info = detect_all_in_one(
                next_image_bgr, 
                args, 
                episode, 
                frame_count,
                all_game_info, # 将当前信息作为前一帧信息传入
                model=model
            )
            
            next_state = encode_state(next_all_game_info)
            
            # 自定义奖励塑形
            # 1. 使用Gym的奖励（分数增加）作为基础
            custom_reward = total_step_reward
            
            # 2. 死亡惩罚（通过视觉检测的HP判断）
            current_hp = all_game_info.get('HP', 3)
            next_hp = next_all_game_info.get('HP', 3)
            
            if next_hp < current_hp:
                custom_reward -= 50 # 丢失一条命的重大惩罚
            
            # 3. 时间惩罚，鼓励快速通关
            custom_reward -= 0.1
            
            # 4. 基于距离的奖励塑形（学习main.py中对YOLO信息的利用）
            def get_pos(info, key):
                centers = info.get(key, [])
                return centers[0] if centers else None

            curr_pac = get_pos(all_game_info, 'pacman_centers')
            next_pac = get_pos(next_all_game_info, 'pacman_centers')

            if curr_pac and next_pac:
                # A. Pill吸引奖励
                curr_pills = all_game_info.get('pill_centers', [])
                next_pills = next_all_game_info.get('pill_centers', [])
                
                def get_min_dist(pos, targets):
                    if not targets: return None
                    return min([((t[0]-pos[0])**2 + (t[1]-pos[1])**2)**0.5 for t in targets])

                d_curr = get_min_dist(curr_pac, curr_pills)
                d_next = get_min_dist(next_pac, next_pills)

                if d_curr is not None and d_next is not None:
                    # 奖励靠近pill的行为
                    # 如果d_curr > d_next，说明我们靠近了（正向差值）
                    diff = d_curr - d_next
                    # 限制差值范围，避免检测闪烁导致的巨大跳跃
                    diff = max(min(diff, 10), -10) 
                    custom_reward += diff * 0.5

                # B. Ghost躲避惩罚
                next_ghosts = next_all_game_info.get('ghosts_centers', [])
                d_ghost_next = get_min_dist(next_pac, next_ghosts)
                
                if d_ghost_next is not None:
                    if d_ghost_next < 20: # 非常接近ghost
                        custom_reward -= 5.0 # 危险惩罚
                    elif d_ghost_next < 40:
                        custom_reward -= 1.0 # 警告惩罚

            # 5. 吃豆奖励（显式检查数量变化）
            curr_pill_num = len(all_game_info.get('pill_centers', []))
            next_pill_num = len(next_all_game_info.get('pill_centers', []))
            if next_pill_num < curr_pill_num:
                custom_reward += 5.0 # 吃到pill的额外奖励（在分数之上）

            # 6. 大分奖励（如吃掉ghost或superpill）
            # 如果分数增加显著（>100），可能是吃了ghost或superpill
            if total_step_reward >= 200:
                custom_reward += 10 # 大操作奖励
            
            done = terminated or truncated
            
            # 判断是否应该存储这个transition
            # 情况1: 两个检测都有效 -> 正常训练
            # 情况2: 当前有效但下一个无效（Pacman死了）-> 仍然训练，带死亡惩罚
            # 情况3: 当前无效 -> 跳过（起始数据就有问题）
            curr_valid = is_valid_detection(all_game_info)
            next_valid = is_valid_detection(next_all_game_info)
            
            should_train = False
            if curr_valid and next_valid:
                # 正常情况：两个状态都有效
                should_train = True
            elif curr_valid and not next_valid:
                # Pacman被杀死（之前检测到，之后检测不到）
                # 这是重要的学习信号 - 保留！
                should_train = True
                if verbose:
                    print(f"  第{frame_count}步: Pacman被杀死（死亡transition已记录）")
            else:
                # 当前状态无效 - 跳过这个transition
                if verbose:
                    print(f"  第{frame_count}步: 跳过（当前状态检测无效）")
            
            if should_train:
                # 存入经验回放
                agent.store(state, action, custom_reward, next_state, done)
                
                # 更新智能体
                loss = agent.update()
            else:
                loss = None
            
            # 更新状态
            state = next_state
            all_game_info = next_all_game_info
            total_reward += custom_reward
            
            # 如果verbose，打印步骤信息
            if verbose and should_train:
                loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                print(f"  第{frame_count}步: 动作={action}, 奖励={custom_reward:.2f}, 损失={loss_str}, 探索率={agent.epsilon:.3f}")
            
            if done:
                print(f"第{episode + 1}/{num_episodes}局, 分数: {all_game_info.get('score', 0)}, 总奖励: {total_reward:.2f}, 探索率: {agent.epsilon:.3f}")
                break
        
        # 定期保存模型
        if (episode + 1) % 10 == 0:
            agent.save_model(f"dqn_pacman_ep{episode+1}.pth")
            print(f"模型已保存: dqn_pacman_ep{episode+1}.pth")

    env.close()
    print("训练完成！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='训练MsPacman的DQN智能体')
    parser.add_argument('--render', action='store_true', default=False, 
                        help='渲染游戏窗口（默认: False，关闭可加速训练）')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='打印详细的逐步训练信息')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                        help='仅打印每局摘要')
    args = parser.parse_args()
    
    play_game('MsPacmanNoFrameskip-v4', render=args.render, verbose=args.verbose)
