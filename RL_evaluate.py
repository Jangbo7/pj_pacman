import gymnasium as gym
import ale_py
import time
import cv2
import os
import tempfile
import numpy as np
import random
from ultralytics import YOLO

# Pacman检测模块
from detect_all import detect_all_in_one

# LinearQ Agent
from LinearQAgent import LinearQAgent
from RL_utils import encode_state, get_legal_actions_mask, calculate_reward


# 不知道为啥
epoch = 0

class MockArgs:
    def __init__(self):
        self.size = 256
        self.visualize_save = False
        self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"#r"Z:\Project\CS\pacman_git\pj_pacman\runs\detect\yolov8n_custom_training2\weights\best.pt"#
        self.your_mission_name = "lyd" 
        self.game_name='ALE/MontezumaRevenge-v5'#'ALE/Pacman-v5'# 'ALE/MontezumaRevenge-v5'蒙特祖马
        self.vlm='qwen3-vl-plus'#'qwen-vl-plus'   'Qwen-VL-Max' qwen3比qwen强
        self.mtzm_process=["先让主人公顺着出生点最近的梯子往下爬","从梯子爬下来之后，下来黑绿相间的是一片在向左滚动的传送带，需要向右去靠近绳子所在地","接着，向右跳到黄色的绳子上（技巧：向右跳而不是直接按跳跃，因为传送带会让你起跳的方向偏左）","第四步：在绳子上保持静止观察一小会","第五步：再接着再向右跳一下（注意是跳不是走）以便离开绳子到平台上","第六步：紧接着顺着那里的梯子往下爬","第七步：最后一直向左走"]
        self.mtzm_object_way=["接近梯子上缘，可以顺着梯子向下走，反之可以顺着梯子往上走","接近绳子，可以根据相对位置向右上方/左上方跳上绳子，或右下方/左下方跳下绳子",""]
        self.eval_mode = True
        self.dqn_model_path = "Saved_models/model_episode_10.npy"

args = MockArgs()


class RLConfig:
    def __init__(self):
        # 环境配置
        self.env_name = 'ALE/MsPacman-v5'  # 或 'ALE/Pacman-v5'
        self.render = True  # 是否渲染游戏画面
        self.episodes = 10  # 训练轮数
        
        # YOLO模型配置
        self.yolo_model_path = "runs/detect/yolov8n_custom_training2/weights/best.pt"
        self.size = 256
        self.visualize_save = False
        self.mission_name = "lyd"
        
        # 训练配置
        self.max_frames = 10000  # 每轮最大帧数
        self.skip_frames = 65  # 跳过前65帧

class RLEvaluate:
    def __init__(self, config):
        self.config = config
        self.env = None
        self.model = None
        self.agent = None
        
        # 初始化YOLO模型
        print(f"正在初始化YOLO模型，路径: {self.config.yolo_model_path}")
        self.model = YOLO(self.config.yolo_model_path)
        
        # 注册ALE环境
        gym.register_envs(ale_py)
    
    def single_action(self, env, action_num, duration):
        """执行单个动作指定帧数"""
        for i in range(duration):
            obs, reward, terminated, truncated, info = env.step(action_num)
        return obs, reward, terminated, truncated, info    

    def eval_one_episode(self, model_path=None, max_frames=5000):
        """
        渲染评估：玩一局，不学习
        - render=True
        - epsilon=0（纯贪心）
        - 只跑一局直到 done 或 max_frames
        """
        # 环境（强制渲染）
        self.env = gym.make(
            self.config.env_name,
            render_mode='human',
            repeat_action_probability=0.0
        )
        observation, info = self.env.reset()

        # 初始动作让画面稳定一下
        observation, _, terminated, truncated, _ = self.single_action(self.env, 0, 1)

        frame = 0
        former_all_game_info = None
        done = False
        total_reward = 0.0

        # 第一帧检测
        image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        all_game_info = detect_all_in_one(
            image_bgr,
            args,
            epoch,
            frame,
            former_all_game_info,
            model=self.model
        )

        # 初始化豆子列表
        pill_list = all_game_info['pill_centers']
        superpill_list = all_game_info['superpill_centers']
        former_all_game_info = all_game_info

        # 初始化 agent（若没有）
        state = encode_state(all_game_info, pill_list, superpill_list)
        if self.agent is None:
            self.agent = LinearQAgent(state_size=len(state), action_size=5)

        # 加载模型
        if model_path:
            self.agent.load_model(model_path)
            print(f"[EVAL] loaded model: {model_path}")
        else:
            print("[EVAL] WARN: model_path is None, using current agent weights")

        # 评估：禁用探索
        self.agent.epsilon = 0.0

        # 主循环（一局）
        while not done and frame < max_frames:
            frame += 1

            # 跳过前 skip_frames 帧（与你 train 一致）
            if frame < self.config.skip_frames:
                self.single_action(self.env, 0, 1)
                continue

            former_pill_list = pill_list.copy()
            former_superpill_list = superpill_list.copy()

            # 当前状态
            state = encode_state(former_all_game_info, former_pill_list, former_superpill_list)

            # 合法动作
            legal_actions_mask = get_legal_actions_mask(former_all_game_info)

            # 选动作（纯贪心）
            action = self.agent.choose_action(state, legal_actions_mask)

            # 执行动作
            observation, _, terminated, truncated, _ = self.single_action(self.env, action, 1)
            done = terminated or truncated

            # 新帧检测
            image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            next_all_game_info = detect_all_in_one(
                image_bgr,
                args,
                epoch,
                frame,
                former_all_game_info,
                model=self.model
            )

            # pacman_centers 兜底（照你 train 的写法）
            if (not next_all_game_info.get('pacman_centers')) or (len(next_all_game_info['pacman_centers']) == 0):
                print(f"[EVAL WARN] frame={frame}: pacman_centers 为空，使用上一帧 pacman 位置兜底")
                if former_all_game_info and former_all_game_info.get('pacman_centers'):
                    pacmanpos = former_all_game_info['pacman_centers'][0]
                else:
                    former_all_game_info = next_all_game_info
                    continue
            else:
                pacmanpos = next_all_game_info['pacman_centers'][0]

            # 更新豆子列表（与你 train 一致）
            if pill_list is not None:
                for pill in pill_list:
                    if np.linalg.norm(np.array(pill) - np.array(pacmanpos)) < 4:
                        pill_list.remove(pill)
                        break
            if superpill_list is not None:
                for superpill in superpill_list:
                    if np.linalg.norm(np.array(superpill) - np.array(pacmanpos)) < 4:
                        superpill_list.remove(superpill)
                        break

            next_pill_list = pill_list.copy()
            next_superpill_list = superpill_list.copy()

            # done 就不需要再算 reward 也行；这里为了统计总回报照样算
            next_state = encode_state(next_all_game_info, next_pill_list, next_superpill_list)
            reward = calculate_reward(
                next_all_game_info,
                former_all_game_info,
                next_pill_list,
                next_superpill_list,
                former_pill_list,
                former_superpill_list,
                action=action,
                state=state,
                next_state=next_state
            )

            next_legal_actions_mask = get_legal_actions_mask(next_all_game_info)

            self.agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_legal_actions_mask=next_legal_actions_mask
                )

            total_reward += reward

            former_all_game_info = next_all_game_info
            state = next_state

            if frame % 50 == 0:
                print(f"[EVAL] frame={frame}, action={action}, total_reward={total_reward:.2f}")

        print(f"[EVAL] done={done}, frames={frame}, total_reward={total_reward:.2f}")
        self.env.close()

if __name__ == "__main__":
    # 创建配置
    config = RLConfig()

    # 创建模型路径
    if not os.path.exists('Saved_models'):
        os.makedirs('Saved_models')
    
    # 创建训练器并开始评估
    trainer = RLEvaluate(config)
    trainer.eval_one_episode(model_path=args.dqn_model_path, max_frames=8000)
