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
        self.dqn_model_path = "Saved_models/model_episode_10.pth"

args = MockArgs()


class RLConfig:
    def __init__(self):
        # 环境配置
        self.env_name = 'ALE/MsPacman-v5'  # 或 'ALE/Pacman-v5'
        self.render = False  # 是否渲染游戏画面
        self.episodes = 10  # 训练轮数
        
        # YOLO模型配置
        self.yolo_model_path = "runs/detect/yolov8n_custom_training2/weights/best.pt"
        self.size = 256
        self.visualize_save = False
        self.mission_name = "lyd"
        
        # 训练配置
        self.max_frames = 10000  # 每轮最大帧数
        self.skip_frames = 65  # 跳过前65帧

class RLTrainer:
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
    
    # def hardcode_to_superpill(self, state, legal_actions_mask, d_threshold=0.1):
    #     """
    #     当 superpill 很近时，硬编码选一个更靠近 superpill 的合法动作。
    #     state 12维中：
    #       d_super=state[8], super_dx=state[9], super_dy=state[10]
    #     返回：action 或 None（表示不接管）
    #     """
    #     dS = float(state[8])
    #     if dS >= d_threshold:
    #         return None

        # legal_actions = [i for i in range(5) if bool(legal_actions_mask[i])]
        # if not legal_actions:
        #     return None

        # # 获取超级豆子的相对坐标
        # super_dx = state[9]  # 超级豆子在x轴上的相对位置
        # super_dy = state[10]  # 超级豆子在y轴上的相对位置

        # # 根据相对坐标选择动作
        # # 动作定义：0-不动，1-上，2-下，3-左，4-右
        # best_action = None
        # min_distance = float('inf')

        # for action in legal_actions:
        #     # 计算执行动作后到超级豆子的预期距离
        #     if action == 0:  # 不动
        #         new_distance = dS
        #     elif action == 1:  # 上 
        #         new_distance = np.sqrt((super_dx) ** 2 + (super_dy + 0.01) ** 2)  # 0.01是估计的移动步长
        #     elif action == 2:  # 右 
        #         new_distance = np.sqrt((super_dx - 0.01) ** 2 + (super_dy) ** 2)
        #     elif action == 3:  # 左 
        #         new_distance = np.sqrt((super_dx + 0.01) ** 2 + (super_dy) ** 2)
        #     elif action == 4:  # 下 
        #         new_distance = np.sqrt((super_dx) ** 2 + (super_dy - 0.01) ** 2)
            
        #     if new_distance < min_distance:
        #         min_distance = new_distance
        #         best_action = action


        # return best_action
    
    def train(self):
        """开始训练"""
        # 初始化环境
        self.env = gym.make(self.config.env_name, render_mode='human' if self.config.render else 'rgb_array',repeat_action_probability=0.0)
        observation, info = self.env.reset()
        for episode in range(self.config.episodes):
            print(f"\n=== 第 {episode + 1}/{self.config.episodes} 轮训练 ===")
            episode_start_time = time.time()
            
            # 初始动作
            observation, _, terminated, truncated, _ = self.single_action(self.env, 0, 1)
            
            # 初始化变量
            frame = 0
            former_all_game_info = None
            pill_list = None
            superpill_list = None
            done = False
            total_reward = 0
            
            # 处理第一帧
            image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            #     cv2.imwrite(tmp.name, image_bgr)
                
            # 检测游戏信息
            all_game_info = detect_all_in_one(
                image_bgr,
                args,
                epoch,                      ## 这里是epoch
                frame,
                former_all_game_info,
                model=self.model
            )
            
            # 记录初始物品位置
            pill_list = all_game_info['pill_centers']
            superpill_list = all_game_info['superpill_centers']
            
            # 更新游戏信息
            former_all_game_info = all_game_info
            
            # 初始化DQN代理
            state = encode_state(all_game_info, pill_list, superpill_list)
            
            # 在训练开始时初始化LinearQ代理，后续不重建
            if self.agent is None:
                self.agent = LinearQAgent(state_size=len(state), action_size=5)
        
            # 主训练循环
            while not done and frame < self.config.max_frames:
                frame += 1
                
                # 跳过前skip_frames帧
                if frame < self.config.skip_frames:
                    self.single_action(self.env, 0, 1)
                    continue
                
                # 记录当前物品位置
                former_pill_list = pill_list.copy()
                former_superpill_list = superpill_list.copy()
                # print(f"former_pill_list: {len(former_pill_list)}")

                # with open('pill.txt', 'a') as f:
                #     f.write(f"len_former_pill_list: {len(former_pill_list)}\n")
                #     f.write(f"len_former_superpill_list: {len(former_superpill_list)}\n")
                
                # 编码当前状态
                state = encode_state(former_all_game_info, former_pill_list, former_superpill_list)
                
                # 获取合法动作掩码
                legal_actions_mask = get_legal_actions_mask(former_all_game_info)
                
                
                # with open('legal_actions.txt', 'a') as f:
                #     f.write(f"legal_actions_mask: {legal_actions_mask}\n")
                # # 我要看看为什么会卡死在左下角
                # pacman_pos = former_all_game_info['pacman_centers'][0]
                # if pacman_pos[0] < 20 and pacman_pos[1] < 20:
                #     with open('corner_behavior.txt', 'x') as f:
                #         f.write(f"legal_actions_mask: {legal_actions_mask}")

                # action = 0
                # # 尝试硬编码接管：接近 superpill 时
                # if former_all_game_info['pacman_centers'] and len(former_all_game_info['pacman_centers']) > 0:
                #     px, py = former_all_game_info['pacman_centers'][0]
                #     if px < 4 and py < 6:
                #         action = 1 
                # else:
                action = self.agent.choose_action(state, legal_actions_mask)
                
                # 执行动作
                observation, _, terminated, truncated, _ = self.single_action(self.env, action, 1)
                done = terminated or truncated
                
                # 处理新帧
                image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                #     cv2.imwrite(tmp.name, image_bgr)
                    
                # 检测新的游戏信息
                next_all_game_info = detect_all_in_one(
                    image_bgr,
                    args,
                    epoch,                      ## 这里是epoch
                    frame,
                    former_all_game_info,
                    model=self.model
                )
                
                # 检测Pacman是否死亡（通过生命值变化）
                former_HP = former_all_game_info.get('HP', 3)  # 默认生命值为3
                next_HP = next_all_game_info.get('HP', 3)      # 默认生命值为3

                if (not next_all_game_info.get('pacman_centers')) or (len(next_all_game_info['pacman_centers']) == 0):
                    print(f"[WARN] frame={frame}: pacman_centers 为空，使用上一帧 pacman 位置兜底")
                    if former_all_game_info and former_all_game_info.get('pacman_centers'):
                        pacmanpos = former_all_game_info['pacman_centers'][0]
                    else:
                        # 连上一帧都没有，那就跳过
                        former_all_game_info = next_all_game_info
                        continue
                else:
                    pacmanpos = next_all_game_info['pacman_centers'][0]
                
                # print(f"pill_list: {len(pill_list)}")
                # 更新物品列表（移除被吃掉的豆子）
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
                # print(f"pill_list: {len(pill_list)}")
                
                # 更新下一状态的物品列表
                next_pill_list = pill_list.copy()
                next_superpill_list = superpill_list.copy()
                # if len(next_pill_list) != len(former_pill_list):
                #     print("11111111111111111111111111111111111111111111111111111")
                #     raise ValueError(f"[ERROR] frame={frame}: 豆子数量变化，从 {len(former_pill_list)} 变为 {len(next_pill_list)}")
                # print(f"next_pill_list: {len(next_pill_list)}")
                

                # 检查游戏是否结束
                if done:
                    print("游戏结束")
                    self.env.reset()
                    break
                
                # 编码下一状态
                next_state = encode_state(next_all_game_info, next_pill_list, next_superpill_list)
                
                # 计算奖励
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

                # with open('reward.txt', 'a') as f:
                #     f.write(f"reward: {reward}\n")

                total_reward += reward
                
                next_legal_actions_mask = get_legal_actions_mask(next_all_game_info)

                self.agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_legal_actions_mask=next_legal_actions_mask
                )
                # 更新游戏信息
                former_all_game_info = next_all_game_info
                state = next_state

                # # pacman 死亡后跳帧（跳过死亡动画帧）
                # if next_all_game_info['state'] == 'run' and next_all_game_info.get('pacman_centers') and next_all_game_info.get('4ghosts_centers'):
                #     p = np.array(next_all_game_info['pacman_centers'][0])
                #     for g in next_all_game_info['4ghosts_centers']:
                #         if np.linalg.norm(np.array(g) - p) < 4:
                #             print(f"[INFO] frame={frame}: 检测到抓住（ghost-pacman 距离很小），跳过动画帧")
                #             SKIP_N = 30  # 先用 20~60 调
                #             for _ in range(SKIP_N):
                #                 if frame >= self.config.max_frames:
                #                     break
                #                 prev_observation = observation
                #                 observation, _, terminated, truncated, _ = self.single_action(self.env, 0, 1)
                #                 frame += 1
                #                 if terminated or truncated:
                #                     done = True
                #                     break
                #             # 跳过后不要用这一步数据更新学习
                #             continue
                
                # 打印训练信息
                if frame % 10 == 0:
                    print(f"第 {frame} 帧，奖励: {reward:.2f}, 探索率: {self.agent.epsilon:.4f}")
            
            # 记录训练时间
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            # 将训练结果写入文件
            with open("training_results.txt", "a", encoding="utf-8") as f:
                f.write(f"第 {episode + 1} 轮训练结束，总帧数: {frame}, 总奖励: {total_reward:.2f}, 当前探索率: {self.agent.epsilon:.4f}, 训练时间: {episode_duration:.2f}秒\n")
            
            self.agent.save_model(f"Saved_models/model_episode_{episode + 1}")

        # 关闭环境
        self.env.close()
        print("\n=== 所有训练轮次完成 ===")

if __name__ == "__main__":
    # 创建配置
    config = RLConfig()

    # 创建模型路径
    if not os.path.exists('Saved_models'):
        os.makedirs('Saved_models')
    
    # 创建训练器并开始训练
    trainer = RLTrainer(config)
    trainer.train()
