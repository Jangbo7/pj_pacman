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
#mtzm
from mtzm_kmeans import  find_and_mark_head,find_and_mark_sword,find_and_mark_key,find_and_mark_bone,find_and_mark_gate,find_and_mark_rope,find_and_mark_ladder,is_rects_adjacent,cluster_black_rects
#pacman
from detect_all import detect_all_in_one,update_ghosts,crop_image,process,find_label,detect_score,detect_HP

# DQN Agent
from DQNAgent import DQNAgent

# Model_args
# 原文是wym的key，跑的时候尽量换自己的！不然token不够用。注册网址如下
# https://bailian.console.aliyun.com/?spm=5176.29597918.nav-v2-dropdown-menu-0.d_main_1_0_3.3ec27b08miv4qJ&tab=model&scm=20140722.M_10904477._.V_1#/model-market/all
dashscope.api_key = "sk-361f43ece66a49e299a35ef26ac687d7"#"sk-a7838ffe06eb4b68bdb8f01ffcd44246" wym #sk-5a3fe1d65d2842619565c5ff7b46a55c  蒋文博 sk-14c3b2cb3b4f4181a4acfee4039d827f 刘一多
dashscope.api_key = "sk-14c3b2cb3b4f4181a4acfee4039d827f"
class MockArgs:
    def __init__(self):
        self.size = 256
        self.visualize_save = True
        self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"#r"Z:\Project\CS\pacman_git\pj_pacman\runs\detect\yolov8n_custom_training2\weights\best.pt"#
        self.your_mission_name = "MissionName" 
        self.game_name='ALE/MsPacmanNoFrameskip-v4'#'ALE/Pacman-v5'# 'ALE/MontezumaRevenge-v5'蒙特祖马
        self.your_mission_name = "lyd" 
        self.game_name='ALE/MontezumaRevenge-v5'#'ALE/Pacman-v5'# 'ALE/MontezumaRevenge-v5'蒙特祖马
        self.vlm='qwen3-vl-plus'#'qwen-vl-plus'   'Qwen-VL-Max' qwen3比qwen强
        self.mtzm_process=["先让主人公顺着出生点最近的梯子往下爬","从梯子爬下来之后，下来黑绿相间的是一片在向左滚动的传送带，需要向右去靠近绳子所在地","接着，向右跳到黄色的绳子上（技巧：向右跳而不是直接按跳跃，因为传送带会让你起跳的方向偏左）","第四步：在绳子上保持静止观察一小会","第五步：再接着再向右跳一下（注意是跳不是走）以便离开绳子到平台上","第六步：紧接着顺着那里的梯子往下爬","第七步：最后一直向左走"]
        self.mtzm_object_way=["接近梯子上缘，可以顺着梯子向下走，反之可以顺着梯子往上走","接近绳子，可以根据相对位置向右上方/左上方跳上绳子，或右下方/左下方跳下绳子",""]
args = MockArgs()


# 初始化YOLO模型
print(f"正在初始化YOLO模型，路径: {args.path}")
model = YOLO(args.path)
# 用于存储前一帧的游戏信息
epoch = 0
# 注册 ALE 环境
gym.register_envs(ale_py)
conversation_history = []
def call_qwen_vl(image_path=None, prompt="", use_history=True, reset_history=False):
    """调用 Qwen-VL 多模态模型分析图像，支持连续对话"""
    global conversation_history
    
    # 如果需要重置历史
    if reset_history:
        conversation_history = []
    
    # 如果提供了图片，添加图片消息
    if image_path:
        user_content = [
            {"image": f"file://{image_path}"},
            {"text": prompt}
        ]
    else:
        user_content = [{"text": prompt}]
    
    # 添加用户消息到历史
    if use_history:
        conversation_history.append({
            "role": "user",
            "content": user_content
        })
        messages = conversation_history
    else:
        messages = [{"role": "user", "content": user_content}]
    
    try:
        response = MultiModalConversation.call(
            model=args.vlm,  # 或 'qwen-vl-max'（更强但更贵） vl3plus
            messages=messages
        )
        
        if response.status_code == 200:
            assistant_response = response.output.choices[0].message.content[0]['text']
            
            # 如果使用历史记录，将助手回复也加入历史
            if use_history:
                conversation_history.append({
                    "role": "assistant",
                    "content": [{"text": assistant_response}]
                })
            
            return assistant_response
        else:
            return f"Error: {response.code} - {response.message}"
    except Exception as e:
        return f"Exception: {str(e)}"



# 转换为适当的数值类型：整数或浮点数
def parse_number(s):
        return float(s) if '.' in s else int(s)
def extract_num(code):
    # 使用正则表达式匹配所有数字（整数或小数）
    numbers = re.findall(r'\b\d+\.?\d*\b', code)
    num = [parse_number(n) for n in numbers]
    return num 
def run_code(num,env):
    # 使用正则表达式匹配所有数字（整数或小数）
    for i in range(len(num)//2):
        print( num[i], num[i+1])
        observation, reward, terminated, truncated, info = single_action(env, num[2*i], num[2*i+1])
        return observation, reward, terminated, truncated, info
            # 1. 首先，找到距离主角最近的active物品
def find_nearest_active_item(head_pos, mtzm_dict, pos_list):
    """
    找到距离主角头部最近的active物品
    
    参数:
        head_pos: 主角位置 (x, y)
        mtzm_dict: 物品状态字典
        pos_list: 位置列表
    
    返回:
        (item_name, item_pos, distance): 物品名称，位置，距离
    """
    if head_pos is None:
        return None, None, float('inf')
    
    # 定义物品索引映射
    item_index_map = {
        "key": 1,           # 钥匙
        "bone": 2,          # 敌人/骨头
        "rope": 3,          # 绳子
        "ladder1": {"top": 4, "bottom": 5, "center": 6},  # 梯子1
        "ladder2": {"top": 7, "bottom": 8, "center": 9},  # 梯子2
        "ladder3": {"top": 10, "bottom": 11, "center": 12} # 梯子3
    }
    
    nearest_item = None
    nearest_pos = None
    min_distance = float('inf')
    
    # 计算距离的函数
    def calculate_distance(pos1, pos2):
        if pos1 is None or pos2 is None:
            return float('inf')
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    # 检查每个active物品
    for item_name, is_active in mtzm_dict.items():
        if not is_active:
            continue  # 跳过非active物品
            
        if item_name in ["key", "bone", "rope"]:
            # 简单物品（钥匙、敌人、绳子）
            idx = item_index_map[item_name]
            item_pos = pos_list[idx]
            if item_pos is not None:
                dist = calculate_distance(head_pos, item_pos)
                if dist < min_distance:
                    min_distance = dist
                    nearest_item = item_name
                    nearest_pos = item_pos
        
        elif item_name.startswith("ladder"):
            # 梯子物品（有顶部、底部、中心三个点）
            ladder_data = item_index_map[item_name]
            for part, idx in ladder_data.items():
                item_pos = pos_list[idx]
                if item_pos is not None:
                    dist = calculate_distance(head_pos, item_pos)
                    if dist < min_distance:
                        min_distance = dist
                        nearest_item = f"{item_name}_{part}"
                        nearest_pos = item_pos
    
    return nearest_item, nearest_pos, min_distance

# ------ DQN utils ------
def encode_state(all_game_info) -> np.ndarray:
    """
    将游戏信息编码为特征状态向量
    输入：
        all_game_info，包含当前帧所有游戏元素信息的字典，格式如下：
        {
            'pacman_boxes': [[x1, y1, x2, y2], ...],      # pacman边界框(左x,右x,上y,下y)
            'pacman_centers': [[x, y], ...],              # pacman中心点列表
            'ghosts_boxes': [[x1, y1, x2, y2], ...],      # 所有ghosts边界框
            'ghosts_centers': [[x, y], ...],              # 所有ghosts中心点
            'pill_centers': [[x, y], ...],                # pills中心点列表
            'pill_num': [n],                              # pills数量
            'superpill_boxes': [[x1, y1, x2, y2], ...],   # superpills边界框
            'superpill_centers': [[x, y], ...],           # superpills中心点
            'door_centers': [[x, y], ...],                # doors中心点(上下传送门)
            'obstacles_mask': mask,                       # 障碍物掩码(只有在第一帧会更新）
            'pacman_decision': {directions},              # 当前帧pacman可行动方向(合法action空间)
            'ghost_num': n,                               # ghosts数量
            'score': n,                                   # 当前帧得分
            'HP': n                                       # 当前帧pacman生命值
        }
    返回：
        state: 编码后的状态向量：
        [
            pacmanx, pacmany,                                            # pacman位置
            min_d_ghost, min_d_ghost_gx, min_d_ghost_gy,                 # 最近ghost距离及位置
            min_d_pill, min_d_pill_px, min_d_pill_py,                    # 最近pill距离及位置
            min_d_superpill, min_d_superpill_px, min_d_superpill_py,     # 最近superpill距离及位置
            all_game_info['score'],                                      # 当前帧得分
            all_game_info['HP']                                          # 当前帧生命值
        ]
    """
    if all_game_info['pacman_centers']:
        pacmanx, pacmany = all_game_info['pacman_centers'][0]
    else:
        pacmanx, pacmany = 0, 0

    ## 最近 ghost
    if all_game_info['ghosts_centers']:
        min_d_ghost = 999
        mid_d_ghostx, mid_d_ghosty = 0, 0
        for gx, gy in all_game_info['ghosts_centers']:
            d_ghost = np.linalg.norm([pacmanx-gx, pacmany-gy])
            if min_d_ghost > d_ghost:
                min_d_ghost = d_ghost
                min_d_ghost_gx, min_d_ghost_gy = gx, gy
    else:
        min_d_ghost = 0
        min_d_ghost_gx, min_d_ghost_gy = 0, 0
    
    ## 最近 pill
    if all_game_info['pill_centers']:
        min_d_pill = 999
        mid_d_pillx, mid_d_pilly = 0, 0
        for px, py in all_game_info['pill_centers']:
            d_pill = np.linalg.norm([pacmanx-px, pacmany-py])
            if min_d_pill > d_pill:
                min_d_pill = d_pill
                mid_d_pillx, mid_d_pilly = px, py
    else:
        min_d_pill = 0
        mid_d_pillx, mid_d_pilly = 0, 0

    # 最近 superpill
    if all_game_info['superpill_centers']:
        min_d_superpill = 999
        mid_d_superpillx, mid_d_superpilly = 0, 0
        for px, py in all_game_info['superpill_centers']:
            d_superpill = np.linalg.norm([pacmanx-px, pacmany-py])
            if min_d_superpill > d_superpill:
                min_d_superpill = d_superpill
            mid_d_superpillx, mid_d_superpilly = px, py
    else:
        min_d_superpill = 0
        mid_d_superpillx, mid_d_superpilly = 0, 0

    # 编码状态
    state = [
        pacmanx, pacmany,
        min_d_ghost, min_d_ghost_gx, min_d_ghost_gy,
        min_d_pill, mid_d_pillx, mid_d_pilly,
        min_d_superpill, mid_d_superpillx, mid_d_superpilly,
        all_game_info['score'],
        all_game_info['HP']
    ]

    
    return np.array(state)

def get_legal_actions_mask(all_game_info) -> np.ndarray:
    """
    获取当前帧pacman可行动方向的掩码
    输入：
        all_game_info，包含当前帧所有游戏元素信息的字典，格式如下：
        {
            'pacman_decision': {directions_mask},              # 当前帧pacman可行动方向(合法action空间)
        }
    返回：
        legal_actions_mask: 可行动方向的掩码向量：
        [
           0/1, 0/1, 0/1, 0/1, 0/1
        ]
    """
    legal_actions_mask = np.zeros(5, dtype=bool)
    # 检查pacman_decision是否为'Caught'状态
    if all_game_info['pacman_decision'] == 'Caught':
        # 如果Pacman被抓住，只能选择静止（NOOP）动作
        legal_actions_mask[0] = 1
        return legal_actions_mask
    decisions = list(all_game_info['pacman_decision'].values())
    for enum, mask in enumerate(decisions):
        if mask > 0.5:
            legal_actions_mask[enum] = 1
    return legal_actions_mask



def main(env_name, render=True, episodes=2):
    if env_name == "ALE/Pacman-v5" or env_name == "ALE/MsPacman-v5":
        # 0静止1上2右3左4下
        env = gym.make(env_name, render_mode='rgb_array' if render else None)
        observation, info = env.reset()
        observation,_,terminated,truncated,_ = single_action(env, 0, 1)     # 从0.01秒改成1帧，对吗？
        image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        frame=0
        former_all_game_info = None
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image_bgr)
            image_bgr[:43, :] = np.array([0, 0, 0])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if image_bgr is None or image_bgr.size == 0:
                print(f"警告: 第 {frame} 帧图像无效，跳过处理")
                pass
            all_game_info = detect_all_in_one(
            image_bgr, 
            args, 
            epoch, 
            frame,
            former_all_game_info,
            model=model
        )
            # #可视化检测结果
            #visualize_detection_results(image_bgr, all_game_info, frame)
            # # 保存ghosts_info到文本文件
            #save_ghosts_info(all_game_info, frame)
            # 更新former_all_game_info
            former_all_game_info = all_game_info
            
            # 显示进度
            print(f"已处理帧 {frame + 1}/∞")
            # print("reward:",reward, "terminated:", terminated, "truncated:", truncated, "info:", info)
            
            # 检查游戏是否结束
            if terminated or truncated:
                print("游戏结束，重新开始...")
                observation, info = env.reset()

            print(f"\n测试完成！结果已保存到 detection_results/{args.your_mission_name}文件夹中。")

            # initialize dqn agent
            state = encode_state(all_game_info)
            dqnAgent = DQNAgent(state_size=len(state), action_size=5)

            # in training loop
            done = False

            # # epoch = epoch + 1

            # marked_img=image_rgb
            # cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
            # print("分析")
            # result = call_qwen_vl('figure/A_'+str(frame)+".png", f"这是游戏pacman的画面，请分析并返回代码，你要吃掉离你最近的豆子，并避免碰到敌人\
            #                       ，必要的时候也可以吃掉大力丸来击退他们,你可以执行操作：0是NOOP，1是UP，2是RIGHT，3是LEFT，4是DOWN。\
            #                         比如observation = single_action(env, 0, 3) 表示保持静止观察3帧（即0.05秒），你当前已知信息（具体坐标，利于你计算跑多少时间）：{former_all_game_info}。请输出类似observation = single_action(env, Int(指令种类), Int(持续帧数))的指令，指令种类和帧数这两个数字组成的代码即可！\
            #                             单次输出总时间不要超过12帧（即0.2秒），绝对不要有其他输出!会干扰我分析代码!")
            # print("Qwen-VL 代码:", result)
            # run_code(extract_num(result),env)
            # observation, reward, terminated, truncated, info = single_action(env, 1, 1)  # 从0.02秒改成1帧，对吗？
        while not done:
            frame+=1

            state = encode_state(all_game_info)             # 第i帧（第一次进循环应该是第0帧）的信息

            legal_actions_mask = get_legal_actions_mask(all_game_info)
            action = dqnAgent.choose_action(state, legal_actions_mask)
            observation, reward, terminated, truncated, info = single_action(env, action, 4)

            done = terminated or truncated

            # 使用临时文件保存图像（避免污染项目目录）
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                image_bgr[:43, :] = np.array([0, 0, 0])
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                if image_bgr is None or image_bgr.size == 0:
                    print(f"警告: 第 {frame} 帧图像无效，跳过处理")
                    pass
                next_all_game_info = detect_all_in_one(
                image_bgr, 
                args, 
                epoch, 
                frame,
                former_all_game_info,                      # former_all_game_info 是第i帧的信息（第一次进循环这里应该是第0帧）
                model=model
            )                                              # next_all_game_info 第i+1帧的信息（第一次进循环这里应该是第1帧）
                # #可视化检测结果
                #visualize_detection_results(image_bgr, all_game_info, frame)
                # # 保存ghosts_info到文本文件
                #save_ghosts_info(all_game_info, frame)
                
                # 显示进度
                print(f"已处理帧 {frame + 1}/∞")
                
                # 检查游戏是否结束
                if done:
                    print("游戏结束，重新开始...")
                    break

                print(f"\n测试完成！结果已保存到 detection_results/{args.your_mission_name}文件夹中。")

                next_state = encode_state(next_all_game_info)

                dqnAgent.store(state, action, reward, next_state, done)
                loss = dqnAgent.update()

                all_game_info = next_all_game_info  # 现在all_game_info 是第i+1帧的信息（第一次进循环这里应该是第1帧）
                former_all_game_info = next_all_game_info  # 现在former_all_game_info 是第i+1帧的信息（第一次进循环这里应该是第1帧）

                # 监控训练过程
                if loss is not None:
                    print(f"第 {frame} 帧，奖励: {reward:.2f}, 损失: {loss:.4f}, 探索率: {dqnAgent.epsilon:.4f}")
                else:
                    print(f"第 {frame} 帧，奖励: {reward:.2f}, 损失: None, 探索率: {dqnAgent.epsilon:.4f}")

                # cv2.imwrite('figure/'+str(frame)+".png", cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
                # image_bgr[:43, :] = np.array([0, 0, 0])
                # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # marked_img=image_rgb
                # cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(f"{frame}.png", image_bgr)
                dx = center_head[0] - center_bone[0][0]  # 1
                dy = center_head[1] - center_bone[0][1]  # 1
                distance = math.sqrt(dx**2 + dy**2) 
                pos_information=[center_head,  center_key, center_bone,center_rope,ladder_info[0]['top'],ladder_info[0]['bottom'],ladder_info[0]['center'],ladder_info[1]['top'],ladder_info[1]['bottom'],ladder_info[1]['center'],ladder_info[2]['top'],ladder_info[2]['bottom'],ladder_info[2]['center'],distance]
                str_information = ["当前主人公位置是：","钥匙位置是", "当前敌人位置是：", "绳子位置是：", "梯子1顶部坐标是：", "梯子1底部坐标是：", "梯子1center位置是：","梯子2顶部坐标是：", "梯子2底部坐标是：", "梯子2center位置是：", "梯子3顶部坐标是：", "梯子3底部坐标是：", "梯子3center位置是：", "当前与敌人距离是："]
                all_information=[]
                for i,j in zip(str_information,pos_information):
                    if i is not None and j  is not None:
                        if i=="当前与敌人距离是：":
                            if j>=20:
                                info_self = f"{i}{j},非常安全，不用考虑敌人;"
                                all_information.append(info_self)
                        else:
                            info_self = f"{i}{j};"
                            all_information.append(info_self)
                # print(distance)
                cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
                print("分析")
                """重要的事情："""
                prompt=f"你目前的信息有：{all_information}。请你进行反思与学习请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，\
                                    你的可执行操作是————0：保持静止观察，1：跳跃，2：顺着梯子往上爬，3：右，4：左，5：\
                                    顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
                                    14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，输出指令种类和时间这两个数字组成的代码即可!无需其他输出!其他输出会严重影响游戏！"
                print(prompt)
                result = call_qwen_vl('figure/A_'+str(frame)+".png", prompt)
                print("Qwen-VL 代码:", result)
                _, reward, terminated, truncated, info = run_code(extract_num(result),env)
                _, reward, terminated, truncated, info = single_action(env, 0, 0.1) 
                # print("分析")
                # result = call_qwen_vl('figure/A_'+str(frame)+".png", f"你可以执行操作：0是NOOP，1是UP，2是RIGHT，3是LEFT，4是DOWN。比如observation = single_action(env, 0, 3) 表示保持静止观察3帧（即0.05秒），\
                #                     你当前已知信息（具体坐标，利于你计算跑多少时间）：{former_all_game_info}。请输出类似observation = single_action(env, Int(指令种类), Int(持续帧数))的指令，指令种类和帧数这两个数字组成的代码即可！\
                #                     单次输出总时间不要超过12帧（即0.2秒），绝对不要有其他输出!会干扰我分析代码!")
                # print("Qwen-VL 代码:", result)
                # run_code(extract_num(result),env)
    # else:
    #     env = gym.make(env_name, render_mode='human' if render else None)
    #     observation, info = env.reset()
    #     observation, reward, terminated, truncated, info = single_action(env, 0, 1) 
    #     # 将 observation (HWC, RGB) 转为 BGR 保存为 PNG
    #     image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    #     frame=0
    #     with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
    #         cv2.imwrite(tmp.name, image_bgr)
    #         image_bgr[:43, :] = np.array([0, 0, 0])
    #         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #         marked_img, ladder_info = find_and_mark_ladder(image_rgb)
    #         marked_img, center_head, points_head = find_and_mark_head(marked_img)#绿色
    #         # marked_img, center_sword, points = find_and_mark_sword(marked_img)#红色
    #         marked_img, center_key, points_key = find_and_mark_key(marked_img)#蓝色
    #         marked_img, center_bone, points_bone = find_and_mark_bone(marked_img)#青色
    #         # marked_img, center_gate, points = find_and_mark_gate(marked_img, 5)#紫色
    #         marked_img, center_rope, points_rope = find_and_mark_rope(marked_img)#黄色
    #         print(center_bone,center_head)
    #         dx = center_head[0] - center_bone[0][0]  # 1
    #         dy = center_head[1] - center_bone[0][1]  # 1
    #         distance = math.sqrt(dx**2 + dy**2) 
    #         print(ladder_info)
    #         print(len(ladder_info))
    #         pos_information=[center_head,  center_key, center_bone,center_rope,ladder_info[0]['top'],ladder_info[0]['bottom'],ladder_info[0]['center'],ladder_info[1]['top'],ladder_info[1]['bottom'],ladder_info[1]['center'],ladder_info[2]['top'],ladder_info[2]['bottom'],ladder_info[2]['center'],distance]
    #         str_information = ["当前主人公位置是：","当前钥匙位置是", "当前敌人位置是：", "当前绳子位置是：", "当前梯子1顶部坐标是：", "当前梯子1底部坐标是：", "当前梯子1center位置是：","当前梯子2顶部坐标是：", "当前梯子2底部坐标是：", "当前梯子2center位置是：", "当前梯子3顶部坐标是：", "当前梯子3底部坐标是：", "当前梯子3center位置是：", "当前与敌人距离是："]
    #         all_information=[]
    #         for i,j in zip(str_information,pos_information):
    #             if i is not None and j  is not None:
    #                 if i=="当前与敌人距离是：":
    #                     if j>=20:
    #                         info = f"{i}{j},非常安全，不用考虑敌人;"
    #                         all_information.append(info)
    #                 else:
    #                     info = f"{i}{j};"
    #                     all_information.append(info)
    #         # print(distance)
    #         cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
    #         print("分析")
    #         """重要的事情："""
    #         prompt=f"你身处游戏蒙特祖马第一关，需要找到去拿到钥匙的路线，如果与敌人距离太近，则要避开敌人,其他情况不用避开。\
    #                               图片尺寸是（160，210，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}。请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，\
    #                               你的可执行操作是————0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：\
    #                               顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
    #                             14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，指令种类和时间这两个数字组成的代码即可！。无需其他输出，其他输出会严重影响游戏"
    #         print(prompt)
    #         result = call_qwen_vl('figure/A_'+str(frame)+".png", prompt)
    #         print("Qwen-VL 代码:", result)
    #         run_code(extract_num(result),env)
    #     while True:
    #         frame+=1
    #         # 使用临时文件保存图像（避免污染项目目录）
    #         with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
    #             image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    #             cv2.imwrite('figure/'+str(frame)+".png", cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
    #             image_bgr[:43, :] = np.array([0, 0, 0])
    #             image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    #             marked_img, center_head, points = find_and_mark_head(image_rgb)#绿色
    #             # marked_img, center_sword, points = find_and_mark_sword(marked_img)#红色
    #             # marked_img, center_key, points = find_and_mark_key(marked_img)#蓝色
    #             marked_img, center_bone, points = find_and_mark_bone(marked_img)#青色
    #             # marked_img, center_gate, points = find_and_mark_gate(marked_img, 5)#紫色
    #             # marked_img, center_rope, points = find_and_mark_rope(marked_img)#黄色
    #             # marked_img, ladder_info = find_and_mark_ladder(marked_img)
    #             print(center_head,  center_key, center_bone,center_rope)#,ladder_info['top'],ladder_info['bottom'],ladder_info['center'])
    #             cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
    #             # cv2.imwrite(f"{frame}.png", image_bgr)
    #             dx = center_head[0] - center_bone[0][0]  # 1
    #             dy = center_head[1] - center_bone[0][1]  # 1
    #             distance = math.sqrt(dx**2 + dy**2) 
    #             pos_information=[center_head,  center_key, center_bone,center_rope,ladder_info[0]['top'],ladder_info[0]['bottom'],ladder_info[0]['center'],ladder_info[1]['top'],ladder_info[1]['bottom'],ladder_info[1]['center'],ladder_info[2]['top'],ladder_info[2]['bottom'],ladder_info[2]['center'],distance]
    #             str_information = ["当前主人公位置是：","钥匙位置是", "当前敌人位置是：", "绳子位置是：", "梯子1顶部坐标是：", "梯子1底部坐标是：", "梯子1center位置是：","梯子2顶部坐标是：", "梯子2底部坐标是：", "梯子2center位置是：", "梯子3顶部坐标是：", "梯子3底部坐标是：", "梯子3center位置是：", "当前与敌人距离是："]
    #             all_information=[]
    #             for i,j in zip(str_information,pos_information):
    #                 if i is not None and j  is not None:
    #                     if i=="当前与敌人距离是：":
    #                         if j>=20:
    #                             info = f"{i}{j},非常安全，不用考虑敌人;"
    #                             all_information.append(info)
    #                     else:
    #                         info = f"{i}{j};"
    #                         all_information.append(info)
    #             # print(distance)
    #             cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
    #             print("分析")
    #             """重要的事情："""
    #             prompt=f"你身处游戏蒙特祖马第一关，需要找到去拿到钥匙的路线，如果与敌人距离太近，则要避开敌人,其他情况不用避开。\
    #                                 图片尺寸是（160，210，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，\
    #                                 你的可执行操作是————0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：\
    #                                 顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
    #                                 14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，指令种类和时间这两个数字组成的代码即可!无需其他输出!其他输出会严重影响游戏！"
    #             print(prompt)
    #             result = call_qwen_vl('figure/A_'+str(frame)+".png", prompt)
    #             print("Qwen-VL 代码:", result)
    #             run_code(extract_num(result),env)
    #             observation, reward, terminated, truncated, info = single_action(env, 0, 0.1) 

                # os.unlink(tmp.name)  # 删除临时文件

    env.close()
# def single_action(env, action_num, duration):
#     start_time = time.time()
    
#     while time.time() - start_time < duration:
#         observation, reward, terminated, truncated, info = env.step(action_num)
    
#     return observation, reward, terminated, truncated, info

def single_action(env, action_num, duration):
    for i in range(duration):
        obs, reward, terminated, truncated, info = env.step(action_num)
        # if (i % 40 == 1):
        #     cv2.imwrite('MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # cv2.imwrite('MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    return obs, reward, terminated, truncated, info

# if __name__== "__main__":
#     main(args.game_name, episodes=2)
if __name__ == "__main__":
    main('ALE/Pacman-v5', episodes=2)#Ms


"""老版本提示词"""
# "你身处第一关，需要找到去拿到钥匙的路线，图像已经被处理，绿色(0,255,255)的框框起来的是主人公，可被你操控，深蓝色框(0,0,255)是钥匙，浅蓝色框(0, 200, 200)是敌人，要避开,图片尺寸是（71，80，3），主人公速度为每秒可移动8像素。你的本关攻略一共有7步，每一步都要一行代码，第一步：先让主人公顺着出生点的梯子往下爬，第二步：下来是一片在向左滚动的传送带，向右走（且由于传送带会反向作用，你要走的时间长一些，1.2秒左右），第三步：接着，向右跳（注意是向右跳不是普通跳）到黄色的绳子上，第四步：在绳子上保持静止观察0.5秒休息，第五步：再接着再向右跳0.5秒（注意是跳不是走）以便离开绳子到平台上，第六步：紧接着顺着那里的梯子往下爬，第七步：最后一直向左走.请输出类似obs = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，你的可执行操作是————{0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：上且跳，11：右且跳，12：左且跳，13：下且跳， \
# 14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳}，指令种类和时间这两个数字组成的代码即可！无需其他输出!无需其他输出!无需其他输出!"

#"你身处第一关，需要找到去拿到钥匙的路线，图像已经被处理，绿色(0,255,255)的框框起来的是主人公，可被你操控，深蓝色框(0,0,255)是钥匙，浅蓝色框(0, 200, 200)是敌人，要避开,图片尺寸是（71，80，3），主人公速度为每秒可移动8像素。你的本关攻略一共有11步，每一步都要一行代码，第一步：开局先保持静止观察1.5秒，第二步：让主人公顺着出生点的梯子往下爬，下来是一片在向左滚动的传送带，第三步：向右走，且由于传送带会反向作用，你要走的时间长一些，1.2秒左右，第四步：并向右跳（注意是向右跳不是普通跳）到黄色的绳子上，最重要的第五步：保持静止观察0.5秒休息（这一步不可省略！！！！！），再接着，并列最重要的第六步：静止0.5秒这一步完成以后，向右跳0.5秒（注意是向右跳不是向右走），以便离开绳子到平台上，第七步：紧接着顺着那里的梯子往下爬，第八步：最后一直向左走5秒，第九步：顺着梯子往上爬。第十步：向左走到钥匙下面。十一步：跳。.请输出类似obs = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，你的可执行操作是————{0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
# 14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳}，指令种类和时间这两个数字组成的代码即可！不要忘记第五和六步，严格执行这两步（现在绳子上静止0.5秒再向右跳0.5秒），非常重要的。无需其他输出!无需其他输出!无需其他输出!"


#王珺pacman提示词：这是游戏pacman的画面，请分析并返回代码，
# 你应该按照你的分析和我我下面的步骤进行操作：
# 注意，鬼的速度与你相当，因此当存在鬼离你较近（曼哈顿距离在3之内）的时候，你应该先放弃吃豆子，而是去朝一个所有鬼都离你较远的地方走\
#  当你和大力丸的曼哈顿距离小于你和鬼的曼哈顿距离时，你应该吃掉大力丸来击退他们,当你吃到大力丸后，你应该追踪曼哈顿距离为5以内的任何鬼，并吃掉他们以获取更高分数\
#  其他状态下，你应该吃离你最近的豆子，如果存在周围都存在豆子的情况，你应该吃掉离鬼最远的豆子
# 你可以执行操作：0是NOOP，1是UP，2是RIGHT，3是LEFT，4是DOWN。\
#     比如observation = single_action(env, 0, 0.05) 表示保持静止观察0.05秒，你当前已知信息（具体坐标，利于你计算跑多少时间）：{former_all_game_info}。请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒))的指令，指令种类和时间这两个数字组成的代码即可！\
#         单次输出总时间不要超过0.2秒，绝对不要有其他输出!会干扰我分析代码!