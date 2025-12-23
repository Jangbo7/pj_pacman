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
from mtzm_kmeans import  find_and_mark_head,find_and_mark_sword,find_and_mark_key,find_and_mark_bone,find_and_mark_gate,find_and_mark_rope,find_and_mark_ladder
#pacman
from detect_all import detect_all_in_one,update_ghosts,crop_image,process,find_label,detect_score,detect_HP

# Model_args
# 原文是wym的key，跑的时候尽量换自己的！不然token不够用。注册网址如下
# https://bailian.console.aliyun.com/?spm=5176.29597918.nav-v2-dropdown-menu-0.d_main_1_0_3.3ec27b08miv4qJ&tab=model&scm=20140722.M_10904477._.V_1#/model-market/all
dashscope.api_key = "sk-a7838ffe06eb4b68bdb8f01ffcd44246" 
class MockArgs:
    def __init__(self):
        self.size = 256
        self.visualize_save = True
        self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"#r"Z:\Project\CS\pacman_git\pj_pacman\runs\detect\yolov8n_custom_training2\weights\best.pt"#
        self.your_mission_name = "MissionName" 
        self.game_name='ALE/Pacman-v5'# 'ALE/MontezumaRevenge-v5'蒙特祖马
        self.vlm='qwen3-vl-plus'#'qwen-vl-plus'   'Qwen-VL-Max' qwen3比qwen强
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

def main(env_name, render=True, episodes=2):
    if env_name == "ALE/Pacman-v5" or env_name == "ALE/MsPacman-v5":
        # 0静止1上2右3左4下
        env = gym.make(env_name, render_mode='human' if render else None)
        observation, info = env.reset()
        observation,_,terminated,truncated,_ = single_action(env, 0, 0.01) 
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
            
            # 检查游戏是否结束
            if terminated or truncated:
                print("游戏结束，重新开始...")
                observation, info = env.reset()

            print(f"\n测试完成！结果已保存到 detection_results/{args.your_mission_name}文件夹中。")
            # epoch = epoch + 1

            marked_img=image_rgb
            cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
            print("分析")
            result = call_qwen_vl('figure/A_'+str(frame)+".png", f"这是游戏pacman的画面，请分析并返回代码，你要吃掉离你最近的豆子，并避免碰到敌人\
                                  ，必要的时候也可以吃掉大力丸来击退他们,你可以执行操作：0是NOOP，1是UP，2是RIGHT，3是LEFT，4是DOWN。\
                                    比如observation = single_action(env, 0, 0.05) 表示保持静止观察0.05秒，你当前已知信息（具体坐标，利于你计算跑多少时间）：{former_all_game_info}。请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒))的指令，指令种类和时间这两个数字组成的代码即可！\
                                        单次输出总时间不要超过0.2秒，绝对不要有其他输出!会干扰我分析代码!")
            print("Qwen-VL 代码:", result)
            run_code(extract_num(result),env)
            observation, reward, terminated, truncated, info = single_action(env, 1, 0.02) 
        while True:
            frame+=1
            # 使用临时文件保存图像（避免污染项目目录）
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
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
                
                # 检查游戏是否结束
                if terminated or truncated:
                    print("游戏结束，重新开始...")
                    observation, info = env.reset()

                print(f"\n测试完成！结果已保存到 detection_results/{args.your_mission_name}文件夹中。")
                cv2.imwrite('figure/'+str(frame)+".png", cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
                image_bgr[:43, :] = np.array([0, 0, 0])
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                marked_img=image_rgb
                cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{frame}.png", image_bgr)
                print("分析")
                result = call_qwen_vl('figure/A_'+str(frame)+".png", f"你可以执行操作：0是NOOP，1是UP，2是RIGHT，3是LEFT，4是DOWN。比如observation = single_action(env, 0, 0.05) 表示保持静止观察0.05秒，\
                                    你当前已知信息（具体坐标，利于你计算跑多少时间）：{former_all_game_info}。请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒))的指令，指令种类和时间这两个数字组成的代码即可！\
                                    单次输出总时间不要超过0.2秒，绝对不要有其他输出!会干扰我分析代码!")
                print("Qwen-VL 代码:", result)
                run_code(extract_num(result),env)
    else:
        env = gym.make(env_name, render_mode='human' if render else None)
        observation, info = env.reset()
        observation, reward, terminated, truncated, info = single_action(env, 0, 1) 
        # 将 observation (HWC, RGB) 转为 BGR 保存为 PNG
        image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        frame=0
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image_bgr)
            image_bgr[:43, :] = np.array([0, 0, 0])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            marked_img, center_head, points_head = find_and_mark_head(image_rgb)#绿色
            # marked_img, center_sword, points = find_and_mark_sword(marked_img)#红色
            marked_img, center_key, points_key = find_and_mark_key(marked_img)#蓝色
            marked_img, center_bone, points_bone = find_and_mark_bone(marked_img)#青色
            # marked_img, center_gate, points = find_and_mark_gate(marked_img, 5)#紫色
            marked_img, center_rope, points_rope = find_and_mark_rope(marked_img)#黄色
            marked_img, ladder_info = find_and_mark_ladder(marked_img)
            print(center_bone,center_head)
            dx = center_head[0] - center_bone[0][0]  # 1
            dy = center_head[1] - center_bone[0][1]  # 1
            distance = math.sqrt(dx**2 + dy**2) 
            pos_information=[center_head,  center_key, center_bone,center_rope,ladder_info['top'],ladder_info['bottom'],ladder_info['center'],distance]
            str_information = ["当前主人公位置是1：","当前钥匙位置是", "当前敌人位置是：", "当前绳子位置是：", "当前梯子顶部坐标是：", "当前梯子底部坐标是：", "当前梯子center位置是：", "当前与敌人距离是："]
            all_information=[]
            for i,j in zip(str_information,pos_information):
                if i is not None and j  is not None:
                    if i=="当前与敌人距离是：":
                        if j>=20:
                            info = f"{i}{j},非常安全，不用考虑敌人;"
                            all_information.append(info)
                    else:
                        info = f"{i}{j};"
                        all_information.append(info)
            # print(distance)
            cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
            print("分析")
            """重要的事情："""
            prompt=f"你身处游戏蒙特祖马第一关，需要找到去拿到钥匙的路线，如果与敌人距离太近，则要避开敌人,其他情况不用避开。\
                                  图片尺寸是（71，80，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}。请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，\
                                  你的可执行操作是————0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：\
                                  顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
                                14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，指令种类和时间这两个数字组成的代码即可！。无需其他输出!无需其他输出!无需其他输出!"
            print(prompt)
            result = call_qwen_vl('figure/A_'+str(frame)+".png", prompt)
            print("Qwen-VL 代码:", result)
            run_code(extract_num(result),env)
        while True:
            frame+=1
            # 使用临时文件保存图像（避免污染项目目录）
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                cv2.imwrite('figure/'+str(frame)+".png", cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
                image_bgr[:43, :] = np.array([0, 0, 0])
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                marked_img, center_head, points = find_and_mark_head(image_rgb)#绿色
                # marked_img, center_sword, points = find_and_mark_sword(marked_img)#红色
                marked_img, center_key, points = find_and_mark_key(marked_img)#蓝色
                marked_img, center_bone, points = find_and_mark_bone(marked_img)#青色
                # marked_img, center_gate, points = find_and_mark_gate(marked_img, 5)#紫色
                marked_img, center_rope, points = find_and_mark_rope(marked_img)#黄色
                marked_img, ladder_info = find_and_mark_ladder(marked_img)
                print(center_head,  center_key, center_bone,center_rope,ladder_info['top'],ladder_info['bottom'],ladder_info['center'])
                cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(f"{frame}.png", image_bgr)
                dx = center_head[0] - center_bone[0][0]  # 1
                dy = center_head[1] - center_bone[0][1]  # 1
                distance = math.sqrt(dx**2 + dy**2) 
                pos_information=[center_head,  center_key, center_bone,center_rope,ladder_info['top'],ladder_info['bottom'],ladder_info['center'],distance]
                str_information = ["当前主人公位置是1：","当前钥匙位置是", "当前敌人位置是：", "当前绳子位置是：", "当前梯子顶部坐标是：", "当前梯子底部坐标是：", "当前梯子center位置是：", "当前与敌人距离是："]
                all_information=[]
                for i,j in zip(str_information,pos_information):
                    if i is not None and j  is not None:
                        if i=="当前与敌人距离是：":
                            if j>=20:
                                info = f"{i}{j},非常安全，不用考虑敌人;"
                                all_information.append(info)
                        else:
                            info = f"{i}{j};"
                            all_information.append(info)
                # print(distance)
                cv2.imwrite('figure/A_'+str(frame)+".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
                print("分析")
                """重要的事情："""
                prompt=f"你身处游戏蒙特祖马第一关，需要找到去拿到钥匙的路线，如果与敌人距离太近，则要避开敌人,其他情况不用避开。\
                                    图片尺寸是（71，80，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，\
                                    你的可执行操作是————0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：\
                                    顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
                                    14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，指令种类和时间这两个数字组成的代码即可！。无需其他输出!无需其他输出!无需其他输出!"
                print(prompt)
                result = call_qwen_vl('figure/A_'+str(frame)+".png", prompt)
                print("Qwen-VL 代码:", result)
                run_code(extract_num(result),env)
                observation, reward, terminated, truncated, info = single_action(env, 0, 0.1) 

                # os.unlink(tmp.name)  # 删除临时文件

    env.close()
def single_action(env, action_num, duration):
    start_time = time.time()
    
    while time.time() - start_time < duration:
        observation, reward, terminated, truncated, info = env.step(action_num)
    
    return observation, reward, terminated, truncated, info

if __name__== "__main__":
    main(args.game_name, episodes=2)
# if __name__ == "__main__":
#     main('ALE/Pacman-v5', episodes=2)#Ms