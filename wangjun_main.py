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

# mtzm
from mtzm_kmeans import (find_and_mark_head, find_and_mark_sword, find_and_mark_key, 
                         find_and_mark_bone, find_and_mark_gate, find_and_mark_rope, 
                         find_and_mark_ladder, is_rects_adjacent, cluster_black_rects)
# pacman
from detect_all import (detect_all_in_one, update_ghosts, crop_image, process, 
                        find_label, detect_score, detect_HP)


# ==================== 配置类 ====================
class MockArgs:
    """游戏配置参数"""
    def __init__(self):
        self.size = 256
        self.visualize_save = True
        self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"
        self.your_mission_name = "MissionName" 
        self.game_name = 'ALE/MontezumaRevenge-v5'  # 'ALE/Pacman-v5' 或 'ALE/MontezumaRevenge-v5'蒙特祖马
        self.vlm = 'qwen3-vl-plus'  # 'qwen-vl-plus' 'Qwen-VL-Max' qwen3比qwen强
        self.mtzm_process = [
            "注意我每一步的动作，不要走错" ##wj增加
            "先让主人公顺着出生点最近的梯子往下爬,",
            "从梯子爬下来之后，下来黑绿相间的是一片在向左滚动的传送带，需要向右去靠近绳子所在地",
            "接着，向右跳到黄色的绳子上（技巧：向右跳而不是直接按跳跃，因为传送带会让你起跳的方向偏左）",
            "第四步：在绳子上保持静止观察一小会",
            "第五步：再接着再向右跳一下（注意是跳不是走）以便离开绳子到平台上",
            "第六步：紧接着顺着那里的梯子往下爬",
            "第七步：最后一直向左走"
        ]


# ==================== 全局配置 ====================
# 原文是wym的key，跑的时候尽量换自己的！不然token不够用。注册网址如下
# https://bailian.console.aliyun.com/?spm=5176.29597918.nav-v2-dropdown-menu-0.d_main_1_0_3.3ec27b08miv4qJ&tab=model&scm=20140722.M_10904477._.V_1#/model-market/all
dashscope.api_key = "sk-361f43ece66a49e299a35ef26ac687d7"

args = MockArgs()
conversation_history = []
epoch = 0

# 注册 ALE 环境
gym.register_envs(ale_py)

# 初始化YOLO模型
print(f"正在初始化YOLO模型，路径: {args.path}")
model = YOLO(args.path)


# ==================== VLM相关函数 ====================
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


# ==================== 数字解析和代码执行 ====================
def parse_number(s):
    """转换为适当的数值类型：整数或浮点数"""
    return float(s) if '.' in s else int(s)


def extract_num(code):
    """使用正则表达式匹配所有数字（整数或小数）"""
    numbers = re.findall(r'\b\d+\.?\d*\b', code)
    num = [parse_number(n) for n in numbers]
    return num


def run_code(num, env):
    """执行提取的数字指令"""
    for i in range(len(num) // 2):
        print(num[i], num[i + 1])
        observation, reward, terminated, truncated, info = single_action(env, num[2 * i], num[2 * i + 1])


def single_action(env, action_num, duration):
    """执行单个动作持续指定时间"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        observation, reward, terminated, truncated, info = env.step(action_num)
    
    return observation, reward, terminated, truncated, info


# ==================== Pacman游戏处理 ====================
def process_pacman_frame(observation, env, frame, former_all_game_info):
    """处理Pacman游戏的单帧"""
    image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    image_bgr[:43, :] = np.array([0, 0, 0])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    if image_bgr is None or image_bgr.size == 0:
        print(f"警告: 第 {frame} 帧图像无效，跳过处理")
        return former_all_game_info
    
    all_game_info = detect_all_in_one(
        image_bgr, 
        args, 
        epoch, 
        frame,
        former_all_game_info,
        model=model
    )
    
    # 更新former_all_game_info
    former_all_game_info = all_game_info
    
    # 显示进度
    print(f"已处理帧 {frame + 1}/∞")
    print(f"\n测试完成！结果已保存到 detection_results/{args.your_mission_name}文件夹中。")
    
    # 保存图像
    marked_img = image_rgb
    cv2.imwrite('figure/A_' + str(frame) + ".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
    
    return former_all_game_info, marked_img


def generate_pacman_prompt(frame, former_all_game_info, is_first_frame=False):
    """生成Pacman游戏的提示词"""
    if is_first_frame:
        return f"这是游戏pacman的画面，请分析并返回代码，注意，你应该基础上吃掉离你最近最安全的豆子。预判更安全的地方，往更安全的地方前进，必要的时候也可以吃掉大力丸来击退他们,\
    你可以执行操作：0是NOOP，1是UP，2是RIGHT，3是LEFT，4是DOWN。比如observation = single_action(env, 0, 0.05) 表示保持静止观察0.05秒，你当前已知信息（具体坐标，利于你计算跑多少时间）：{former_all_game_info}。\
    请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒))的指令，指令种类和时间这两个数字组成的代码即可！单次输出总时间不要超过0.2秒，绝对不要有其他输出!会干扰我分析代码!"
    else:
        return f"你可以执行操作：0是NOOP，1是UP，2是RIGHT，3是LEFT，4是DOWN。比如observation = single_action(env, 0, 0.05) 表示保持静止观察0.05秒，你当前已知信息（具体坐标，利于你计算跑多少时间）：{former_all_game_info}。\
    请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒))的指令，指令种类和时间这两个数字组成的代码即可！单次输出总时间不要超过1秒，绝对不要有其他输出!会干扰我分析代码!"


def run_pacman_game(env):
    """运行Pacman游戏主循环"""
    # 0静止1上2右3左4下
    observation, info = env.reset()
    observation, _, terminated, truncated, _ = single_action(env, 0, 0.01)
    
    frame = 0
    former_all_game_info = None
    
    # 处理第一帧
    former_all_game_info, marked_img = process_pacman_frame(observation, env, frame, former_all_game_info)
    
    print("分析")
    result = call_qwen_vl('figure/A_' + str(frame) + ".png", 
                          generate_pacman_prompt(frame, former_all_game_info, is_first_frame=True))
    print("Qwen-VL 代码:", result)
    run_code(extract_num(result), env)
    observation, reward, terminated, truncated, info = single_action(env, 1, 0.02)
    
    # 主循环
    while True:
        frame += 1
        
        # 检查游戏是否结束
        if terminated or truncated:
            print("游戏结束，重新开始...")
            observation, info = env.reset()
        
        # 处理当前帧
        former_all_game_info, marked_img = process_pacman_frame(observation, env, frame, former_all_game_info)
        
        cv2.imwrite('figure/' + str(frame) + ".png", cv2.cvtColor(cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"{frame}.png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
        
        print("分析")
        result = call_qwen_vl('figure/A_' + str(frame) + ".png", 
                              generate_pacman_prompt(frame, former_all_game_info, is_first_frame=False))
        print("Qwen-VL 代码:", result)
        run_code(extract_num(result), env)


# ==================== Montezuma游戏处理 ====================
def extract_montezuma_info(image_rgb):
    """提取蒙特祖马游戏中的关键信息"""
    marked_img, ladder_info = find_and_mark_ladder(image_rgb)
    marked_img, center_head, points_head = find_and_mark_head(marked_img)  # 绿色
    marked_img, center_key, points_key = find_and_mark_key(marked_img)  # 蓝色
    marked_img, center_bone, points_bone = find_and_mark_bone(marked_img)  # 青色
    marked_img, center_rope, points_rope = find_and_mark_rope(marked_img)  # 黄色
    
    print(center_bone, center_head)
    dx = center_head[0] - center_bone[0][0]
    dy = center_head[1] - center_bone[0][1]
    distance = math.sqrt(dx**2 + dy**2)
    
    print(ladder_info)
    print(len(ladder_info))
    
    return marked_img, center_head, center_key, center_bone, center_rope, ladder_info, distance


def build_information_list(center_head, center_key, center_bone, center_rope, ladder_info, distance):
    """构建游戏信息列表"""
    pos_information = [
        center_head, center_key, center_bone, center_rope,
        ladder_info[0]['top'], ladder_info[0]['bottom'], ladder_info[0]['center'],
        ladder_info[1]['top'], ladder_info[1]['bottom'], ladder_info[1]['center'],
        ladder_info[2]['top'], ladder_info[2]['bottom'], ladder_info[2]['center'],
        distance
    ]
    
    str_information = [
        "当前主人公位置是：", "当前钥匙位置是", "当前敌人位置是：", "当前绳子位置是：",
        "当前梯子1顶部坐标是：", "当前梯子1底部坐标是：", "当前梯子1center位置是：",
        "当前梯子2顶部坐标是：", "当前梯子2底部坐标是：", "当前梯子2center位置是：",
        "当前梯子3顶部坐标是：", "当前梯子3底部坐标是：", "当前梯子3center位置是：",
        "当前与敌人距离是："
    ]
    
    all_information = []
    for i, j in zip(str_information, pos_information):
        if i is not None and j is not None:
            if i == "当前与敌人距离是：":
                if j >= 20:
                    info = f"{i}{j},非常安全，不用考虑敌人;"
                    all_information.append(info)
            else:
                info = f"{i}{j};"
                all_information.append(info)
    
    return all_information


def generate_montezuma_prompt(all_information):
    """生成蒙特祖马游戏的提示词"""
    # prompt = f"你身处游戏蒙特祖马第一关，需要找到去拿到钥匙的路线，图片尺寸是（160，210，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}。\
    # 请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，你的可执行操作是————\
    # 0：保持静止观察，1：跳跃，2：顺着梯子往上爬，3：右，4：左，5：顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳，14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，\
    # 指令种类和时间这两个数字组成的代码即可！。无需其他输出，其他输出会严重影响游戏"
    # prompt = (
    #     f"你身处游戏蒙特祖马第一关，需要找到去拿到钥匙的路线，图片尺寸是（160，210，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}。"
    #     "请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，你的可执行操作是————"
    #     "0：保持静止观察，1：跳跃，2：顺着梯子往上爬，3：右，4：左，5：顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳，14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，"
    #     "指令种类和时间这两个数字组成的代码即可！"
    #     "无需其他输出，其他输出会严重影响游戏"
    # )
    prompt = (
        f"你身处游戏蒙特祖马第一关，需要按照固定的步骤把主人公安全带到钥匙所在位置，图片尺寸是（160，210，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}。"
        "你现在应该根据下面的步骤判断当前正在执行哪一步，并只输出合适的动作。"
        "1、顺着梯子1往下爬直到底端；"
        "2、向右走；"
        "3、在桥边沿向右上跳到绳子上，确定自己的位置，注意不要沿着绳子向下走；"
        "4、向右跳到梯子3顶端，并顺着梯子3往下爬到底；"
        "5、保持静止观察；"
        "6、当鬼向左移动且距离较近，可冲刺跳过时，请向左跳；"
        "7、保持向左移动直到梯子2底部；"
        "8、顺着梯子2往上爬到顶端；"
        "9、向左上跳拿钥匙，并走回梯子2顶部；"
        "10、顺着梯子2往下爬；"
        "11、当鬼向左移动且距离较近，可冲刺跳过时，请向右跳；"
        "12、保持向右移动直到梯子3底部；"
        "13、顺着梯子3往上爬到顶部；"
        "14、向左上跳到绳子上，再向左跳到梯子1底部；"
        "15、向上爬到梯子1顶部；"
        "16、向右移动，在桥边沿向右跳并继续向右走。"
        "你可执行的操作是：0保持静止观察，1跳跃，2顺着梯子往上爬，3右，4左，5顺着梯子往下爬，6右上，7左上，8右下，9左下，10向上跳，11向右跳，12向右跳，13向下跳，14右上且跳，15左上且跳，16右下且跳，17左下且跳。"
        "爬梯子的动作时间可以适当长一些，为0.7s。"
        "对于跳跃动作，时间可以适当调整为0.3s到0.5s之间。"
        "对于移动动作，时间可以根据距离适当调整，一般在0.3s。"
        "也不要在绳子上往下移动，不要输出4、5、8、9、13、16、17等会带你离开绳子的向下方向，只要有绳子信息，就继续在绳子上保持静止或通过向左上/向右跳俯冲来改变位置。"
        "严格按照observation = single_action(env, Int(指令种类), float(持续时间，单位是秒))格式输出，只有两个数字组成的代码，绝不要有其他输出。"
    )
    
    return prompt


def run_montezuma_game(env):
    """运行蒙特祖马游戏主循环"""
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = single_action(env, 0, 1)
    
    frame = 0
    image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, image_bgr)
        image_bgr[:43, :] = np.array([0, 0, 0])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 提取游戏信息
        marked_img, center_head, center_key, center_bone, center_rope, ladder_info, distance = extract_montezuma_info(image_rgb)
        all_information = build_information_list(center_head, center_key, center_bone, center_rope, ladder_info, distance)
        
        cv2.imwrite('figure/A_' + str(frame) + ".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
        
        print("分析")
        prompt = generate_montezuma_prompt(all_information)
        print(prompt)
        result = call_qwen_vl('figure/A_' + str(frame) + ".png", prompt)
        print("Qwen-VL 代码:", result)
        run_code(extract_num(result), env)
    
    # 主循环
    while True:
        frame += 1
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            cv2.imwrite('figure/' + str(frame) + ".png", cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
            image_bgr[:43, :] = np.array([0, 0, 0])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            marked_img, center_head, points = find_and_mark_head(image_rgb)  # 绿色
            marked_img, center_bone, points = find_and_mark_bone(marked_img)  # 青色
            
            print(center_head, center_key, center_bone, center_rope)
            cv2.imwrite('figure/A_' + str(frame) + ".png", cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
            
            dx = center_head[0] - center_bone[0][0]
            dy = center_head[1] - center_bone[0][1]
            distance = math.sqrt(dx**2 + dy**2)
            
            all_information = build_information_list(center_head, center_key, center_bone, center_rope, ladder_info, distance)
            
            print("分析")
            prompt = generate_montezuma_prompt(all_information)
            print(prompt)
            result = call_qwen_vl('figure/A_' + str(frame) + ".png", prompt)
            print("Qwen-VL 代码:", result)
            run_code(extract_num(result), env)
            observation, reward, terminated, truncated, info = single_action(env, 0, 0.1)


# ==================== 主函数 ====================
def main(env_name, render=True, episodes=2):
    """主函数：根据游戏类型启动相应的游戏循环"""
    if env_name == "ALE/Pacman-v5" or env_name == "ALE/MsPacman-v5":
        env = gym.make(env_name, render_mode='human' if render else None)
        run_pacman_game(env)
    else:
        env = gym.make(env_name, render_mode='human' if render else None)
        run_montezuma_game(env)
    
    env.close()



main(args.game_name, episodes=2)


# ==================== 历史版本提示词（已废弃） ====================
"""老版本提示词"""
# "你身处第一关，需要找到去拿到钥匙的路线，图像已经被处理，绿色(0,255,255)的框框起来的是主人公，可被你操控，深蓝色框(0,0,255)是钥匙，浅蓝色框(0, 200, 200)是敌人，要避开,图片尺寸是（71，80，3），主人公速度为每秒可移动8像素。你的本关攻略一共有7步，每一步都要一行代码，第一步：先让主人公顺着出生点的梯子往下爬，第二步：下来是一片在向左滚动的传送带，向右走（且由于传送带会反向作用，你要走的时间长一些，1.2秒左右），第三步：接着，向右跳（注意是向右跳不是普通跳）到黄色的绳子上，第四步：在绳子上保持静止观察0.5秒休息，第五步：再接着再向右跳0.5秒（注意是跳不是走）以便离开绳子到平台上，第六步：紧接着顺着那里的梯子往下爬，第七步：最后一直向左走.请输出类似obs = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，你的可执行操作是————{0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：上且跳，11：右且跳，12：左且跳，13：下且跳， \
# 14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳}，指令种类和时间这两个数字组成的代码即可！无需其他输出!无需其他输出!无需其他输出!"

# "你身处第一关，需要找到去拿到钥匙的路线，图像已经被处理，绿色(0,255,255)的框框起来的是主人公，可被你操控，深蓝色框(0,0,255)是钥匙，浅蓝色框(0, 200, 200)是敌人，要避开,图片尺寸是（71，80，3），主人公速度为每秒可移动8像素。你的本关攻略一共有11步，每一步都要一行代码，第一步：开局先保持静止观察1.5秒，第二步：让主人公顺着出生点的梯子往下爬，下来是一片在向左滚动的传送带，第三步：向右走，且由于传送带会反向作用，你要走的时间长一些，1.2秒左右，第四步：并向右跳（注意是向右跳不是普通跳）到黄色的绳子上，最重要的第五步：保持静止观察0.5秒休息（这一步不可省略！！！！！），再接着，并列最重要的第六步：静止0.5秒这一步完成以后，向右跳0.5秒（注意是向右跳不是向右走），以便离开绳子到平台上，第七步：紧接着顺着那里的梯子往下爬，第八步：最后一直向左走5秒，第九步：顺着梯子往上爬。第十步：向左走到钥匙下面。十一步：跳。.请输出类似obs = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，你的可执行操作是————{0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
# 14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳}，指令种类和时间这两个数字组成的代码即可！不要忘记第五和六步，严格执行这两步（现在绳子上静止0.5秒再向右跳0.5秒），非常重要的。无需其他输出!无需其他输出!无需其他输出!"

# "先让主人公顺着出生点最近的梯子往下爬,","从梯子爬下来之后，下来黑绿相间的是一片在向左滚动的传送带，需要向右去靠近绳子所在地",
# "接着，向右跳到黄色的绳子上（技巧：向右跳而不是直接按跳跃，因为传送带会让你起跳的方向偏左）","第四步：在绳子上保持静止观察一小会","第五步：再接着再向右跳一下（注意是跳不是走）以便离开绳子到平台上",
# "第六步：紧接着顺着那里的梯子往下爬","第七步：最后一直向左走"

# 你身处游戏蒙特祖马第一关，需要找到去拿到钥匙的路线，如果与敌人距离太近，则要避开敌人,其他情况不用避开。\
# 图片尺寸是（160，210，3），主人公速度为每秒可移动8像素。你目前的信息有：{all_information}。请输出类似observation = single_action(env, Int(指令种类), float(持续时间，单位是秒，所以基本都是1秒))的指令，\
# 你的可执行操作是————0：保持静止观察，1：跳跃，2：顺着梯子网上爬，3：右，4：左，5：\
# 顺着梯子往下爬，6：右上，7：左上，8：右下，9：左下，10：向上跳，11：向右跳，12：向右跳，13：向下跳， \
# 14：右上且跳，15：左上且跳，16：右下且跳，17：左下且跳，指令种类和时间这两个数字组成的代码即可！。无需其他输出，其他输出会严重影响游戏"