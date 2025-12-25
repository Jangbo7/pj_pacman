from utils_all.info_utils import *
from utils_all.decision_utils import pacman_decision
from utils_all.img_utils import pad_image_to_size   
import cv2
import numpy as np
from PIL import Image
import math


def find_closest_point(target_point, point_list):
    """
    找到点列表中与目标点距离最近的点的索引
    
    :param target_point: 目标点坐标 [x, y]
    :param point_list: 点列表 [[x1, y1], [x2, y2], ...]
    :return: 最近点的索引，如果没有有效点则返回-1
    """
    if not point_list:
        return -1
    
    min_distance = float('inf')
    closest_index = -1
    
    for i, point in enumerate(point_list):
        if point and len(point) == 2:  # 确保点有效
            distance = math.hypot(point[0] - target_point[0], point[1] - target_point[1])
            if distance < min_distance:
                min_distance = distance
                closest_index = i
    
    return closest_index

def detect_all_in_one(env_img, args, epoch, iter, former_all_game_info, model=None):
    """
    在一帧游戏中检测所有游戏元素,包括pacman、ghosts、pills、superpills、doors以及障碍物信息
    
    :param env_img: 游戏环境图像(直接返回的observation)

    :param args: 配置参数对象,包含:
                 size:图片大小参数(设置为256不要改)//
                 visualize_save:参数用于检测detect稳定性(True/False)//
                 mission_name参数用于保存结果文件夹命名('jwb1')//         
                 path: YOLO模型路径("runs/detect/yolov8n_custom_training2/weights/best.pt")//

    :param epoch: (因为死亡/吃掉所有pills)初始化的次数,用于防止已经保存的可视化文件的覆盖

    :param iter: 从游戏初始化开始的帧计数/帧序号(0开始),死亡(全局死亡或者丢失一条命都算)/吃掉所有pills需要重置

    :param former_all_game_info: 上一帧的游戏信息字典,包含ghosts位置等历史信息
    :param model: 已初始化的YOLO模型实例（可选），如果不提供则会使用args.path中的路径创建新模型
    可视化结果保存到detect_results/your_mission_name中包含每帧检测结果以及一个文本文件
    
    :return: 包含当前帧所有游戏元素信息的字典，格式如下：
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
            'legal_action_num': n,                        # 当前帧pacman可行动方向数量
            'ghost_num': n,                               # ghosts数量
            'score': n,                                   # 当前帧得分
            'HP': n ,                                     # 当前帧pacman生命值
            'state': 'init' or 'run' or 'chase',  # 当前pacman游戏状态
'
        }
    """

    # 检测当前帧的分数和生命值
    score = detect_score(cv2.cvtColor(env_img, cv2.COLOR_RGB2BGR), "./utils_all/mspatch")
    HP = detect_HP(cv2.cvtColor(env_img, cv2.COLOR_RGB2BGR))
    
    env_img, _ = pad_image_to_size(env_img, (args.size, args.size))    
    path = args.path    
    
    if model is None:
        from ultralytics import YOLO
        model = YOLO(path)
        
    if iter == 0:
        ghost_num = 1
        obstacles_mask = detect_obstacles(env_img, args)
        state = 'init'
        
        # 1. 首先初始化所有ghost的默认位置
        center_x = args.size // 2
        center_y = args.size // 2
        default_center = [center_x, center_y]
        
        # 边界框初始化为边长7个pixel的正方形，以中心点为中心
        half_size = 7 // 2
        default_box = [center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size]
        
        # 初始化5个ghost的默认信息：ghost0为None，其他ghost使用默认值
        ghosts_info = {
            'ghosts_boxes': [None, default_box, default_box, default_box, default_box],
            'ghosts_centers': [None, default_center, default_center, default_center, default_center]
        }
        
        # 2. 进行检测
        ghost_info, pacman_info = detect_gp_with_yolo(env_img, model)

        # 如果YOLO没检测到pacman，则使用detector重新检测
        if len(pacman_info['pacman_boxes']) == 0:
            ghost_info, pacman_info = detect_gp_with_detector(env_img, args, path)
            print("YOLO didn't detect pacman, using detector to detect...")
        else:
            ghost_info, _ = detect_gp_with_detector(env_img, args, path)  

        # 3. 根据检测结果更新ghost位置信息
        detected_boxes = ghost_info['ghost_boxes']
        detected_centers = ghost_info['ghost_centers']
        
        # iter==0时不会有ghost0，所以直接设置为4个空列表
        ghosts_info['ghosts_boxes'][0] = [[], [], [], []]
        ghosts_info['ghosts_centers'][0] = [[], [], [], []]
        
        # 处理其他ghost（ghost1-ghost4）的检测结果，保持原有逻辑
        for i in range(1, min(5, len(detected_boxes))):
            if detected_boxes[i] and len(detected_boxes[i]) > 0:
                ghosts_info['ghosts_boxes'][i] = detected_boxes[i][0]  # 取第一个检测结果
                ghosts_info['ghosts_centers'][i] = detected_centers[i][0]  # 取第一个检测结果
        
    else:
        # 获取上一帧的ghosts_info和ghost_num
        ghost_num = former_all_game_info.get('ghost_num', 1)
        
        # 确保ghosts_boxes和ghosts_centers的长度至少为5
        former_boxes = former_all_game_info.get('ghosts_boxes', [])
        former_centers = former_all_game_info.get('ghosts_centers', [])
        
        # 扩展列表到长度5，不足的部分用None填充
        # while len(former_boxes) < 5:
        #     former_boxes.append(None)
        # while len(former_centers) < 5:
        #     former_centers.append(None)
            
        ghosts_info = {
            'ghosts_boxes': former_boxes,
            'ghosts_centers': former_centers
        }

        obstacles_mask = former_all_game_info['obstacles_mask']
        
        # 进行检测
        ghost_info, pacman_info = detect_gp_with_yolo(env_img, model)
        
        # 如果YOLO没检测到pacman，则使用detector重新检测
        print(pacman_info['pacman_boxes'])
        if len(pacman_info['pacman_boxes']) == 0:
            ghost_info, pacman_info = detect_gp_with_detector(env_img, args, path)
            print("YOLO didn't detect pacman, using detector to detect...")
        else:
            ghost_info, _ = detect_gp_with_detector(env_img, args, path)  
        
        # 根据检测结果更新所有ghost的位置
        if len(ghost_info['ghost_boxes']) > 0:
            detected_boxes = ghost_info['ghost_boxes']
            detected_centers = ghost_info['ghost_centers']
            
            # 处理ghost0的检测结果：转换为4元素列表格式
            if detected_boxes[0] and len(detected_boxes[0]) > 0:
                # 1. 复制ghost1-4的坐标到ghost0的坐标列表中
                ghosts_info['ghosts_boxes'][0] = ghosts_info['ghosts_boxes'][1:5].copy()
                ghosts_info['ghosts_centers'][0] = ghosts_info['ghosts_centers'][1:5].copy()
                
                # 创建已使用索引的集合，避免重复替换
                used_indices = set()
                
                # 2. 对于每个检测到的ghost0实例，找到最近的坐标并替换
                for detected_ghost0_box, detected_ghost0_center in zip(detected_boxes[0], detected_centers[0]):
                    # 计算与当前检测到的ghost0距离最近的未使用坐标
                    closest_idx = -1
                    min_distance = float('inf')
                    
                    for i, center in enumerate(ghosts_info['ghosts_centers'][0]):
                        if i not in used_indices and center and len(center) == 2:
                            distance = math.hypot(center[0] - detected_ghost0_center[0], center[1] - detected_ghost0_center[1])
                            if distance < min_distance:
                                min_distance = distance
                                closest_idx = i
                    
                    # 3. 用检测到的ghost0坐标替换那个最近的坐标
                    if closest_idx != -1:
                        ghosts_info['ghosts_boxes'][0][closest_idx] = detected_ghost0_box
                        ghosts_info['ghosts_centers'][0][closest_idx] = detected_ghost0_center
                        used_indices.add(closest_idx)
                
                # 4. 用ghost0的4个坐标依次替换ghost1-4坐标
                for i in range(1, 5):
                    if i <= len(ghosts_info['ghosts_boxes'][0]):
                        ghosts_info['ghosts_boxes'][i] = ghosts_info['ghosts_boxes'][0][i-1]
                        ghosts_info['ghosts_centers'][i] = ghosts_info['ghosts_centers'][0][i-1]



                state = 'chase'
                print(f'pacman state: {state}')
            else:
                # 处理其他ghost（ghost1-ghost4）的检测结果，保持原有逻辑,ghost1-4与ghost0不同时被检测
                for i in range(1, min(5, len(detected_boxes))):
                    if detected_boxes[i] and len(detected_boxes[i]) > 0:
                        ghosts_info['ghosts_boxes'][i] = detected_boxes[i][0]  # 取第一个检测结果
                        ghosts_info['ghosts_centers'][i] = detected_centers[i][0]  # 取第一个检测结果

                state = 'run'
                print(f'pacman state: {state}')

            # 更新ghost_num
            ghost_num = 4  # 固定设置为4个ghost
            
    pill_info = detect_pills_with_detector(env_img, args, path)
    superpill_info = detect_superpill(env_img)
    door_info = detect_doors()
    decision , legal_action_num = pacman_decision(pacman_info, obstacles_mask)

    # 合并所有信息到一个大字典中
    all_game_info = {
        # pacman信息
        'pacman_boxes': pacman_info.get('pacman_boxes', []),
        'pacman_centers': pacman_info.get('pacman_centers', []),
        
        # ghosts信息（注意这里用的是ghosts_info而不是ghost_info）
        'ghosts_boxes': ghosts_info.get('ghosts_boxes', []),
        'ghosts_centers': ghosts_info.get('ghosts_centers', []),
        
        # pills信息
        'pill_centers': pill_info.get('pill_centers', []),
        'pill_num': pill_info.get('pill_num', [0]),
        
        # superpills信息
        'superpill_boxes': superpill_info.get('superpill_boxes', []),
        'superpill_centers': superpill_info.get('superpill_centers', []),
        
        # doors信息
        'door_centers': door_info.get('door_centers', []),
        
        # 障碍物掩码
        'obstacles_mask': obstacles_mask,
        
        # 决策信息
        'pacman_decision': decision,

        'legal_action_num': legal_action_num,
        
        # ghost数量
        'ghost_num': ghost_num,

        # 当前帧得分
        'score': score,

        # 当前帧pacman生命值
        'HP': HP,
         
        # 状态信息
        'state': state
    }
    
    
    if args.visualize_save:
        # if iter > 300:
        save_and_visualize_detection_results(env_img, all_game_info,iter,epoch,args)

    return all_game_info

def crop_image(img, left, top, right, bottom):
    if isinstance(img, Image.Image):
        width, height = img.size
        cropped_img = img.crop((left, top, right, bottom))
    else:  
        height, width = img.shape[:2]  
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        cropped_img = img[top:bottom, left:right]
    
    return cropped_img

def find_label(img, digit, compare_path) -> int:
    input_img = np.array(crop_image(img, 95-8*digit, 187, 102-8*digit, 194))
    comp_imgs = [np.array(Image.open(f"{compare_path}/patch_{i}.png")) for i in range(10)]
    comp_imgs.append(np.array(Image.open(f"{compare_path}/patch_.png")))

    processed_input = input_img
    label = 0
    min_mse = 10
    for i, comp in enumerate(comp_imgs):
        mse = np.mean((processed_input - comp) ** 2)
        if mse < min_mse:
            label = i
            min_mse = mse

    return label % 10

def detect_score(img, compare_path="./utils_all/mspatch") -> int:
    score = 0
    score += find_label(img, 4, compare_path)
    score *= 10
    score += find_label(img, 3, compare_path)
    score *= 10
    score += find_label(img, 2, compare_path)
    score *= 10
    score += find_label(img, 1, compare_path)
    score *= 10
    score += find_label(img, 0, compare_path)

    return score

def detect_HP(img) -> int:
    img1 = np.array(crop_image(img, 14, 174, 15, 175))
    img2 = np.array(crop_image(img, 30, 174, 31, 175))
    img3 = np.array(crop_image(img, 46, 174, 47, 175))
    
    return int((img1 > 0).sum() > 0) + int((img2 > 0).sum() > 0) + int((img3 > 0).sum() > 0)