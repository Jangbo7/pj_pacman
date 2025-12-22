from utils_all.info_utils import *
from utils_all.decision_utils import pacman_decision
import cv2
import numpy as np
from PIL import Image

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
            'ghost_num': n,                               # ghosts数量
            'score': n,                                   # 当前帧得分
            'HP': n                                       # 当前帧pacman生命值
        }
    """

    # 检测当前帧的分数和生命值
    score = detect_score(env_img, "./utils_all/patch")
    HP = detect_HP(env_img)
    env_img = cv2.resize(env_img,(256,256))
    path = args.path    
    
    if model is None:
        from ultralytics import YOLO
        model = YOLO(path)
        
    if iter == 0:
        ghost_num = 1
        obstacles_mask = detect_obstacles(env_img, args)
        ghost_info, pacman_info = detect_gp_with_yolo(env_img, model)

        # 初始化ghosts_info，包含4个ghost的信息
        ghosts_info = {
            'ghosts_boxes': [],
            'ghosts_centers': []
        }

        # 填充初始ghost信息（4只鬼重叠）
        if ghost_info['ghost_boxes']:
            # 如果检测到了ghost，复制第一项填充所有位置
            for i in range(4):
                ghosts_info['ghosts_boxes'].append(ghost_info['ghost_boxes'][0])
                ghosts_info['ghosts_centers'].append(ghost_info['ghost_centers'][0])
        else:
            # 如果没有检测到ghost，创建默认值
            default_box = [0, 0, 0, 0]
            default_center = [0, 0]
            for i in range(4):
                ghosts_info['ghosts_boxes'].append(default_box)
                ghosts_info['ghosts_centers'].append(default_center)
        
    else:
        ghost_num = former_all_game_info.get('ghost_num', 1)
        ghosts_info = {
            'ghosts_boxes': former_all_game_info.get('ghosts_boxes', []),
            'ghosts_centers': former_all_game_info.get('ghosts_centers', [])
        }

        obstacles_mask = former_all_game_info['obstacles_mask']
        ghost_info, pacman_info = detect_gp_with_yolo(env_img, model)
        if len(ghost_info['ghost_boxes']) != 0:
            ghost_num = detect_ghost_num(
                ghost_info=ghost_info,
                ghosts_info=ghosts_info['ghosts_centers'],
                ghost_num=ghost_num,
                args=args
            )
            index = iter % 4 
            ghosts_info = update_ghosts(ghost_info, index, ghost_num, ghosts_info)
            
    pill_info = detect_pills_with_detector(env_img, args, path)
    superpill_info = detect_superpill(env_img)
    door_info = detect_doors()
    decision = pacman_decision(pacman_info, obstacles_mask)

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
        
        # ghost数量
        'ghost_num': ghost_num,

        # 当前帧得分
        'score': score,

        # 当前帧pacman生命值
        'HP': HP
    }
    
    
    if args.visualize_save:
        save_and_visualize_detection_results(env_img, all_game_info,iter,epoch,args)

    return all_game_info

def update_ghosts(ghost_info, index, ghost_num,ghosts_info):
        if ghost_num == 1:
            for i in range(4):
                ghosts_info['ghosts_boxes'][i] = ghost_info['ghost_boxes'][0]
                ghosts_info['ghosts_centers'][i] = ghost_info['ghost_centers'][0]
        elif ghost_num == 2 or ghost_num == 3: 
            if ghost_num-1 >= index:
                ghosts_info['ghosts_boxes'][index] = ghost_info['ghost_boxes'][0]
                ghosts_info['ghosts_centers'][index] = ghost_info['ghost_centers'][0]
                if index+2 < 4: 
                    ghosts_info['ghosts_boxes'][index+2] = ghost_info['ghost_boxes'][0]
                    ghosts_info['ghosts_centers'][index+2] = ghost_info['ghost_centers'][0]
            else:
                ghosts_info['ghosts_boxes'][index] = ghost_info['ghost_boxes'][0]
                ghosts_info['ghosts_boxes'][index-2] = ghost_info['ghost_boxes'][0]
                ghosts_info['ghosts_centers'][index] = ghost_info['ghost_centers'][0]
                ghosts_info['ghosts_centers'][index-2] = ghost_info['ghost_centers'][0]
        elif ghost_num == 4: 
            ghosts_info['ghosts_boxes'][index] = ghost_info['ghost_boxes'][0]
            ghosts_info['ghosts_centers'][index] = ghost_info['ghost_centers'][0]
        return ghosts_info

# def crop_image(img, left, top, right, bottom):
#     width, height = img.size
#     left = max(0, left)
#     top = max(0, top)
#     right = min(width, right)
#     bottom = min(height, bottom)
    
#     cropped_img = img.crop((left, top, right, bottom))
    
#     return cropped_img

def crop_image(img, left, top, right, bottom):
    # 检查图像类型
    if isinstance(img, Image.Image):  # 如果是PIL.Image对象
        width, height = img.size
        cropped_img = img.crop((left, top, right, bottom))
    else:  # 如果是numpy数组(OpenCV图像)
        height, width = img.shape[:2]  # OpenCV图像的shape是(height, width, channels)
        # 确保坐标在有效范围内
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        # 裁剪图像
        cropped_img = img[top:bottom, left:right]
    
    return cropped_img

def process(img):
    result = np.zeros_like(img)
    result[np.any(img != [0, 0, 0], axis=-1)] = [255, 255, 255]
    return result

def find_label(img, digit, compare_path) -> int:
    input_img = np.array(crop_image(img, 95-8*digit, 206, 103-8*digit, 215))
    comp_imgs = [np.array(Image.open(f"{compare_path}/patch_{i}.png")) for i in range(10)]
    comp_imgs.append(np.array(Image.open(f"{compare_path}/patch_.png")))

    processed_input = process(input_img)
    label = 0
    min_mse = 10
    for i, comp in enumerate(comp_imgs):
        mse = np.mean((processed_input - process(comp)) ** 2)
        if mse < min_mse:
            label = i
            min_mse = mse

    return label % 10

def detect_score(img, compare_path="./utils_all/patch") -> int:
    score = 0
    score += find_label(img, 3, compare_path)
    score *= 10
    score += find_label(img, 2, compare_path)
    score *= 10
    score += find_label(img, 1, compare_path)
    score *= 10
    score += find_label(img, 0, compare_path)

    return score

def detect_HP(img) -> int:
    img1 = np.array(crop_image(img, 9, 219, 10, 220))
    img2 = np.array(crop_image(img, 17, 219, 18, 220))
    img3 = np.array(crop_image(img, 25, 219, 26, 220))
    
    return int((img1 > 0).sum() > 0) + int((img2 > 0).sum() > 0) + int((img3 > 0).sum() > 0)
