from ultralytics import YOLO
from supervisor.pill_detector import PillDetector
from supervisor.object_detector import ObjectDetector
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import os
from .img_utils import pad_image_to_size

def detect_gp_with_yolo(image, model):
    """
    Use yolo to detect ghost and pacman
    
    :param image: 输入图像
    :param model: 已初始化的YOLO模型实例

    :return: 包含ghost和pacman边界框及中心点的字典
             格式: {
                 'ghost_boxes': [[x1, y1, x2, y2], ...],
                 'ghost_centers': [[x, y], ...],
                 'pacman_boxes': [[x1, y1, x2, y2], ...],
                 'pacman_centers': [[x, y], ...]
             }
    """
    # pr_image = cv2.resize(image, (256, 256))
    pr_image, _ = pad_image_to_size(image, (256, 256)) 
    results = model.predict(source=pr_image, conf=0.5)
    
    # 初始化返回字典
    result_dict_g = {
        'ghost_boxes': [],
        'ghost_centers': [],
    }
    result_dict_p = {

        'pacman_boxes': [],
        'pacman_centers': []
    }
    
    # 类别名称映射
    class_names = {0: 'pacman', 1: 'ghost'}
    
    # 用于跟踪最高置信度的pacman
    best_pacman_confidence = -1
    best_pacman_box = None
    best_pacman_center = None
    
    # 处理检测结果
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])  # 获取置信度
                
                # 计算中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 根据类别ID处理结果
                if class_id == 0:  # pacman
                    # 只保留置信度最高的pacman
                    if confidence > best_pacman_confidence:
                        best_pacman_confidence = confidence
                        best_pacman_box = [x1, y1, x2, y2]
                        best_pacman_center = [center_x, center_y]
                elif class_id == 1:  # ghost
                    result_dict_g['ghost_boxes'].append([x1, y1, x2, y2])
                    result_dict_g['ghost_centers'].append([center_x, center_y])
    
    # 添加置信度最高的pacman到结果中（如果检测到了）
    if best_pacman_box is not None:
        # 计算边界框面积
        box_width = best_pacman_box[2] - best_pacman_box[0]
        box_height = best_pacman_box[3] - best_pacman_box[1]
        box_area = box_width * box_height
        # print(f'box_area{box_area}')
       
        # if 60 <= box_area <= 140:
        result_dict_p['pacman_boxes'].append(best_pacman_box)
        result_dict_p['pacman_centers'].append(best_pacman_center)
        # else:
        #     # 面积不符合要求，置为None
        #     result_dict_p['pacman_boxes'] = []
        #     result_dict_p['pacman_centers'] = []
    
    return result_dict_g, result_dict_p



def detect_ghost_num(ghost_info, ghosts_info, ghost_num, args):
    """
    检测当前帧鬼的数量
    
    :param ghost_info: 当前帧检测到的鬼位置信息（字典格式）
    :param ghosts_info: 之前记录的鬼位置信息（列表格式）
    :param ghost_num: 当前已知的鬼数量
    :param args: 参数配置
    :return: 更新后的鬼数量
    """
    # 如果没有先前的鬼信息，直接返回当前数量
    if not ghosts_info:
        return ghost_num
    
    # 获取当前帧中所有鬼的位置
    current_centers = ghost_info.get('ghost_centers', [])
    
    # 如果当前帧没有检测到鬼，返回原数量
    if not current_centers:
        return ghost_num
    
    # 计算当前检测到的鬼与之前记录的鬼之间的最小距离
    min_distance = float('inf')
    
    # 使用之前记录的鬼位置信息（已经是列表格式）
    prev_centers = ghosts_info
    
    # 计算距离
    for prev_center in prev_centers:
        prev_x, prev_y = prev_center
        for curr_center in current_centers:
            curr_x, curr_y = curr_center
            # 计算两点间欧几里得距离
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            if distance < min_distance:
                min_distance = distance
    
    print(f'min_distance{min_distance}')
    # 如果最小距离超过阈值，说明可能新增了鬼
    # 考虑到图像大小为256x256，将阈值设为30是合理的
    if min_distance > 4 and ghost_num < 4:
        ghost_num += 1
    
    return ghost_num


def detect_pills_with_detector(env_img, args, path):
    """
    使用PillDetector检测pill位置
    
    :param env_img: 输入图像
    :param args: 参数配置
    :param path: 未使用参数（为了保持接口一致）
    
    :return: 包含pill中心点和数量的字典
             格式: {
                 'pill_centers': [[x1, y1], [x2, y2], ...],
                 'pill_num': [a]  # a为检测到的pill数量
             }
    """
    # pill_color = (223, 192, 111)
    pill_color = (228,111,111)
    # 不用管此处的iter和epoch
    pill_detector = PillDetector(env_img, args, iter_num=0, epoch=0)
    pill_positions, pill_count = pill_detector.detect_pills(target_colors=pill_color, min_area=3, 
                                                            max_area=16, min_count=8)
    
    # 构造返回字典
    result_dict = {
        'pill_centers': list(pill_positions),  # pill_positions本身就是一个(x, y)坐标列表
        'pill_num': [pill_count]  # 将pill数量包装在列表中
    }
    
    return result_dict

def detect_gp_with_detector(env_img, args, path):
    """
    使用ObjectDetector检测ghost和pacman位置
    
    :param env_img: 输入图像
    :param args: 参数配置
    :param path: 未使用参数（为了保持接口一致）
    
    :return: 两个字典：
             1. ghost_info: 包含所有鬼信息的字典
                格式: {
                    'ghost_boxes': [[[x1, y1, x2, y2], ...], ...]  # ghost0-ghost4的列表，每个ghost类型包含多个检测结果的列表
                    'ghost_centers': [[[x, y], ...], ...]  # ghost0-ghost4的列表，每个ghost类型包含多个检测结果的列表
                }
             2. pacman_info: 包含pacman信息的字典
                格式: {
                    'pacman_boxes': [[x1, y1, x2, y2], ...]  # 包含所有检测到的pacman的边界框
                    'pacman_centers': [[x, y], ...]  # 包含所有检测到的pacman的中心点
                }
    """
    target_color =[np.array([66,114,194]),np.array([84,184,153]),np.array([200,72,72]),
            np.array([198,89,179]),np.array([180,122,48]),np.array([210,164,74])]
    
    # 创建ObjectDetector实例，使用0作为默认的iter_num参数
    object_detector = ObjectDetector(env_img, args, iter_num=0, epoch=0)
    annotated_image, detected_objects = object_detector.extract_complex_objects_with_holes(target_colors=target_color, 
                                                                                        min_area=15, max_area=85)
    
    # 初始化ghost信息，每个ghost类型使用列表存储多个检测结果
    ghost_boxes = [[], [], [], [], []]
    ghost_centers = [[], [], [], [], []]
    
    # 初始化pacman信息
    pacman_boxes = []
    pacman_centers = []
    
    # 处理检测到的对象
    for obj in detected_objects:
        x, y, w, h = obj['bbox']
        color_idx = obj['color_index']
        
        # 计算边界框坐标 [x1, y1, x2, y2]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        box = [x1, y1, x2, y2]
        
        # 计算中心点坐标
        center_x = x + w // 2
        center_y = y + h // 2
        center = [center_x, center_y]
        
        # 根据颜色索引分配对象类型
        if color_idx < 5:  # 前5种颜色对应ghost0-ghost4
            ghost_boxes[color_idx].append(box)
            ghost_centers[color_idx].append(center)
        elif color_idx == 5:  # 第6种颜色对应pacman
            pacman_boxes.append(box)
            pacman_centers.append(center)
    
    # 构建返回的字典格式
    ghost_info = {
        'ghost_boxes': ghost_boxes,
        'ghost_centers': ghost_centers
    }
    
    pacman_info = {
        'pacman_boxes': pacman_boxes,
        'pacman_centers': pacman_centers
    }
    
    return ghost_info, pacman_info


def detect_superpill(env_img, iter=0):
    """
    使用ObjectDetector检测超级药丸的位置
    
    :param env_img: 输入图像
    :param iter: 迭代次数，用于决定检测策略
    
    :return: 包含超级药丸边界框和中心点的字典
             格式: {
                 'superpill_boxes': [[x1, y1, x2, y2], ...],
                 'superpill_centers': [[x, y], ...]
             }
    """
    class MockArgs:
        def __init__(self):
            self.size = 256
            self.capture = False
    
    mock_args = MockArgs()
    
    superpill_color = np.array([228, 111, 111])
    target_colors = [superpill_color]
    
    img_height, img_width = env_img.shape[:2]
    
    object_detector = ObjectDetector(env_img, mock_args, iter_num=iter, epoch=0)
    annotated_image, superpill_clusters = object_detector.extract_multiple_colors_clusters(
        target_colors=target_colors,
        min_area=10,
        max_area=30,
        classify_objects=False
    )
    
    detected_boxes = []
    detected_centers = []
    
    for cluster in superpill_clusters:
        x, y, w, h = cluster['bbox']
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        box = [x1, y1, x2, y2]
        center_x = x + w // 2
        center_y = y + h // 2
        center = [center_x, center_y]
        detected_boxes.append(box)
        detected_centers.append(center)
    
    if iter == 0:
        if len(detected_boxes) < 2:
            return {'superpill_boxes': [], 'superpill_centers': []}
        
        left_indices = sorted(range(len(detected_centers)), 
                             key=lambda i: detected_centers[i][0])[:2]
        right_indices = sorted(range(len(detected_centers)), 
                              key=lambda i: detected_centers[i][0])[-2:]
        
        left_centers = [detected_centers[i] for i in left_indices]
        right_centers = [detected_centers[i] for i in right_indices]
        left_boxes = [detected_boxes[i] for i in left_indices]
        right_boxes = [detected_boxes[i] for i in right_indices]
        
        left_centers_sorted = sorted(left_centers, key=lambda c: c[1])
        right_centers_sorted = sorted(right_centers, key=lambda c: c[1])
        left_boxes_sorted = [left_boxes[left_centers.index(c)] for c in left_centers_sorted]
        right_boxes_sorted = [right_boxes[right_centers.index(c)] for c in right_centers_sorted]
        
        initial_superpill_data = {
            'boxes': [left_boxes_sorted[0], right_boxes_sorted[0], 
                      left_boxes_sorted[1], right_boxes_sorted[1]],
            'centers': [left_centers_sorted[0], right_centers_sorted[0],
                        left_centers_sorted[1], right_centers_sorted[1]],
            'left_indices': [0, 2],
            'right_indices': [1, 3],
            'left_detected': True,
            'right_detected': True
        }
        
        detect_superpill.initial_superpill_data = initial_superpill_data
        
        return {
            'superpill_boxes': initial_superpill_data['boxes'][:],
            'superpill_centers': initial_superpill_data['centers'][:]
        }
    else:
        initial_superpill_data = getattr(detect_superpill, 'initial_superpill_data', None)
        if initial_superpill_data is None:
            return {'superpill_boxes': [], 'superpill_centers': []}
        
        initial_boxes = initial_superpill_data['boxes']
        initial_centers = initial_superpill_data['centers']
        left_detected = initial_superpill_data.get('left_detected', True)
        right_detected = initial_superpill_data.get('right_detected', True)
        
        has_left = any(c[0] < img_width // 2 for c in detected_centers)
        has_right = any(c[0] >= img_width // 2 for c in detected_centers)
        
        if (iter // 2) % 2 == 0:
            if not has_left:
                left_detected = False
        else:
            if not has_right:
                right_detected = False
        
        initial_superpill_data['left_detected'] = left_detected
        initial_superpill_data['right_detected'] = right_detected
        
        filtered_boxes = []
        filtered_centers = []
        
        for i, center in enumerate(initial_centers):
            if i in [0, 2] and left_detected:
                filtered_boxes.append(initial_boxes[i])
                filtered_centers.append(center)
            elif i in [1, 3] and right_detected:
                filtered_boxes.append(initial_boxes[i])
                filtered_centers.append(center)
        print(len(filtered_boxes))
        return {
            'superpill_boxes': filtered_boxes,
            'superpill_centers': filtered_centers
        }

def detect_doors():
    return {'door_centers':[[128,22],[128,202]]}

def detect_obstacles(env_img, args):
    """
    提取图片中RGB(228, 111, 111)颜色的所有区域作为障碍物掩码，并移除pill区域
    
    :param env_img: 输入图像
    :param args: 参数配置
    
    :return: 障碍物掩码
    """
    # 创建一个模拟的args对象，因为我们只需要size属性
    class MockArgs:
        def __init__(self, size=256):
            self.size = size
            # 添加capture属性以避免错误
            self.capture = getattr(args, 'capture', False)
    
    mock_args = MockArgs(getattr(args, 'size', 256))
    

    obstacle_color = (228, 111, 111)
    
    # 调整图像大小
    # resized_img = cv2.resize(env_img, (mock_args.size, mock_args.size))
    resized_img, _ = pad_image_to_size(env_img, (mock_args.size, mock_args.size))   

    # 创建障碍物掩码
    # 注意OpenCV使用BGR格式，所以我们需要转换颜色顺序
    bgr_color = np.array([obstacle_color[2], obstacle_color[1], obstacle_color[0]], dtype=np.uint8)
    obstacle_mask = cv2.inRange(resized_img, bgr_color, bgr_color)
    
    # 使用PillDetector检测pill区域
    pill_detector = PillDetector(resized_img, mock_args, iter_num=0, epoch=0)
    pill_positions, pill_count, pill_mask = pill_detector.detect_pills(
        obstacle_color, min_area=3, max_area=16, min_count=8, mask=True)
    
    # 从障碍物掩码中移除pill区域
    obstacle_mask = cv2.bitwise_and(obstacle_mask, cv2.bitwise_not(pill_mask))
    
    # 使用ObjectDetector检测superpill区域并移除
    detector = ObjectDetector(resized_img, mock_args, 1, 1)
    target_color =[np.array([228, 111, 111]) ]
    annotated_image, all_clusters = detector.extract_multiple_colors_clusters(
        target_colors = target_color,
        min_area=10,
        max_area=30,
        classify_objects=False
    )
    
    # 从检测到的聚类中创建superpill掩码
    superpill_mask = np.zeros_like(obstacle_mask)
    
    # 根据superpill的面积特征来识别它们，而不是使用label
    for cluster in all_clusters:
        x, y, w, h = cluster['bbox']
        area = w * h
        
        # 根据superpill的面积范围来识别（10到30像素的面积）
        # 但因为这是聚类结果，面积可能更大，所以使用更合适的标准
        if 10 <= area <= 30:
            cv2.rectangle(superpill_mask, (x, y), (x + w, y + h), 255, -1)
    
    # 从障碍物掩码中移除superpill区域
    obstacle_mask = cv2.bitwise_and(obstacle_mask, cv2.bitwise_not(superpill_mask))
    
    return obstacle_mask

def save_and_visualize_detection_results(env_img, all_game_info, frame_idx,epoch,args):
    file_name = args.your_mission_name  
    visualize_detection_results(env_img,all_game_info, frame_idx,epoch,file_name)
    save_ghosts_info(all_game_info, frame_idx,epoch,file_name)


def visualize_detection_results(env_img, all_game_info, frame_idx, epoch=0, file_name="default"):
    """
    可视化检测结果
    
    :param env_img: 原始环境图像
    :param all_game_info: detect_all_in_one函数返回的所有游戏信息
    :param frame_idx: 帧索引
    :param epoch: 训练轮次
    :param file_name: 文件名标识，用于创建子目录
    """
    # 调整图像大小以匹配检测结果
    # resized_img = cv2.resize(env_img, (160, 250))
    
    # 创建显示图像
    # display_img = resized_img.copy()
    display_img = env_img.copy()
    
    # 绘制ghost边界框（红色）
    ghost_boxes = all_game_info.get('ghosts_boxes', [])
    for i, bbox in enumerate(ghost_boxes):
        if i == 0:  # ghost0是一个包含4个边界框的列表
            for sub_bbox in bbox:
                if len(sub_bbox) == 4:  # 确保边界框格式正确
                    x1, y1, x2, y2 = sub_bbox
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        else:
            if len(bbox) == 4:  # 确保边界框格式正确
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    # 绘制ghost中心点（红色圆圈）
    ghost_centers = all_game_info.get('ghosts_centers', [])
    for i, center in enumerate(ghost_centers):
        if i == 0:  # ghost0是一个包含4个中心点的列表
            for sub_center in center:
                if len(sub_center) == 2:  # 确保中心点格式正确
                    cx, cy = sub_center
                    cv2.circle(display_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        else:
            if len(center) == 2:  # 确保中心点格式正确
                cx, cy = center
                cv2.circle(display_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    
    # 绘制pacman边界框（绿色）
    pacman_boxes = all_game_info.get('pacman_boxes', [])
    for bbox in pacman_boxes:
        if len(bbox) == 4:  # 确保边界框格式正确
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    
    # 绘制pacman中心点（绿色圆圈）
    pacman_centers = all_game_info.get('pacman_centers', [])
    for center in pacman_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 3, (0, 255, 0), -1)
    
    # 绘制superpill边界框（青色）
    superpill_boxes = all_game_info.get('superpill_boxes', [])
    for bbox in superpill_boxes:
        if len(bbox) == 4:  # 确保边界框格式正确
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
    
    # 绘制superpill中心点（青色圆圈）
    superpill_centers = all_game_info.get('superpill_centers', [])
    for center in superpill_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 3, (255, 255, 0), -1)
    
    # 绘制door中心点（紫色圆圈）
    door_centers = all_game_info.get('door_centers', [])
    for center in door_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 3, (255, 0, 255), -1)
    
    # 创建pill显示图像
    pill_img = display_img.copy()
    
    # 绘制pill中心点（黄色圆圈）
    pill_centers = all_game_info.get('pill_centers', [])
    for center in pill_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(pill_img, (int(cx), int(cy)), 3, (0, 255, 255), -1)
    
    # 显示图像
    plt.figure(figsize=(15, 5))
    
    # 显示带有边界框和中心点的图像
    plt.subplot(1, 3, 1)
    # 转换BGR到RGB以正确显示颜色
    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    plt.imshow(display_img_rgb)
    plt.title(f'Frame {frame_idx} - Bounding Boxes and Centers')
    plt.axis('off')
    
    # 显示pill位置
    plt.subplot(1, 3, 2)
    pill_img_rgb = cv2.cvtColor(pill_img, cv2.COLOR_BGR2RGB)
    plt.imshow(pill_img_rgb)
    plt.title(f'Frame {frame_idx} - Pill Positions')
    plt.axis('off')
    
    # 显示障碍物掩码
    plt.subplot(1, 3, 3)
    obstacles_mask = all_game_info.get('obstacles_mask', np.zeros((256, 256)))
    # obstacles_mask = cv2.resize(obstacles_mask, (160, 250))
    plt.imshow(obstacles_mask, cmap='gray')
    plt.title(f'Frame {frame_idx} - Obstacles Mask')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果图像
    save_dir = os.path.join("detection_results", file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用zfill方法确保帧编号始终有适当位数，支持超过999帧的情况
    frame_str = str(frame_idx).zfill(6)  # 支持最多999999帧
    plt.savefig(os.path.join(save_dir, f"{epoch}_detection_frame_{frame_str}.png"))
    plt.close()


def save_ghosts_info(all_game_info, frame_idx, epoch=0, file_name="default"):
    """
    保存ghosts_info到文本文件
    
    :param all_game_info: detect_all_in_one函数返回的所有游戏信息
    :param frame_idx: 帧索引
    :param epoch: 训练轮次
    :param file_name: 文件名标识，用于创建子目录
    """
    save_dir = os.path.join("detection_results", file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 准备要写入的文本内容
    ghosts_info_text = f"Epoch {epoch}, Frame {frame_idx}:\n"
    ghosts_info_text += f"Ghost Num: {all_game_info.get('ghost_num', 0)}\n"
    ghosts_info_text += f"Ghosts Boxes: {all_game_info.get('ghosts_boxes', [])}\n"
    ghosts_info_text += f"Ghosts Centers: {all_game_info.get('ghosts_centers', [])}\n"
    ghosts_info_text += f"Pacman Decision: {all_game_info.get('pacman_decision', {})}\n"
    ghosts_info_text += "-" * 50 + "\n"
    
    # 写入文件，使用统一的文件名，所有epoch和frame信息都在内容中体现
    txt_file_path = os.path.join(save_dir, "ghosts_info.txt")
    with open(txt_file_path, "a") as f:
        f.write(ghosts_info_text)
