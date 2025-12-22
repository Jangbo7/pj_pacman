from ultralytics import YOLO
from supervisor.pill_detector import PillDetector
from supervisor.object_detector import ObjectDetector
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import os





def detect_gp_with_yolo(image, path):
    """
    Use yolo to detect ghost and pacman
    
    :param image: 输入图像
    :param path: 模型路径

    :return: 包含ghost和pacman边界框及中心点的字典
             格式: {
                 'ghost_boxes': [[x1, y1, x2, y2], ...],
                 'ghost_centers': [[x, y], ...],
                 'pacman_boxes': [[x1, y1, x2, y2], ...],
                 'pacman_centers': [[x, y], ...]
             }
    """
    pr_image = cv2.resize(image, (256, 256))
    model = YOLO(path)
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
        result_dict_p['pacman_boxes'].append(best_pacman_box)
        result_dict_p['pacman_centers'].append(best_pacman_center)
    
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
    if min_distance > 5 and ghost_num < 4:
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
    pill_color = (223, 192, 111)
    pill_detector = PillDetector(env_img, args, iter, epoch=0)
    pill_positions, pill_count = pill_detector.detect_pills(pill_color, min_area=4, 
                                                            max_area=20, min_count=8)
    
    # 构造返回字典
    result_dict = {
        'pill_centers': list(pill_positions),  # pill_positions本身就是一个(x, y)坐标列表
        'pill_num': [pill_count]  # 将pill数量包装在列表中
    }
    
    return result_dict

def detect_g_with_detector(env_img, args, path):
    """
    使用ObjectDetector检测ghost位置
    
    :param env_img: 输入图像
    :param args: 参数配置
    :param path: 未使用参数（为了保持接口一致）
    
    :return: 包含ghost边界框及中心点的字典
             格式: {
                 'ghost_boxes': [[x1, y1, x2, y2], ...],
                 'ghost_centers': [[x, y], ...]
             }
    """
    object_detector = ObjectDetector(env_img, args, iter, epoch=0)
    annotated_image, ghost_objects = object_detector.extract_complex_objects_with_holes([(252, 144, 200)], 
                                                                                        min_area=36, max_area=400)
    
    # 初始化返回字典
    result_dict = {
        'ghost_boxes': [],
        'ghost_centers': []
    }
    
    # 处理检测到的ghost对象
    for obj in ghost_objects:
        x, y, w, h = obj['bbox']
        # 计算边界框坐标 [x1, y1, x2, y2]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        result_dict['ghost_boxes'].append([x1, y1, x2, y2])
        
        # 计算中心点坐标
        center_x = x + w // 2
        center_y = y + h // 2
        result_dict['ghost_centers'].append([center_x, center_y])
    
    return result_dict


def detect_superpill(env_img):
    # superpill_color = (252, 144, 200)
    # object_detector = ObjectDetector(env_img)
    # annotated_image, superpill_objects = object_detector.extract_multiple_colors_clusters(target_colors=superpill_objects,
    #                                                                                        min_area=36, max_area=40,classify_objects=True)
    return {'superpill_boxes': [[240,40,246,50],[240,175,246,185],[10,40,16,50],[10,175,16,185]],
            'superpill_centers': [[243, 45], [243, 180], [13, 45], [13, 180]]}

def detect_doors():
    return {'door_centers':[[128,20],[128,200]] }
def detect_obstacles(env_img, args):
    """
    提取图片中RGB(223, 192, 111)颜色的所有区域作为障碍物掩码，并移除pill区域
    
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
    
    # 提取RGB(223, 192, 111)颜色区域作为初始障碍物掩码
    obstacle_color = (223, 192, 111)
    
    # 调整图像大小
    resized_img = cv2.resize(env_img, (mock_args.size, mock_args.size))
    
    # 创建障碍物掩码
    # 注意OpenCV使用BGR格式，所以我们需要转换颜色顺序
    bgr_color = np.array([obstacle_color[2], obstacle_color[1], obstacle_color[0]], dtype=np.uint8)
    obstacle_mask = cv2.inRange(resized_img, bgr_color, bgr_color)
    
    # 使用PillDetector检测pill区域
    pill_detector = PillDetector(resized_img, mock_args, iter_num=0, epoch=0)
    pill_positions, pill_count, pill_mask = pill_detector.detect_pills(
        obstacle_color, min_area=1, max_area=20, min_count=8, mask=True)
    
    # 从障碍物掩码中移除pill区域
    # 通过将pill_mask区域置零来实现
    obstacle_mask = cv2.bitwise_and(obstacle_mask, cv2.bitwise_not(pill_mask))
    
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
    resized_img = cv2.resize(env_img, (256, 256))
    
    # 创建显示图像
    display_img = resized_img.copy()
    
    # 绘制ghost边界框（绿色）
    ghost_boxes = all_game_info.get('ghosts_boxes', [])
    for bbox in ghost_boxes:
        if len(bbox) == 4:  # 确保边界框格式正确
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制ghost中心点（绿色圆圈）
    ghost_centers = all_game_info.get('ghosts_centers', [])
    for center in ghost_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (cx, cy), 5, (0, 255, 0), -1)
    
    # 绘制pacman边界框（蓝色）
    pacman_boxes = all_game_info.get('pacman_boxes', [])
    for bbox in pacman_boxes:
        if len(bbox) == 4:  # 确保边界框格式正确
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # 绘制pacman中心点（蓝色圆圈）
    pacman_centers = all_game_info.get('pacman_centers', [])
    for center in pacman_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (cx, cy), 5, (255, 0, 0), -1)
    
    # 绘制superpill边界框（青色）
    superpill_boxes = all_game_info.get('superpill_boxes', [])
    for bbox in superpill_boxes:
        if len(bbox) == 4:  # 确保边界框格式正确
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    # 绘制superpill中心点（青色圆圈）
    superpill_centers = all_game_info.get('superpill_centers', [])
    for center in superpill_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (cx, cy), 5, (255, 255, 0), -1)
    
    # 绘制door中心点（紫色圆圈）
    door_centers = all_game_info.get('door_centers', [])
    for center in door_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (cx, cy), 5, (255, 0, 255), -1)
    
    # 创建pill显示图像
    pill_img = resized_img.copy()
    
    # 绘制pill中心点（黄色圆圈）
    pill_centers = all_game_info.get('pill_centers', [])
    for center in pill_centers:
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(pill_img, (cx, cy), 3, (0, 255, 255), -1)
    
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
