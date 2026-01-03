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
    pill_color = (228,111,111)
    if (np.sum((env_img[:, :, 2] == 228) & (env_img[:, :, 1] == 111) & (env_img[:, :, 0] == 111)) > 0):
        pill_color = (228,111,111)
    elif (np.sum((env_img[:, :, 2] == 101) & (env_img[:, :, 1] == 111) & (env_img[:, :, 0] == 228)) > 0):
        pill_color = (101,111,228)
    pill_detector = PillDetector(env_img, args, iter_num=0, epoch=0)
    pill_positions, pill_count = pill_detector.detect_pills(target_colors=pill_color, min_area=3, max_area=16, min_count=8)
    
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


def detect_superpill(env_img, former_superpill_info, iter=0):
    """
    使用ObjectDetector检测超级药丸的位置，pipeline如下：
    
    iter == 0 时：
        - 寻找superpills
        - 在y轴对称位置生成新的superpills（单帧只显示一侧，实际初始有两对）
        - 记录所有坐标作为初始值
    
    iter > 0 时：
        - 当 (iter//2)%2 == 0：检测左侧superpills
            * 检查初始位置中的左侧药丸是否在图中检测到
            * 未检测到的从结果中删除
            * 如果左侧检测到新的superpill且不在列表中，则添加
        
        - 当 (iter//2)%2 == 1：检测右侧superpills
            * 检查初始位置中的右侧药丸是否在图中检测到
            * 未检测到的从结果中删除
            * 如果右侧检测到新的superpill且不在列表中，则添加
    
    :param env_img: 输入图像
    :param former_superpill_info: 前一次detect_superpill函数结果，格式为：
        {
            'superpill_boxes': [[x1,y1,x2,y2],...],
            'superpill_centers': [[x,y],...],
            'initial_data': {...}  # iter==0时保存的初始数据
        }
    :param iter: 当前帧序号（0开始）
    :return: 包含超级药丸边界框和中心点的字典
             格式: {
                 'superpill_boxes': [[x1, y1, x2, y2], ...],
                 'superpill_centers': [[x, y], ...],
                 'initial_data': {...}  # 保存初始数据供后续使用
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
        if len(detected_boxes) == 0:
            return {
                'superpill_boxes': [],
                'superpill_centers': [],
                'initial_data': None
            }
        
        # 检测到的是左侧还是右侧（根据第一个检测到的 superpill 判断）
        first_center = detected_centers[0]
        is_detected_left = first_center[0] < img_width // 2
        
        # 根据 y 轴对称生成另一侧的 superpill
        generated_boxes = []
        generated_centers = []
        
        for box, center in zip(detected_boxes, detected_centers):
            if is_detected_left:
                # 检测到左侧，生成右侧
                new_center_x = img_width - center[0]
                new_center = [new_center_x, center[1]]
                new_box_x1 = img_width - box[2]
                new_box_x2 = img_width - box[0]
                new_box = [new_box_x1, box[1], new_box_x2, box[3]]
            else:
                # 检测到右侧，生成左侧
                new_center_x = img_width - center[0]
                new_center = [new_center_x, center[1]]
                new_box_x1 = img_width - box[2]
                new_box_x2 = img_width - box[0]
                new_box = [new_box_x1, box[1], new_box_x2, box[3]]
            
            generated_centers.append(new_center)
            generated_boxes.append(new_box)
        
        # 合并检测到的和生成的 superpills
        if is_detected_left:
            all_centers = detected_centers + generated_centers
            all_boxes = detected_boxes + generated_boxes
        else:
            all_centers = generated_centers + detected_centers
            all_boxes = generated_boxes + detected_boxes
        
        # 按 y 坐标排序（上下位置）
        sorted_pairs = sorted(zip(all_centers, all_boxes), key=lambda p: p[0][1])
        sorted_centers = [p[0] for p in sorted_pairs]
        sorted_boxes = [p[1] for p in sorted_pairs]
        print(sorted_boxes)
        initial_data = {
            'boxes': [sorted_boxes[0], sorted_boxes[2], 
                      sorted_boxes[1], sorted_boxes[3]],
            'centers': [sorted_centers[0], sorted_centers[2],
                        sorted_centers[1], sorted_centers[3]],
            'left_indices': [0, 2],
            'right_indices': [1, 3]
        }
        
        return {
            'superpill_boxes': initial_data['boxes'],
            'superpill_centers': initial_data['centers'],
            'initial_data': initial_data
        }
    else:
        """
        iter > 0 的处理逻辑：
        
        直接对上一帧的superpill结果进行分析处理，而非使用initial_data：
        
        1. 获取上一帧的superpill信息（former_superpill_info）
           - superpill_boxes: 上一帧检测到的超级药丸边界框列表
           - superpill_centers: 上一帧检测到的超级药丸中心点列表
        
        2. 根据当前帧号判断应该检测哪一侧：
           - 当 (iter//2)%2 == 0 时，检测左侧超级药丸
           - 当 (iter//2)%2 == 1 时，检测右侧超级药丸
           （//2是因为超级药丸每隔2帧变化一次显示位置）
        
        3. 对于应该检测的那一侧（以左侧为例）：
           a) 分离上一帧结果中的左侧和右侧superpill（基于x坐标判断）
              * left_indices: 上一帧中左侧superpill的索引列表
              * right_indices: 上一帧中右侧superpill的索引列表
            
           b) 以上一帧结果为基础，复制到result_boxes和result_centers
            
           c) 步骤1：检查上一帧中该侧的superpill在当前帧中是否被找到
              * 遍历该侧所有superpill的中心点
              * 在当前帧检测结果中寻找匹配的药丸（曼哈顿距离<10像素）
              * 找到：不操作（保留该superpill）
              * 没找到：记录该索引待删除
            
           d) 步骤2：检查是否有新的superpill需要添加
              * 遍历当前帧检测到的superpill
              * 检查是否已经存在于结果中（避免重复）
              * 如果是新的superpill（距离>=10像素），添加到结果中
        
        4. 对于不应该检测的另一侧：
           * 直接保留上一帧的superpill（因为它们应该存在但当前帧不显示）
           * 不进行任何修改
        
        :return: 返回处理后的superpill信息（基于上一帧结果进行增删）
                 格式: {
                     'superpill_boxes': [[x1, y1, x2, y2], ...],
                     'superpill_centers': [[x, y], ...]
                 }
        """
        
        # 直接使用上一帧的superpill信息，不需要initial_data
        if former_superpill_info is None:
            return {
                'superpill_boxes': [],
                'superpill_centers': []
            }
        
        # 获取上一帧的superpill结果
        prev_boxes = former_superpill_info.get('superpill_boxes', [])
        prev_centers = former_superpill_info.get('superpill_centers', [])
        
        # 如果上一帧没有superpill，返回空结果
        if len(prev_boxes) == 0:
            return {
                'superpill_boxes': [],
                'superpill_centers': []
            }
        
        # 根据迭代次数判断当前帧应该检测哪一侧
        is_left_side = (iter // 2) % 2 == 0
        
        if is_left_side:
            # ========== 处理左侧超级药丸 ==========
            
            # 分离左侧和右侧的superpill（基于x坐标判断）
            left_indices = []
            right_indices = []
            
            for i, center in enumerate(prev_centers):
                if center[0] < img_width // 2:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
            
            # 复制上一帧结果作为基础
            result_boxes = prev_boxes.copy()
            result_centers = prev_centers.copy()
            
            # 步骤1：检查上一帧中左侧的superpill是否在当前帧中检测到
            # 找出需要检查的索引（左侧的索引）
            indices_to_check = left_indices
            
            # 收集需要删除的索引（避免在遍历时修改列表）
            indices_to_remove = []
            
            for idx in indices_to_check:
                init_center = prev_centers[idx]
                found = False
                for det_center in detected_centers:
                    # 只检查左侧的检测结果
                    if det_center[0] < img_width // 2:
                        # 计算曼哈顿距离
                        dist = abs(det_center[0] - init_center[0]) + abs(det_center[1] - init_center[1])
                        if dist < 10:
                            found = True
                            break
                # 没找到：记录该索引待删除
                if not found:
                    # 找到该索引在result中的实际位置
                    actual_idx = result_centers.index(init_center)
                    indices_to_remove.append(actual_idx)
            
            # 从高到低删除，避免索引偏移
            for idx in sorted(indices_to_remove, reverse=True):
                result_boxes.pop(idx)
                result_centers.pop(idx)
            
            # 步骤2：检查是否有新的superpill需要添加
            # 遍历当前帧检测到的superpill
            for j, (box, center) in enumerate(zip(detected_boxes, detected_centers)):
                if center[0] < img_width // 2:
                    # 检查是否已经存在于结果中
                    is_new = True
                    for existing_center in result_centers:
                        dist = abs(center[0] - existing_center[0]) + abs(center[1] - existing_center[1])
                        if dist < 10:
                            is_new = False
                            break
                    # 如果是新的superpill，添加到结果中
                    if is_new:
                        result_boxes.append(box)
                        result_centers.append(center)
            
            return {
                'superpill_boxes': result_boxes,
                'superpill_centers': result_centers
            }
        else:
            # ========== 处理右侧超级药丸 ==========
            
            # 分离左侧和右侧的superpill（基于x坐标判断）
            left_indices = []
            right_indices = []
            
            for i, center in enumerate(prev_centers):
                if center[0] < img_width // 2:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
            
            # 复制上一帧结果作为基础
            result_boxes = prev_boxes.copy()
            result_centers = prev_centers.copy()
            
            # 步骤1：检查上一帧中右侧的superpill是否在当前帧中检测到
            indices_to_check = right_indices
            
            # 收集需要删除的索引（避免在遍历时修改列表）
            indices_to_remove = []
            
            for idx in indices_to_check:
                init_center = prev_centers[idx]
                found = False
                for det_center in detected_centers:
                    # 只检查右侧的检测结果
                    if det_center[0] >= img_width // 2:
                        # 计算曼哈顿距离
                        dist = abs(det_center[0] - init_center[0]) + abs(det_center[1] - init_center[1])
                        if dist < 10:
                            found = True
                            break
                # 没找到：记录该索引待删除
                if not found:
                    # 找到该索引在result中的实际位置
                    actual_idx = result_centers.index(init_center)
                    indices_to_remove.append(actual_idx)
            
            # 从高到低删除，避免索引偏移
            for idx in sorted(indices_to_remove, reverse=True):
                result_boxes.pop(idx)
                result_centers.pop(idx)
            
            # 步骤2：检查是否有新的superpill需要添加
            for j, (box, center) in enumerate(zip(detected_boxes, detected_centers)):
                if center[0] >= img_width // 2:
                    # 检查是否已经存在于结果中
                    is_new = True
                    for existing_center in result_centers:
                        dist = abs(center[0] - existing_center[0]) + abs(center[1] - existing_center[1])
                        if dist < 10:
                            is_new = False
                            break
                    # 如果是新的superpill，添加到结果中
                    if is_new:
                        result_boxes.append(box)
                        result_centers.append(center)
            
            return {
                'superpill_boxes': result_boxes,
                'superpill_centers': result_centers
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
            # for sub_bbox in bbox:
            #     if len(sub_bbox) == 4:  # 确保边界框格式正确
            #         x1, y1, x2, y2 = sub_bbox
            #         cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            pass
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
    
    # 创建legal_action显示图像
    action_img = env_img.copy()
    
    # 获取pacman中心位置和legal_action信息
    pacman_centers = all_game_info.get('pacman_centers', [])
    legal_action = all_game_info.get('pacman_decision', {})
    
    if pacman_centers and legal_action:
        # 获取pacman中心坐标
        pacman_center = pacman_centers[0]  # 假设只有一个pacman
        cx, cy = pacman_center
        
        # 定义箭头长度
        arrow_length = 20
        
        # 根据legal_action绘制箭头
        if legal_action.get('up', 0) == 1:
            cv2.arrowedLine(action_img, (int(cx), int(cy)), (int(cx), int(cy) - arrow_length), (0, 255, 0), 2, tipLength=0.3)
        if legal_action.get('down', 0) == 1:
            cv2.arrowedLine(action_img, (int(cx), int(cy)), (int(cx), int(cy) + arrow_length), (0, 255, 0), 2, tipLength=0.3)
        if legal_action.get('left', 0) == 1:
            cv2.arrowedLine(action_img, (int(cx), int(cy)), (int(cx) - arrow_length, int(cy)), (0, 255, 0), 2, tipLength=0.3)
        if legal_action.get('right', 0) == 1:
            cv2.arrowedLine(action_img, (int(cx), int(cy)), (int(cx) + arrow_length, int(cy)), (0, 255, 0), 2, tipLength=0.3)
    
    # 显示图像
    plt.figure(figsize=(20, 5))
    
    # 显示带有边界框和中心点的图像
    plt.subplot(1, 4, 1)
    # 转换BGR到RGB以正确显示颜色
    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    plt.imshow(display_img_rgb)
    plt.title(f'Frame {frame_idx} - Bounding Boxes and Centers')
    plt.axis('off')
    
    # 显示pill位置
    plt.subplot(1, 4, 2)
    pill_img_rgb = cv2.cvtColor(pill_img, cv2.COLOR_BGR2RGB)
    plt.imshow(pill_img_rgb)
    plt.title(f'Frame {frame_idx} - Pill Positions')
    plt.axis('off')
    
    # 显示障碍物掩码
    plt.subplot(1, 4, 3)
    obstacles_mask = all_game_info.get('obstacles_mask', np.zeros((256, 256)))
    # obstacles_mask = cv2.resize(obstacles_mask, (160, 250))
    plt.imshow(obstacles_mask, cmap='gray')
    plt.title(f'Frame {frame_idx} - Obstacles Mask')
    plt.axis('off')
    
    # 显示legal_action
    plt.subplot(1, 4, 4)
    action_img_rgb = cv2.cvtColor(action_img, cv2.COLOR_BGR2RGB)
    plt.imshow(action_img_rgb)
    plt.title(f'Frame {frame_idx} - Legal Actions')
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


def save_and_visualize_detection_results_4vlm(env_img, all_game_info, frame_idx, epoch=0, args=None):
    """
    VLM专用的可视化检测结果函数
    
    :param env_img: 原始环境图像
    :param all_game_info: detect_all_in_one函数返回的所有游戏信息
    :param frame_idx: 帧索引
    :param epoch: 训练轮次
    :param args: 配置参数对象，包含your_mission_name属性
    """
    if args is None:
        file_name = "default"
    else:
        file_name = args.your_mission_name + "_4vlm"
    
    # 创建显示图像
    display_img = env_img.copy()
    
    # 可视化4ghosts_boxes（用红色框和中心点）
    ghosts_boxes_4 = all_game_info.get('4ghosts_boxes', [])
    ghosts_centers_4 = all_game_info.get('4ghosts_centers', [])
    
    # 绘制4个ghost的边界框和中心点
    for i, (bbox, center) in enumerate(zip(ghosts_boxes_4, ghosts_centers_4)):
        if len(bbox) == 4:  # 确保边界框格式正确
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 红色框
        if len(center) == 2:  # 确保中心点格式正确
            cx, cy = center
            cv2.circle(display_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)  # 红色实心圆
    
    # 可视化legal_action（用绿色箭头从pacman位置指向可行动方向）
    pacman_centers = all_game_info.get('pacman_centers', [])
    legal_action = all_game_info.get('pacman_decision', {})
    
    if pacman_centers and legal_action:
        # 获取pacman中心坐标
        pacman_center = pacman_centers[0]  # 假设只有一个pacman
        cx, cy = pacman_center
        
        # 定义箭头长度
        arrow_length = 25
        
        # 根据legal_action绘制箭头
        if legal_action.get('up', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx), int(cy) - arrow_length), (0, 255, 0), 3, tipLength=0.3)
        if legal_action.get('down', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx), int(cy) + arrow_length), (0, 255, 0), 3, tipLength=0.3)
        if legal_action.get('left', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx) - arrow_length, int(cy)), (0, 255, 0), 3, tipLength=0.3)
        if legal_action.get('right', 0) == 1:
            cv2.arrowedLine(display_img, (int(cx), int(cy)), (int(cx) + arrow_length, int(cy)), (0, 255, 0), 3, tipLength=0.3)
    
    # 添加文字标注
    # 在图像左上角添加信息文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    color = (255, 255, 255)  # 白色文字
    thickness = 1
    
    # 显示帧信息和ghost数量
    info_text = f"Frame: {frame_idx}, Epoch: {epoch}, Ghosts: {len(ghosts_boxes_4)}"
    cv2.putText(display_img, info_text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # 显示legal action信息
    if legal_action:
        actions = []
        for direction in ['up', 'down', 'left', 'right']:
            if legal_action.get(direction, 0) == 1:
                actions.append(direction)
        action_text = f"Actions: {', '.join(actions) if actions else 'None'}"
        cv2.putText(display_img, action_text, (10, 50), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # 保存结果图像
    save_dir = os.path.join("detection_results", file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用zfill方法确保帧编号始终有适当位数，支持超过999帧的情况
    frame_str = str(frame_idx).zfill(6)  # 支持最多999999帧
    image_path = os.path.join(save_dir, f"{epoch}_4vlm_frame_{frame_str}.png")
    
    # 保存图像
    cv2.imwrite(image_path, display_img)
    
    print(f"4VLM可视化保存完成: {image_path}")


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
