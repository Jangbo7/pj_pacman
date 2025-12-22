from .info_utils import *
import numpy as np

def pacman_decision(pacman_info, obstacle_info):
    """
    根据pacman位置和障碍物信息判断pacman可以移动的方向
    
    :param pacman_info: pacman检测信息，包含边界框等
    :param obstacle_info: 障碍物掩码 (二值图像)
    
    
    :return: 
        - 'Caught' 如果没有检测到pacman
        - 字典 {up: 1/0, down: 1/0, left: 1/0, right: 1/0} 表示可行动方向
          1表示可以移动，0表示不可移动
    """
    
    if len(pacman_info['pacman_boxes']) == 0:
        print('Caught')
        return 'Caught'
    else:
        # 获取pacman边界框坐标
        pacman_box = pacman_info['pacman_boxes'][0]  # 只取第一个（也是唯一一个）pacman
        x1, y1, x2, y2 = pacman_box
        
        # 获取障碍物掩码的尺寸
        mask_height, mask_width = obstacle_info.shape
        
        # 初始化可行动方向字典，默认都可以移动
        directions = {'up': 1, 'down': 1, 'left': 1, 'right': 1}
        
        # 首先将整个边界框缩小，每条边向内收缩2个像素
        shrink_amount = 2
        shrunk_x1 = min(x1 + shrink_amount, x2 - shrink_amount)
        shrunk_y1 = min(y1 + shrink_amount, y2 - shrink_amount)
        shrunk_x2 = max(x1 + shrink_amount, x2 - shrink_amount)
        shrunk_y2 = max(y1 + shrink_amount, y2 - shrink_amount)
        
        # 确保缩小后的边界框仍然有效
        if shrunk_x1 >= shrunk_x2 or shrunk_y1 >= shrunk_y2:
            # 如果缩小后边界框无效，则使用原始边界框
            shrunk_x1, shrunk_y1, shrunk_x2, shrunk_y2 = x1, y1, x2, y2
        
        # 设置容忍度（默认为5像素）
        tolerance = 5
        
        # 检查向上移动
        # 在shrunk bbox的基础上，向上扩展tolerance像素，检查是否与障碍物重合
        up_check_y1 = max(0, shrunk_y1 - tolerance)
        up_check_y2 = shrunk_y1
        if up_check_y1 < up_check_y2:
            up_region = obstacle_info[up_check_y1:up_check_y2, shrunk_x1:shrunk_x2]
            if np.any(up_region > 0):  # 检查是否有障碍物重合
                directions['up'] = 0
        
        # 检查向下移动
        # 在shrunk bbox的基础上，向下扩展tolerance像素，检查是否与障碍物重合
        down_check_y1 = shrunk_y2
        down_check_y2 = min(mask_height, shrunk_y2 + tolerance)
        if down_check_y1 < down_check_y2:
            down_region = obstacle_info[down_check_y1:down_check_y2, shrunk_x1:shrunk_x2]
            if np.any(down_region > 0):  # 检查是否有障碍物重合
                directions['down'] = 0
        
        # 检查向左移动
        # 在shrunk bbox的基础上，向左扩展tolerance像素，检查是否与障碍物重合
        left_check_x1 = max(0, shrunk_x1 - tolerance)
        left_check_x2 = shrunk_x1
        if left_check_x1 < left_check_x2:
            left_region = obstacle_info[shrunk_y1:shrunk_y2, left_check_x1:left_check_x2]
            if np.any(left_region > 0):  # 检查是否有障碍物重合
                directions['left'] = 0
        
        # 检查向右移动
        # 在shrunk bbox的基础上，向右扩展tolerance像素，检查是否与障碍物重合
        right_check_x1 = shrunk_x2
        right_check_x2 = min(mask_width, shrunk_x2 + tolerance)
        if right_check_x1 < right_check_x2:
            right_region = obstacle_info[shrunk_y1:shrunk_y2, right_check_x1:right_check_x2]
            if np.any(right_region > 0):  # 检查是否有障碍物重合
                directions['right'] = 0
        
        return directions
    

if __name__ == '__main__': 
    import sys
    import os
    # 添加项目根目录到路径中
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    import cv2
    from .info_utils import detect_gp_with_yolo, detect_obstacles
    
    env_path = "../datas/experiment.png"
    model_path = "../runs/detect/yolov8n_custom_training2/weights/best.pt"
    
    # 检查文件是否存在
    if os.path.exists(env_path) and os.path.exists(model_path):
        env = cv2.imread(env_path) 
        env_img = cv2.resize(env, (256, 256))
        class MockArgs:
            def __init__(self):
                self.size = 256
                self.capture = False
        
        args = MockArgs()
        from ultralytics import YOLO
        model = YOLO(model_path)
        _, pacman_info = detect_gp_with_yolo(env_img, model)
        obstacle_info = detect_obstacles(env_img, args)
        result = pacman_decision(pacman_info, obstacle_info)    
        print(result)
    else:
        print(f"文件不存在: {env_path} 或 {model_path}")
