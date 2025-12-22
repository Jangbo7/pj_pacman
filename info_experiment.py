from utils_all.info_utils import *
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_game_info(env_img, iter):
    """
    获取游戏信息并在图像上可视化检测结果
    
    :param env: 游戏环境
    :param iter: 迭代次数
    :return: 游戏信息字典
    """
    # 获取环境图像
    # env_img = env.render(mode='rgb_array')
    
    # 创建一个模拟的args对象
    class MockArgs:
        def __init__(self):
            self.size = 256
            self.capture = False
    
    args = MockArgs()
    
    # 调用各个检测函数获取信息
    # 注意：这里需要根据实际情况提供正确的模型路径
    from ultralytics import YOLO
    model = YOLO("runs/detect/yolov8n_custom_training2/weights/best.pt")
    gp_info = detect_gp_with_yolo(env_img, model)
    # g_info = detect_g_with_detector(env_img, args, None)
    pill_info = detect_pills_with_detector(env_img, args, None)
    superpill_info = detect_superpill(env_img)
    door_info = detect_doors()
    obstacles_mask = detect_obstacles(env_img, args)
    
    # 可视化所有检测结果
    visualize_detections(env_img, gp_info, pill_info, superpill_info, door_info, obstacles_mask)
    
    # 构建返回信息字典
    all_info = {
        'gp_info': gp_info,
        'pill_info': pill_info,
        'superpill_info': superpill_info,
        'door_info': door_info,
        'obstacles_mask': obstacles_mask
    }
    
    return all_info


def visualize_detections(env_img, gp_info, pill_info, superpill_info, door_info, obstacles_mask):
    """
    可视化所有检测结果
    
    :param env_img: 原始环境图像
    :param gp_info: ghost和pacman检测信息
    :param pill_info: pill检测信息
    :param superpill_info: superpill检测信息
    :param door_info: door检测信息
    :param obstacles_mask: 障碍物掩码
    """
    # 调整图像大小以匹配检测结果
    resized_img = cv2.resize(env_img, (256, 256))
    
    # 创建主显示图像（带边界框和中心点）
    display_img = resized_img.copy()
    
    # 绘制ghost边界框（绿色）
    for bbox in gp_info.get('ghost_boxes', []):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制ghost中心点（绿色圆圈）
    for center in gp_info.get('ghost_centers', []):
        cx, cy = center
        cv2.circle(display_img, (cx, cy), 5, (0, 255, 0), -1)
    
    # 绘制pacman边界框（蓝色）
    for bbox in gp_info.get('pacman_boxes', []):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # 绘制pacman中心点（蓝色圆圈）
    for center in gp_info.get('pacman_centers', []):
        cx, cy = center
        cv2.circle(display_img, (cx, cy), 5, (255, 0, 0), -1)
    
    # 绘制superpill边界框（青色）
    for bbox in superpill_info.get('superpill_boxes', []):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    # 绘制superpill中心点（青色圆圈）
    for center in superpill_info.get('superpill_centers', []):
        cx, cy = center
        cv2.circle(display_img, (cx, cy), 5, (255, 255, 0), -1)
    
    # 绘制door中心点（紫色圆圈）
    for center in door_info.get('door_centers', []):
        cx, cy = center
        cv2.circle(display_img, (cx, cy), 5, (255, 0, 255), -1)
    
    # 创建pill显示图像
    pill_img = resized_img.copy()
    
    # 绘制pill中心点（黄色圆圈）
    for center in pill_info.get('pill_centers', []):
        cx, cy = center
        cv2.circle(pill_img, (cx, cy), 3, (0, 255, 255), -1)
    
    # 显示图像
    plt.figure(figsize=(15, 5))
    
    # 显示带有边界框和中心点的图像
    plt.subplot(1, 3, 1)
    # 转换BGR到RGB以正确显示颜色
    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    plt.imshow(display_img_rgb)
    plt.title('Bounding Boxes and Centers')
    plt.axis('off')
    
    # 显示pill位置
    plt.subplot(1, 3, 2)
    pill_img_rgb = cv2.cvtColor(pill_img, cv2.COLOR_BGR2RGB)
    plt.imshow(pill_img_rgb)
    plt.title('Pill Positions')
    plt.axis('off')
    
    # 显示障碍物掩码
    plt.subplot(1, 3, 3)
    plt.imshow(obstacles_mask, cmap='gray')
    plt.title('Obstacles Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果图像
    cv2.imwrite('detection_results_bboxes.png', display_img)
    cv2.imwrite('detection_results_pills.png', pill_img)
    cv2.imwrite('detection_results_obstacles.png', obstacles_mask)


if __name__ == '__main__':
    # 示例用法
    env = cv2.imread("datas/experiment.png")
    env = cv2.resize(env, (256, 256))
    iter = 0
    results = get_game_info(env, iter)  
    print("Info Gainer Module")
    print("This module provides functions to gather and visualize game information.")