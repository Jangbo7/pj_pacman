import gymnasium as gym
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from detect_all import detect_all_in_one
from utils_all.game_utils import create_pacman_environment


# def visualize_detection_results(env_img, all_game_info, frame_idx):
#     """
#     可视化检测结果
    
#     :param env_img: 原始环境图像
#     :param all_game_info: detect_all_in_one函数返回的所有游戏信息
#     :param frame_idx: 帧索引
#     """
#     # 调整图像大小以匹配检测结果
#     resized_img = cv2.resize(env_img, (256, 256))
    
#     # 创建显示图像
#     display_img = resized_img.copy()
    
#     # 绘制ghost边界框（绿色）
#     ghost_boxes = all_game_info.get('ghosts_boxes', [])
#     for bbox in ghost_boxes:
#         if len(bbox) == 4:  # 确保边界框格式正确
#             x1, y1, x2, y2 = bbox
#             cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     # 绘制ghost中心点（绿色圆圈）
#     ghost_centers = all_game_info.get('ghosts_centers', [])
#     for center in ghost_centers:
#         if len(center) == 2:  # 确保中心点格式正确
#             cx, cy = center
#             cv2.circle(display_img, (cx, cy), 5, (0, 255, 0), -1)
    
#     # 绘制pacman边界框（蓝色）
#     pacman_boxes = all_game_info.get('pacman_boxes', [])
#     for bbox in pacman_boxes:
#         if len(bbox) == 4:  # 确保边界框格式正确
#             x1, y1, x2, y2 = bbox
#             cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
#     # 绘制pacman中心点（蓝色圆圈）
#     pacman_centers = all_game_info.get('pacman_centers', [])
#     for center in pacman_centers:
#         if len(center) == 2:  # 确保中心点格式正确
#             cx, cy = center
#             cv2.circle(display_img, (cx, cy), 5, (255, 0, 0), -1)
    
#     # 绘制superpill边界框（青色）
#     superpill_boxes = all_game_info.get('superpill_boxes', [])
#     for bbox in superpill_boxes:
#         if len(bbox) == 4:  # 确保边界框格式正确
#             x1, y1, x2, y2 = bbox
#             cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
#     # 绘制superpill中心点（青色圆圈）
#     superpill_centers = all_game_info.get('superpill_centers', [])
#     for center in superpill_centers:
#         if len(center) == 2:  # 确保中心点格式正确
#             cx, cy = center
#             cv2.circle(display_img, (cx, cy), 5, (255, 255, 0), -1)
    
#     # 绘制door中心点（紫色圆圈）
#     door_centers = all_game_info.get('door_centers', [])
#     for center in door_centers:
#         if len(center) == 2:  # 确保中心点格式正确
#             cx, cy = center
#             cv2.circle(display_img, (cx, cy), 5, (255, 0, 255), -1)
    
#     # 创建pill显示图像
#     pill_img = resized_img.copy()
    
#     # 绘制pill中心点（黄色圆圈）
#     pill_centers = all_game_info.get('pill_centers', [])
#     for center in pill_centers:
#         if len(center) == 2:  # 确保中心点格式正确
#             cx, cy = center
#             cv2.circle(pill_img, (cx, cy), 3, (0, 255, 255), -1)
    
#     # 显示图像
#     plt.figure(figsize=(15, 5))
    
#     # 显示带有边界框和中心点的图像
#     plt.subplot(1, 3, 1)
#     # 转换BGR到RGB以正确显示颜色
#     display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
#     plt.imshow(display_img_rgb)
#     plt.title(f'Frame {frame_idx} - Bounding Boxes and Centers')
#     plt.axis('off')
    
#     # 显示pill位置
#     plt.subplot(1, 3, 2)
#     pill_img_rgb = cv2.cvtColor(pill_img, cv2.COLOR_BGR2RGB)
#     plt.imshow(pill_img_rgb)
#     plt.title(f'Frame {frame_idx} - Pill Positions')
#     plt.axis('off')
    
#     # 显示障碍物掩码
#     plt.subplot(1, 3, 3)
#     obstacles_mask = all_game_info.get('obstacles_mask', np.zeros((256, 256)))
#     plt.imshow(obstacles_mask, cmap='gray')
#     plt.title(f'Frame {frame_idx} - Obstacles Mask')
#     plt.axis('off')
    
#     plt.tight_layout()
    
#     # 保存结果图像
#     save_dir = "cao"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     plt.savefig(os.path.join(save_dir, f"detection_frame_{frame_idx:03d}.png"))
#     plt.close()


# def save_ghosts_info(all_game_info, frame_idx):
#     """
#     保存ghosts_info到文本文件
    
#     :param all_game_info: detect_all_in_one函数返回的所有游戏信息
#     :param frame_idx: 帧索引
#     """
#     save_dir = "cao"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # 准备要写入的文本内容
#     ghosts_info_text = f"Frame {frame_idx}:\n"
#     ghosts_info_text += f"Ghost Num: {all_game_info.get('ghost_num', 0)}\n"
#     ghosts_info_text += f"Ghosts Boxes: {all_game_info.get('ghosts_boxes', [])}\n"
#     ghosts_info_text += f"Ghosts Centers: {all_game_info.get('ghosts_centers', [])}\n"
#     ghosts_info_text += "-" * 50 + "\n"
    
#     # 写入文件
#     txt_file_path = os.path.join(save_dir, "ghosts_info.txt")
#     with open(txt_file_path, "a" if frame_idx > 0 else "w") as f:
#         f.write(ghosts_info_text)


def test_detect_all_in_one():
    """
    测试detect_all_in_one函数
    """
    # 初始化环境
    env = create_pacman_environment()
    
    # 重置环境获取初始状态
    observation, info = env.reset()
    
    # 创建一个模拟的args对象
    class MockArgs:
        def __init__(self):
            self.size = 256
            self.visualize_save = True
            self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"
            self.your_mission_name = "MissionName"
            

    
    args = MockArgs()
    
    # 用于存储前一帧的游戏信息
    former_all_game_info = None
    
    print("开始测试detect_all_in_one函数...")
    print("=" * 40)
    epoch = 0
    # 测试50帧
    for frame_idx in range(50):
        # 随机选择一个动作
        action = env.action_space.sample()
        
        # 执行动作并获取结果
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 使用observation作为环境图像
        env_img = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        
        # 检查图像是否有效
        if env_img is None or env_img.size == 0:
            print(f"警告: 第 {frame_idx} 帧图像无效，跳过处理")
            continue
        
        # 调用detect_all_in_one函数
        all_game_info = detect_all_in_one(
            env_img, 
            args, 
            epoch, 
            frame_idx,
            former_all_game_info
        )
        
        # 可视化检测结果
        # visualize_detection_results(env_img, all_game_info, frame_idx)
        
        # # 保存ghosts_info到文本文件
        # save_ghosts_info(all_game_info, frame_idx)
        
        # 更新former_all_game_info
        former_all_game_info = all_game_info
        
        # 显示进度
        print(f"已处理帧 {frame_idx + 1}/50")
        
        # 检查游戏是否结束
        if terminated or truncated:
            print("游戏结束，重新开始...")
            observation, info = env.reset()
    
    # 关闭环境
    env.close()
    print(f"\n测试完成！结果已保存到 '{os.path.abspath('cao')}' 文件夹中")
    # epoch = epoch + 1
    


if __name__ == "__main__":
    test_detect_all_in_one()