import gymnasium as gym
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from detect_all import detect_all_in_one
from utils_all.game_utils import create_pacman_environment
from ultralytics import YOLO

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
    
    # 初始化YOLO模型
    print(f"正在初始化YOLO模型，路径: {args.path}")
    model = YOLO(args.path)
    
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
        
        # 调用detect_all_in_one函数，并传入预初始化的YOLO模型
        all_game_info = detect_all_in_one(
            env_img, 
            args, 
            epoch, 
            frame_idx,
            former_all_game_info,
            model=model
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
    print(f"\n测试完成！结果已保存到 detection_results/{args.your_mission_name}文件夹中。")
    # epoch = epoch + 1
    


if __name__ == "__main__":
    test_detect_all_in_one()