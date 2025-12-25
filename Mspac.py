import gymnasium as gym
import ale_py
import cv2
import os
import numpy as np

gym.register_envs(ale_py)

def play_manually(env_name='MsPacmanNoFrameskip-v4', save_frames=True):
    """
    手动玩 Ms. Pacman，使用键盘控制
    
    按键说明：
        W 或 ↑ : 向上
        S 或 ↓ : 向下
        A 或 ← : 向左
        D 或 → : 向右
        空格   : 不操作
        Q      : 退出游戏
        R      : 重新开始
    
    参数:
        save_frames: 是否保存每一帧到文件夹
    """
    # 创建输出文件夹
    if save_frames:
        output_dir = 'MsPacman_img'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    env = gym.make(env_name, render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    frame_count = 0
    total_reward = 0
    action = 0  # 默认不操作
    
    print("=" * 60)
    print("Ms. Pacman 手动游戏")
    print("=" * 60)
    print("按键说明：")
    print("  W/↑ : 向上")
    print("  S/↓ : 向下")
    print("  A/← : 向左")
    print("  D/→ : 向右")
    print("  空格: 不操作")
    print("  Q   : 退出游戏")
    print("  R   : 重新开始")
    print("=" * 60)
    print("\n游戏开始！按键操作...\n")
    
    # 创建窗口
    cv2.namedWindow('Ms. Pacman', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ms. Pacman', 800, 1000)
    
    running = True
    
    while running:
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 保存帧
        if save_frames:
            frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
            cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        
        # 显示游戏画面
        display_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        
        # 添加信息显示
        info_img = np.zeros((200, display_img.shape[1], 3), dtype=np.uint8)
        cv2.putText(info_img, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, f'Score: {int(total_reward)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        action_names = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN']
        cv2.putText(info_img, f'Action: {action_names[action]}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if save_frames:
            cv2.putText(info_img, 'Recording: ON', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 合并显示
        combined_img = np.vstack([display_img, info_img])
        cv2.imshow('Ms. Pacman', combined_img)
        
        frame_count += 1
        
        # 检查游戏是否结束
        if terminated or truncated:
            print(f"\n游戏结束！")
            print(f"总分: {int(total_reward)}")
            print(f"总帧数: {frame_count}")
            print("\n按 R 重新开始，按 Q 退出")
            
            # 等待用户选择
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('r') or key == ord('R'):
                    obs, info = env.reset(seed=None)  # 随机种子
                    frame_count = 0
                    total_reward = 0
                    action = 0
                    print("\n游戏重新开始！")
                    break
                elif key == ord('q') or key == ord('Q'):
                    running = False
                    break
            continue
        
        # 获取键盘输入（等待时间很短，让游戏流畅运行）
        key = cv2.waitKey(30) & 0xFF
        
        # 处理按键
        if key == ord('w') or key == 82:  # W 或 ↑
            action = 1  # 向上
        elif key == ord('s') or key == 84:  # S 或 ↓
            action = 4  # 向下
        elif key == ord('a') or key == 81:  # A 或 ←
            action = 3  # 向左
        elif key == ord('d') or key == 83:  # D 或 →
            action = 2  # 向右
        elif key == ord(' '):  # 空格
            action = 0  # 不操作
        elif key == ord('q') or key == ord('Q'):
            running = False
            print("\n退出游戏")
        elif key == ord('r') or key == ord('R'):
            obs, info = env.reset(seed=None)
            frame_count = 0
            total_reward = 0
            action = 0
            print("\n游戏重新开始！")
    
    cv2.destroyAllWindows()
    env.close()
    
    if save_frames:
        print(f"\n已保存 {frame_count} 帧图像到文件夹: {output_dir}")



play_manually(save_frames=True)
    
    # 如果只想玩游戏不保存帧，改成：
    # play_manually(save_frames=False)
    
##大力丸吃一次鬼加200分