# import gymnasium as gym
# import ale_py
# import cv2
# import os

# gym.register_envs(ale_py)

# def save_all_frames(env_name='MsPacmanNoFrameskip-v4', num_frames=300, action=3):
#     """
#     保存游戏的每一帧到文件夹
    
#     参数:
#         env_name: 环境名称
#         num_frames: 要保存的帧数
#         action: 执行的动作 (3=左, 2=右, 1=上, 4=下, 0=无操作)
#     """
#     # 创建输出文件夹
#     output_dir = 'MsPacman_img'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     env = gym.make(env_name, render_mode='rgb_array')
#     obs, info = env.reset(seed=42)
    
#     # 保存第0帧
#     frame_path = os.path.join(output_dir, f'frame_{0:04d}.png')
#     cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
#     print(f"已保存: {frame_path}")
    
#     # 保存后续帧
#     for frame in range(1, num_frames):
#         obs, reward, terminated, truncated, info = env.step(action)
        
#         if terminated or truncated:
#             print(f"\n游戏在第 {frame} 帧结束")
#             break
        
#         # 保存当前帧
#         frame_path = os.path.join(output_dir, f'frame_{frame:04d}.png')
#         cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        
#         if frame % 10 == 0:
#             print(f"已保存: {frame_path}")
    
#     env.close()
    
#     print(f"\n总共保存了 {frame + 1} 帧图像到文件夹: {output_dir}")
#     print(f"图像尺寸: {obs.shape[1]}x{obs.shape[0]}")


# def save_frames_with_multiple_actions(env_name='MsPacmanNoFrameskip-v4', frames_per_action=50):
#     """
#     尝试不同动作并保存帧
#     """
#     output_dir = 'All_test_img'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     env = gym.make(env_name, render_mode='rgb_array')
#     obs, info = env.reset(seed=42)
    
#     # 保存初始帧
#     cv2.imwrite(os.path.join(output_dir, f'frame_{0:04d}.png'), 
#                 cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    
#     actions = [0, 3, 2, 1, 4]  # 无操作, 左, 右, 上, 下
#     action_names = ['noop', 'left', 'right', 'up', 'down']
    
#     frame_count = 1
    
#     for action, action_name in zip(actions, action_names):
#         print(f"\n执行动作: {action_name} (代码: {action})")
        
#         for i in range(frames_per_action):
#             obs, reward, terminated, truncated, info = env.step(action)
            
#             if terminated or truncated:
#                 print(f"游戏在第 {frame_count} 帧结束")
#                 env.close()
#                 return
            
#             frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
#             cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            
#             if frame_count % 25 == 0:
#                 print(f"已保存: {frame_path}")
            
#             frame_count += 1
    
#     env.close()
#     print(f"\n总共保存了 {frame_count} 帧")



# print("开始保存游戏帧...\n")
# # save_all_frames(num_frames=300, action=3)  # action=3 是向左
    
# #     # 方案2: 如果想尝试不同动作，注释掉上面，取消注释下面这行
# save_frames_with_multiple_actions(frames_per_action=50)


import gymnasium as gym
import ale_py
import cv2
import os

gym.register_envs(ale_py)

def save_frames_no_action(env_name='MsPacmanNoFrameskip-v4', num_frames=500):
    """
    保存游戏开始后不操作的每一帧
    
    参数:
        env_name: 环境名称
        num_frames: 要保存的帧数
    """
    # 创建输出文件夹
    output_dir = 'Non_action_MsPacman_img'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    env = gym.make(env_name, render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    # 保存第0帧（初始帧）
    frame_path = os.path.join(output_dir, f'frame_{0:04d}.png')
    cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    print(f"已保存: {frame_path}")
    
    # 保存后续帧，动作为0（不操作）
    for frame in range(1, num_frames):
        obs, reward, terminated, truncated, info = env.step(0)  # 0 = 不操作
        
        if terminated or truncated:
            print(f"\n游戏在第 {frame} 帧结束")
            break
        
        # 保存当前帧
        frame_path = os.path.join(output_dir, f'frame_{frame:04d}.png')
        cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        
        if frame % 10 == 0:
            print(f"已保存第 {frame} 帧")
    
    env.close()
    
    print(f"\n完成！总共保存了 {frame + 1} 帧图像到文件夹: {output_dir}")
    print(f"图像尺寸: {obs.shape}")


print("开始记录游戏画面（玩家不操作）...\n")
save_frames_no_action(num_frames=500)