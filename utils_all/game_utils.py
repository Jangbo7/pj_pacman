import gymnasium as gym
import numpy as np
import time
import ale_py
import cv2

def create_pacman_environment(render_mode="human"):
    """
    创建Pac-Man游戏环境
    
    Args:
        render_mode: 渲染模式 ("human", "rgb_array", None)
        
    Returns:
        env: Atari Pac-Man环境实
    """
    # 创建Ms. Pac-Man环境，这是Atari游戏中最接近经典Pac-Man的游戏

    # 'MsPacman-v4'
    # env = gym.make("MsPacmanNoFrameskip-v4", render_mode=render_mode)
    env = gym.make("ALE/MsPacman-v5", render_mode=render_mode)
    return env


def list_available_atari_games():
    """
    列出一些可用的Atari游戏环境
    """
    atari_games = [
        "ALE/MsPacman-v5",
        "ALE/Pacman-v5", 
        "ALE/Breakout-v5",
        "ALE/SpaceInvaders-v5",
        "ALE/Pong-v5"
    ]
    
    print("一些可用的Atari游戏环境:")
    for game in atari_games:
        print(f"  - {game}")


def preprocess_observation(observation, target_size=(256, 256)):
    """
    预处理游戏观测数据
    
    Args:
        observation: 原始观测数据
        target_size: 目标尺寸 (height, width)
        
    Returns:
        processed_obs: 处理后的观测数据
    """
    # 如果观测数据是图像，则调整大小
    if len(observation.shape) >= 2:
        import cv2
        # 调整图像大小
        processed_obs = cv2.resize(observation, target_size)
        return processed_obs
    else:
        # 如果不是图像数据，直接返回
        return observation


def get_game_info(env):
    """
    获取游戏环境信息
    
    Args:
        env: 游戏环境
        
    Returns:
        info_dict: 包含环境信息的字典
    """
    info_dict = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "action_meanings": getattr(env.unwrapped, "get_action_meanings", lambda: [])()
    }
    return info_dict