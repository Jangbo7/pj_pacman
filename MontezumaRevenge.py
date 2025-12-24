import gymnasium as gym
import ale_py
import cv2
import random

gym.register_envs(ale_py)

part1 = []

def play_game(env_name, render=True):
    try:
        env = gym.make(env_name, render_mode='human' if render else None)
        obs, info = env.reset(seed=42)
        single_action(env, 5, 8)
        single_action(env, 3, 10)
        single_action(env, 11, 5)
        single_action(env, 5, 3)
        single_action(env, 11, 6)
        single_action(env, 5, 10)

        single_action(env, 4, 12)
        single_action(env, 12, 4)
        single_action(env, 4, 12)

        single_action(env, 2, 10)
        single_action(env, 4, 4)
        single_action(env, 1, 4)
        single_action(env, 3, 6)
        single_action(env, 5, 10)

        single_action(env, 0, 3)
        single_action(env, 3, 10)
        single_action(env, 11, 4)
        single_action(env, 3, 14)

        single_action(env, 2, 10)
        single_action(env, 4, 2)
        single_action(env, 12, 4)
        single_action(env, 5, 2)
        single_action(env, 12, 4)
        single_action(env, 2, 15)

        direction = random.choice([0, 1])
        single_action(env, direction+3, 2)
        single_action(env, direction+11, 4)
        single_action(env, direction+3, 16)

        env.close()
        
    except Exception as e:
        print(f"错误: {e}")

def single_action(env, action_num, duration):
    for _ in range(duration):
        obs, reward, terminated, truncated, info = env.step(action_num)
        # cv2.imwrite('mtzm/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # cv2.imwrite('mtzm/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    return obs

def paint(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

play_game('ALE/MontezumaRevenge-v5', True)