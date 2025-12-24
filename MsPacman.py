import gymnasium as gym
import ale_py
import time
import cv2

gym.register_envs(ale_py)

part1 = []

def play_game(env_name, render=True):
    try:
        env = gym.make(env_name, render_mode='human' if render else None)
        obs, info = env.reset(seed=42)
        obs = single_action(env, 0, 80)
        obs = single_action(env, 3, 240)
        obs = single_action(env, 4, 70)
        obs = single_action(env, 3, 40)
        obs = single_action(env, 1, 70)

        obs = single_action(env, 2, 40)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 1, 20)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 2, 20)
        obs = single_action(env, 1, 10)
        obs = single_action(env, 2, 80)
        obs = single_action(env, 4, 20)
        obs = single_action(env, 1, 80)
        obs = single_action(env, 3, 60)
        obs = single_action(env, 4, 45)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 2, 20)
        obs = single_action(env, 4, 20)
        obs = single_action(env, 2, 60)
        obs = single_action(env, 1, 20)
        obs = single_action(env, 3, 30)
        obs = single_action(env, 1, 60)
        obs = single_action(env, 3, 50)
        obs = single_action(env, 4, 40)

        obs = single_action(env, 2, 60)
        obs = single_action(env, 1, 40)
        obs = single_action(env, 2, 80)
        obs = single_action(env, 4, 30)
        obs = single_action(env, 3, 90)
        obs = single_action(env, 4, 35)
        obs = single_action(env, 2, 70)
        obs = single_action(env, 1, 20)
        obs = single_action(env, 4, 20)
        obs = single_action(env, 2, 100)
        obs = single_action(env, 1, 30)
        obs = single_action(env, 2, 20)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 1, 20)
        obs = single_action(env, 2, 20)
        obs = single_action(env, 1, 40)
        
        obs = single_action(env, 3, 40)
        obs = single_action(env, 4, 40)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 2, 40)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 4, 30)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 4, 70)
        obs = single_action(env, 3, 30)
        obs = single_action(env, 2, 15)
        obs = single_action(env, 4, 25)
        obs = single_action(env, 3, 40)
        obs = single_action(env, 1, 20)
        obs = single_action(env, 4, 25)
        obs = single_action(env, 3, 25)
        obs = single_action(env, 1, 30)
        obs = single_action(env, 4, 30)
        obs = single_action(env, 2, 25)
        obs = single_action(env, 4, 25)
        obs = single_action(env, 2, 45)
        obs = single_action(env, 1, 25)
        obs = single_action(env, 2, 25)
        obs = single_action(env, 1, 40)
        obs = single_action(env, 2, 25)
        obs = single_action(env, 4, 25)
        obs = single_action(env, 2, 27)
        obs = single_action(env, 4, 40)

        obs = single_action(env, 3, 36)
        obs = single_action(env, 1, 60)
        obs = single_action(env, 4, 40)
        obs = single_action(env, 3, 120)
        obs = single_action(env, 1, 90)
        obs = single_action(env, 2, 140)
        obs = single_action(env, 4, 25)
        obs = single_action(env, 2, 20)
        obs = single_action(env, 1, 20)
        obs = single_action(env, 2, 30)
        obs = single_action(env, 3, 20)
        obs = single_action(env, 1, 20)

        env.close()
        
    except Exception as e:
        print(f"错误: {e}")

def single_action(env, action_num, duration):
    for i in range(duration):
        obs, reward, terminated, truncated, info = env.step(action_num)
        # if (i % 40 == 1):
        #     cv2.imwrite('MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    cv2.imwrite('MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    return obs

def paint(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

play_game('MsPacmanNoFrameskip-v4', True)


