import gymnasium as gym
import ale_py
import time
import cv2

gym.register_envs(ale_py)

def part1(env, l=3, r=2, u=1, d=4):
    obs = single_action(env, l, 320)
    obs = single_action(env, d, 70)
    obs = single_action(env, l, 40)
    obs = single_action(env, u, 70)

    obs = single_action(env, r, 40)
    obs = single_action(env, l, 20)
    obs = single_action(env, u, 20)
    obs = single_action(env, l, 20)
    obs = single_action(env, r, 20)
    obs = single_action(env, u, 10)
    obs = single_action(env, r, 80)
    obs = single_action(env, d, 20)
    obs = single_action(env, u, 80)
    obs = single_action(env, l, 60)
    obs = single_action(env, d, 45)
    obs = single_action(env, l, 20)
    obs = single_action(env, r, 20)
    obs = single_action(env, d, 20)
    obs = single_action(env, r, 60)
    obs = single_action(env, u, 20)
    obs = single_action(env, l, 30)
    obs = single_action(env, u, 60)
    obs = single_action(env, l, 50)
    obs = single_action(env, d, 40)

    obs = single_action(env, r, 60)
    obs = single_action(env, u, 40)
    obs = single_action(env, r, 80)
    obs = single_action(env, d, 30)
    obs = single_action(env, l, 90)
    obs = single_action(env, d, 35)
    obs = single_action(env, r, 70)
    obs = single_action(env, u, 20)
    obs = single_action(env, d, 20)
    obs = single_action(env, r, 100)
    obs = single_action(env, u, 30)
    obs = single_action(env, r, 20)
    obs = single_action(env, l, 20)
    obs = single_action(env, u, 20)
    obs = single_action(env, r, 20)
    obs = single_action(env, u, 40)
        
    obs = single_action(env, l, 40)
    obs = single_action(env, d, 40)
    obs = single_action(env, l, 20)
    obs = single_action(env, r, 40)
    obs = single_action(env, l, 20)
    obs = single_action(env, d, 30)
    obs = single_action(env, l, 20)
    obs = single_action(env, d, 70)
    obs = single_action(env, l, 30)
    obs = single_action(env, r, 15)
    obs = single_action(env, d, 25)
    obs = single_action(env, l, 40)
    obs = single_action(env, u, 20)
    obs = single_action(env, d, 25)
    obs = single_action(env, l, 25)
    obs = single_action(env, u, 30)
    obs = single_action(env, d, 30)
    obs = single_action(env, r, 25)
    obs = single_action(env, d, 25)
    obs = single_action(env, r, 45)
    obs = single_action(env, u, 25)
    obs = single_action(env, r, 25)
    obs = single_action(env, u, 40)
    obs = single_action(env, r, 25)
    obs = single_action(env, d, 25)
    obs = single_action(env, r, 27)
    obs = single_action(env, d, 40)

    obs = single_action(env, l, 36)
    obs = single_action(env, u, 60)
    obs = single_action(env, d, 40)
    obs = single_action(env, l, 120)
    obs = single_action(env, u, 90)
    obs = single_action(env, r, 140)
    obs = single_action(env, d, 25)
    obs = single_action(env, r, 20)
    obs = single_action(env, u, 20)
    obs = single_action(env, r, 30)
    obs = single_action(env, l, 20)
    obs = single_action(env, u, 20)

def play_game(env_name, render=True):
    try:
        env = gym.make(env_name, render_mode='human' if render else None)
        obs, info = env.reset(seed=42)
        
        part1(env)
        part1(env)

        #part1(env)

        env.close()
        
    except Exception as e:
        print(f"错误: {e}")

def single_action(env, action_num, duration):
    for i in range(duration):
        obs, reward, terminated, truncated, info = env.step(action_num)
        # if (i % 40 == 1):
        #     cv2.imwrite('MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # cv2.imwrite('MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    return obs

def paint(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

play_game('MsPacmanNoFrameskip-v4', True)


