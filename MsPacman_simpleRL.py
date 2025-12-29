import gymnasium as gym
import ale_py
import numpy as np
from typing import List, Tuple, Dict, Any
from enum import Enum
import cv2
import os
import matplotlib.pyplot as plt
from detect_all import detect_all_in_one, detect_score
from utils_all.game_utils import create_pacman_environment
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier as Agent
import warnings
warnings.filterwarnings('ignore')

class PacmanState:
    def __init__(self, pacman_x, pacman_y, ghost_distance, superpill_distance, pill_number, decision, scared_ghost, score):
        self.pacman_x = pacman_x
        self.pacman_y = pacman_y
        self.ghost_distance = ghost_distance
        self.superpill_distance = superpill_distance
        self.pill_number = pill_number
        self.decision = decision
        self.scared_ghost = scared_ghost
        self.score = score
        self.next_action = 0

class PacmanEvaluator:
    def __init__(self):
        self.weight = np.load('weight.npy')
        self.agent = Agent(n_neighbors=1).fit(self.weight[:,:8], self.weight[:, 8:])
        
    def evaluate_state(self, state: PacmanState) -> float:
        value_list = self.agent.predict(np.array([[state.pacman_x, state.pacman_y, state.ghost_distance, state.superpill_distance, state.pill_number, state.decision, state.scared_ghost, state.score]]))
        v = 1
        if (int(value_list[0, state.next_action-1]) >= 0.25):
            T = 1
        elif ((state.decision >> (state.next_action-1)) & 0b1 == 0):
            T = 1e-2
        else:
            T = 1e-1
        
        return T*int(value_list[0, 4])

class PacmanRLReasoner:
    def __init__(self, evaluator: PacmanEvaluator = None):
        self.evaluator = evaluator or PacmanEvaluator()
        self.action_space = [1, 4, 3, 2]
    
    def choose_action(self, state: PacmanState):
        best_action = 0
        best_value = -1000
        
        for action in self.action_space:
            state.next_action = action
            
            state_value = self.evaluator.evaluate_state(state)
            
            if state_value > best_value:
                best_action = action
                best_value = state_value
        
        return [int(best_action), int(best_value)]

def create_state(obs, former_all_game_info):
    if len(former_all_game_info["pacman_centers"]) == 0:
        pacman_x = 0
        pacman_y = 0
    else:
        pacman_x, pacman_y = former_all_game_info["pacman_centers"][0]

    ghost_distance = 1000
    for i in range(1, 5):
        distance = abs(former_all_game_info["ghosts_centers"][i][0]-pacman_x) + abs(former_all_game_info["ghosts_centers"][i][1]-pacman_y)
        if (distance < ghost_distance):
            ghost_distance = distance

    superpill_distance = 1000
    for i in range(len(former_all_game_info["superpill_centers"])):
        distance = abs(former_all_game_info["superpill_centers"][i][0]-pacman_x) + abs(former_all_game_info["superpill_centers"][i][1]-pacman_y)
        if (distance < superpill_distance):
            superpill_distance = distance

    pill_number = former_all_game_info["pill_num"][0]

    decision = 0
    decision += former_all_game_info["pacman_decision"]["up"]
    decision <<= 2
    decision += former_all_game_info["pacman_decision"]["right"]
    decision <<= 2
    decision += former_all_game_info["pacman_decision"]["left"]
    decision <<= 2
    decision += former_all_game_info["pacman_decision"]["down"]
        
    image_rgb = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    mask = (image_rgb[:, :, 0] == 66) & (image_rgb[:, :, 1] == 114) & (image_rgb[:, :, 2] == 194)
    scared_ghost = int(np.sum(mask))

    score = former_all_game_info["score"] % 3000

    state = PacmanState(pacman_x, pacman_y, ghost_distance, superpill_distance, pill_number, decision, scared_ghost, score)

    return state

def single_action(env, action_num, duration):
    for i in range(duration):
        obs, reward, terminated, truncated, info = env.step(action_num)
    
    cv2.imwrite(f'MsPacman/cut.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    return obs, terminated, truncated

if __name__ == "__main__":
    class MockArgs:
        def __init__(self):
            self.size = 256
            self.visualize_save = True
            self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"
            self.your_mission_name = "test"

    args = MockArgs()
    model = YOLO(args.path)
    
    env = gym.make('MsPacmanNoFrameskip-v4', render_mode='human')
    obs, info = env.reset(seed=42)
    
    former_all_game_info = None
    
    evaluator = PacmanEvaluator()
    reasoner = PacmanRLReasoner(evaluator)
    terminated = False
    truncated = False

    while (not (terminated or truncated)):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        former_all_game_info = detect_all_in_one(obs, args, 0, 0, former_all_game_info, model=model)
        state = create_state(obs, former_all_game_info)
        action, value = reasoner.choose_action(state)

        obs, terminated, truncated = single_action(env, action, value)

    env.close()