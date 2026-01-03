# LinearQAgent.py
import numpy as np
import random

class LinearQAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=0.005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9997
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 每个动作一组线性权重 (A, D)
        self.W = np.zeros((action_size, state_size), dtype=np.float32)

    def choose_action(self, state, legal_actions_mask):
        legal_actions = [i for i in range(self.action_size) if legal_actions_mask[i]]

        # 安全兜底：如果检测出错导致没有合法动作，就 NOOP
        if len(legal_actions) == 0:
            return 0

        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)

        q_values = self.W @ state  # (A,)
        legal_q_values = q_values[legal_actions]

        best_idx = int(np.argmax(legal_q_values))     # 在 legal_q_values 里的索引
        # with open('q_values.txt', 'a') as f:
        #     f.write(f"legal_actions_mask: {legal_actions_mask}\n")
        #     f.write(f"legal_actions: {legal_actions}, legal_q_values: {legal_q_values}, best_action: {legal_actions[best_idx]}\n")
        return int(legal_actions[best_idx])           # 映射回真实动作编号


    def update(self, state, action, reward, next_state, done, next_legal_actions_mask):
        # TD target
        if done:
            target = reward
        else:
            q_next = self.W @ next_state
            q_next = q_next[next_legal_actions_mask.astype(bool)]
            target = reward + self.gamma * np.max(q_next)

        # TD error
        q_sa = np.dot(self.W[action], state)
        td_error = target - q_sa

        # 梯度更新
        self.W[action] += self.lr * td_error * state

        # epsilon 衰减（每一步一次）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        np.save(path, self.W)

    def load_model(self, path):
        self.W = np.load(path)