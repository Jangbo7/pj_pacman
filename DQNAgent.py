import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNAgent:
    """
    Deep Q-Network Agent for Reinforcement Learning.
    Attributes:
        state_size (int): Dimension of the state space.
        action_size (int): Dimension of the action space.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy policy.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.gamma = 0.99
        self.lr = 1e-3

        self.reward = 0

        self.memory = ReplayBuffer(capacity=10000)

        self.q_net = QNetwork(state_size, action_size)
        self.target_q_net = QNetwork(state_size, action_size)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # 每隔多少次梯度更新同步 target 网络，这是 DQN 稳定训练的关键之一
        self.target_update_every = 10
        self.step_count = 0

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)


    def choose_action(self, state, legal_actions_mask: np.ndarray):
        legal_actions = [i for i in range(self.action_size) if legal_actions_mask[i] > 0.5]
        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)
        
         # ---- 关键：np -> torch，并加 batch 维 ----
        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state).float()
        elif torch.is_tensor(state):
            state_t = state.float()
        else:
            state_t = torch.tensor(state, dtype=torch.float32)

        state_t = state_t.unsqueeze(0).to(self.device)  # (1, D)

        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0)    # (A,)

        # 只在合法动作里取最大
        legal_q = q_values[legal_actions]
        best_idx = torch.argmax(legal_q).item()
        return legal_actions[best_idx]

    def store(self, s, a, r, s_next, done):
        # ---- 关键：存进 replay 的统一是 CPU Tensor，update 时再搬到 GPU ----
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        elif not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.float32)
        else:
            s = s.float()

        if isinstance(s_next, np.ndarray):
            s_next = torch.from_numpy(s_next).float()
        elif not torch.is_tensor(s_next):
            s_next = torch.tensor(s_next, dtype=torch.float32)
        else:
            s_next = s_next.float()

        self.memory.push(s.cpu(), int(a), float(r), s_next.cpu(), float(done))


    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        # 采样 + Bellman 更新
        batch = self.memory.sample(batch_size)

        # batch 里 s / s_next 已经是 CPU Tensor，能 stack
        S = torch.stack([s for s, _, _, _, _ in batch]).to(self.device)                                        # (B, D)
        A = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.int64).to(self.device)                    # (B,)
        R = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32).to(self.device)                  # (B,)
        S_next = torch.stack([s_next for _, _, _, s_next, _ in batch]).to(self.device)                         # (B, D)
        Done = torch.tensor([done for _, _, _, _, done in batch], dtype=torch.float32).to(self.device)         # (B,)    

        q_all = self.q_net(S)                                              # (B, A)
        q_values = q_all[torch.arange(batch_size, device=self.device), A]  # (B,)

        with torch.no_grad():
            q_next = self.target_q_net(S_next).max(dim=1)[0]               # (B,)
            q_target = R + self.gamma * q_next * (1 - Done)                # (B,)

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target Network 更新
        self.update_target()

        # Epsilon 更新
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())
    
    def update_target(self):
        self.step_count += 1
        if self.step_count % self.target_update_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_q_net.load_state_dict(torch.load(path))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """
    Deep Q-Network 
    输入：状态向量 (D,)
    网络结构：
    - 全连接层 128 个神经元，ReLU 激活
    - 全连接层 128 个神经元，ReLU 激活
    - 输出层 5 个神经元（对应 5 个动作）
    输出：每个动作的 Q 值 (A,)
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)
