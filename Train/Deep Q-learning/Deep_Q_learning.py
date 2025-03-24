import sys
sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Helper function")

from Best_fit import can_pack_all_products_guillotine, GuillotineStockSheet
from cutting_stock_env_2DCSP_S_DeepQ import CuttingStockEnv

# ======================================
# Deep Q-Learning Agent (tự xây dựng với PyTorch)
# ======================================
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# (SỬA DQN) Định nghĩa mạng Q (QNetwork) sử dụng MLP đơn giản
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# (SỬA DQN) Replay Buffer đơn giản
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# (SỬA DQN) Định nghĩa Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01,
                batch_size=64, target_update_freq=1000, replay_buffer_capacity=10000):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = int(np.prod(env.action_space.nvec))
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

        # Theo dõi metrics
        self.rewards_history = []
        self.costs_history = []
        self.patterns_history = []
        self.epsilon_history = []
        self.success_history = []

        self.total_steps = 0

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # Lấy hành động ngẫu nhiên, note: env.action_space.sample() trả về tuple
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            # Giải mã action từ index
            return np.unravel_index(action_idx, self.env.action_space.nvec)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)  # shape: (batch_size, state_dim)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)  # shape: (batch_size,)
        dones = torch.FloatTensor(dones).to(self.device)  # shape: (batch_size,)

        # Convert actions từ tuple thành index
        actions_idx = []
        for action in actions:
            actions_idx.append(np.ravel_multi_index(action, self.env.action_space.nvec))
        actions_idx = torch.LongTensor(actions_idx).to(self.device)  # shape: (batch_size,)

        q_values = self.q_network(states)  # shape: (batch_size, action_dim)
        q_value = q_values.gather(1, actions_idx.unsqueeze(1)).squeeze(1)  # shape: (batch_size,)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target = rewards + self.gamma * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000, max_steps=200):
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            success_flag = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                self.total_steps += 1

                self.update()

                # Cập nhật target network định kỳ
                if self.total_steps % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                if done:
                    if np.all(self.env.current_demand == 0):
                        success_flag = 1
                    break

            # Giảm epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Lấy cost từ info
            episode_cost = info.get("cost", 0.0)

            self.rewards_history.append(episode_reward)
            self.costs_history.append(episode_cost)
            self.patterns_history.append(len(self.env.patterns))
            self.epsilon_history.append(self.epsilon)
            self.success_history.append(success_flag)

            if episode % 100 == 0:
                print(f"Episode {episode}: Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, Epsilon={self.epsilon:.4f}")

    def plot_metrics(self):
        episodes = np.arange(len(self.rewards_history))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 10))

        plt.subplot(231)
        plt.plot(episodes, self.rewards_history)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("Episode Reward")

        plt.subplot(232)
        plt.plot(episodes, self.costs_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Cost")
        plt.title("Total Cost per Episode")

        plt.subplot(233)
        plt.plot(episodes, self.patterns_history)
        plt.xlabel("Episode")
        plt.ylabel("Number of Patterns")
        plt.title("Number of Patterns")

        plt.subplot(234)
        plt.plot(episodes[:100], self.epsilon_history[:100])
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay (First 100 Episodes)")

        plt.subplot(235)
        plt.plot(episodes, self.success_history)
        plt.xlabel("Episode")
        plt.ylabel("Success (1 if demand==0)")
        plt.title("Success Rate")

        plt.tight_layout()
        plt.show()

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        print(f"Model saved to {filename}")

# ======================================
# Khởi tạo môi trường và DQN Agent
# ======================================
if __name__ == '__main__':
    env = CuttingStockEnv(
            product_ranges=[
                (40, 80, 30, 60, 1, 10),  # Sản phẩm 1: width 40-80, height 30-60, demand 1-10
                (50, 100, 40, 70, 1, 5)   # Sản phẩm 2: width 50-100, height 40-70, demand 1-5
            ],
            pattern_size_range=(150, 250, 100, 200),  # Pattern size động
            max_steps=200,
            max_patterns=10
        )

    dqn_agent = DQNAgent(env, lr=1e-4, gamma=0.95, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01,
                        batch_size=128, target_update_freq=500, replay_buffer_capacity=10000)

    dqn_agent.train(episodes=20000, max_steps=200)
    dqn_agent.plot_metrics()
    dqn_agent.save_model("dqn_model.pt")
