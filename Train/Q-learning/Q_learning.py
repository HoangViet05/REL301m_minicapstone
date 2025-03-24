# ---------------------------
# Lớp QLearningAgent với các biểu đồ bổ sung và lưu model
# ---------------------------
import sys
sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Helper function")

import matplotlib.pyplot as plt
import pickle
import numpy as np
from Best_fit import can_pack_all_products_guillotine, GuillotineStockSheet
from cutting_stock_env_2DCSP_S_Q import CuttingStockEnv

class QLearningAgent:
    def __init__(self, env, alpha=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.state_dim = env.observation_space.shape[0]
        # Với action_space là MultiDiscrete([3, n_products]), nvec = [3, n_products]
        self.action_dim = np.prod(env.action_space.nvec)
        self.weights = np.random.randn(self.state_dim, self.action_dim) * 0.01
        
        # Theo dõi metrics
        self.rewards_history = []
        self.costs_history = []
        self.patterns_history = []
        self.epsilon_history = []
        self.success_history = []  # Nếu có thể tính được thành công
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploration
        else:
            q_values = state.dot(self.weights)
            return np.unravel_index(np.argmax(q_values), self.env.action_space.nvec)
    
    def update_weights(self, state, action, reward, next_state, done):
        action_idx = np.ravel_multi_index(action, self.env.action_space.nvec)
        current_q = state.dot(self.weights)[action_idx]
        next_q = 0 if done else np.max(next_state.dot(self.weights))
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        # self.weights[:, action_idx] += self.alpha * td_error * state
        self.weights[:, action_idx] += self.alpha * (td_error * state - 0.001 * self.weights[:, action_idx])
        
    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            episode_cost = 0.0
            success_flag = 0  # 1 nếu hoàn thành (demand == 0), 0 nếu không
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_weights(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
                if done:
                    episode_cost = info.get("cost", 0.0)
                    if np.all(self.env.current_demand == 0):
                        success_flag = 1
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            self.rewards_history.append(episode_reward)
            self.costs_history.append(episode_cost)
            self.patterns_history.append(len(self.env.patterns))
            self.epsilon_history.append(self.epsilon)
            self.success_history.append(success_flag)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, Epsilon={self.epsilon:.4f}")

            if episode % 500 == 0:
                avg_cost = np.mean(agent.costs_history[-100:])
                success_rate = np.mean(agent.success_history[-100:])
                print(f"Episode {episode}: Avg Cost (100 eps) = {avg_cost:.2f}, Success Rate = {success_rate:.2f}")
                    
    def plot_metrics(self):
        episodes = np.arange(len(self.rewards_history))
        
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

        # plt.subplot(231)
        # plt.plot(episodes[19950:20000], self.rewards_history[19950:20000])
        # plt.xlabel("Episode")
        # plt.ylabel("Episode Reward")
        # plt.title("Episode Reward")

        # plt.subplot(232)
        # plt.plot(episodes[19950:20000], self.costs_history[19950:20000])
        # plt.xlabel("Episode")
        # plt.ylabel("Total Cost")
        # plt.title("Total Cost per Episode")

        # plt.subplot(233)
        # plt.plot(episodes[19950:20000], self.patterns_history[19950:20000])
        # plt.xlabel("Episode")
        # plt.ylabel("Number of Patterns")
        # plt.title("Number of Patterns")
        
        plt.subplot(234)
        plt.plot(episodes, self.epsilon_history)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay")
        
        plt.subplot(235)
        plt.plot(episodes, self.success_history)
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title("Success (1 if demand==0)")
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filename):
        # Lưu trọng số của agent
        with open(filename, "wb") as f:
            pickle.dump(self.weights, f)
        print(f"Model saved to {filename}")

# ---------------------------
# Khởi tạo môi trường và agent
# ---------------------------
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

    agent = QLearningAgent(env, alpha=0.0001, gamma=0.99, epsilon_decay=0.9999)
    agent.train(episodes=30000)
    agent.plot_metrics()
    agent.save_model("qlearning_model.pkl")

