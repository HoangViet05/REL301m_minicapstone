import sys
# Thêm đường dẫn nếu cần
sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Helper function")
sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Train/Deep Q-learning")
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from cutting_stock_env_2DCSP_S_DeepQ import CuttingStockEnv
from Deep_Q_learning import QNetwork


def load_dqn_agent(model_path, env):
    state_dim = env.observation_space.shape[0]
    action_dim = int(np.prod(env.action_space.nvec))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device


def get_greedy_action(model, device, state, action_shape):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    action_idx = q_values.argmax().item()
    return np.unravel_index(action_idx, action_shape)


def visualize_and_save_patterns(env, save_dir, episode_num):
    plt.ioff()
    env.visualize_patterns()
    fig = plt.gcf()
    fig.canvas.draw()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"episode_{episode_num+1}_{timestamp}.png")
    fig.savefig(filename)

    plt.close(fig)
    plt.ion()


def test_dqn(model, device, env, episodes=10, visualize=False):
    base_save_path = r"E:\\Learn_space\\FPT\\REL301m\\REL301m_mini_capstone_GROUP4\\Test\\result_DeepQ_Learning"
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_save_path, timestamp_folder)
    os.makedirs(save_dir, exist_ok=True)

    for episode in range(episodes):
        state = env.reset()
        done = False
        step = 0

        print(f"\n=== Episode {episode + 1} ===")
        print(f"Products: {env.products}, Pattern Size: {env.pattern_size}")

        while not done:
            action = get_greedy_action(model, device, state, env.action_space.nvec)
            next_state, reward, done, info = env.step(action)

            print(f"\n--- Step {step + 1} ---")
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            print(f"Demand: {env.current_demand}, Pattern: {env.current_pattern}, Remaining Area: {env.remaining_area}")

            state = next_state
            step += 1

        total_cost = info.get("cost", None)
        print(f"\nEpisode {episode + 1} kết thúc sau {step} bước. Total cost: {total_cost}")
        print(f"Patterns đã xác nhận: {env.patterns}")

        if visualize:
            visualize_and_save_patterns(env, save_dir, episode)


if __name__ == '__main__':
    env = CuttingStockEnv(
        product_ranges=[
            (40, 40, 30, 30, 1, 10),
            (50, 50, 40, 40, 1, 20)
        ],
        pattern_size_range=(150, 200, 150, 200),
        max_steps=200,
        max_patterns=10
    )

    model_path = "E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Model/dqn_model.pt"
    model, device = load_dqn_agent(model_path, env)

    test_dqn(model, device, env, episodes=30, visualize=True)
