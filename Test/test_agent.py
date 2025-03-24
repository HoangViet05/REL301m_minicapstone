# import sys
# sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Train/Q-learning")
# sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Helper function")

# import pickle
# import numpy as np
# from cutting_stock_env_2DCSP_S_Q import CuttingStockEnv
# from Q_learning import QLearningAgent

# def load_agent(filename, env):
#     # Load trọng số đã lưu từ file .pkl
#     with open(filename, "rb") as f:
#         weights = pickle.load(f)
#     # Khởi tạo agent mới với cùng môi trường test
#     agent = QLearningAgent(env)
#     # Gán trọng số đã học cho agent
#     agent.weights = weights
#     # Đặt epsilon=0 để sử dụng chính sách greedy
#     agent.epsilon = 0.0
#     return agent

# def test_agent(agent, episodes=10, visualize=False):
#     for episode in range(episodes):
#         state = agent.env.reset()
#         done = False
#         step = 0
#         print(f"\n=== Episode {episode + 1} ===")
#         print(f"Khởi tạo môi trường với Products: {agent.env.products}, Pattern Size: {agent.env.pattern_size}")
#         while not done:
#             # Chọn hành động theo chính sách greedy (dựa trên Q-values đã học)
#             action = agent.get_action(state)
#             next_state, reward, done, info = agent.env.step(action)
            
#             # In thông tin chi tiết sau mỗi bước
#             print(f"\n--- Step {step + 1} ---")
#             print(f"Action: {action} (action_type={action[0]}, product_idx={action[1]})")
#             print(f"Reward: {reward}")
#             print(f"Done: {done}")
#             print(f"State trước: {state}")
#             print(f"State sau: {next_state}")
#             print(f"Current Demand: {agent.env.current_demand}")
#             print(f"Current Pattern: {agent.env.current_pattern}")
#             print(f"Number of Confirmed Patterns: {len(agent.env.patterns)}")
#             print(f"Remaining Area: {agent.env.remaining_area}")
            
#             state = next_state
#             step += 1
        
#         total_cost = info.get("cost", None)
#         print(f"\nEpisode {episode + 1} kết thúc sau {step} bước. Total cost: {total_cost}")
#         print(f"Danh sách pattern đã xác nhận: {agent.env.patterns}")
        
#         # Nếu visualize=True, vẽ trực quan các pattern đã xác nhận
#         if visualize:
#             agent.env.visualize_patterns()

# if __name__ == '__main__':
#     # Tạo môi trường test với các thông số tương tự như đã train (có thể điều chỉnh theo thực tế)
#     test_env = CuttingStockEnv(
#         product_ranges=[
#             (40, 40, 30, 30, 3, 5),
#             (50, 50, 40, 40, 5, 10)
#         ],
#         pattern_size_range=(150, 200, 150, 200),
#         max_steps=200,
#         max_patterns=10
#     )
    
#     # Load agent từ file model.pkl (đảm bảo file này đã được lưu sau quá trình train)
#     agent = load_agent(
#         "E:\\Learn_space\\FPT\\REL301m\\REL301m_mini_capstone_GROUP4\\model\\qlearning_model.pkl", 
#         test_env
#     )
    
#     # Chạy agent trên môi trường test trong một số episode và visualize kết quả
#     test_agent(agent, episodes=30, visualize=True)


import sys
# Thêm đường dẫn nếu cần
sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Train/Q-learning")
sys.path.append("E:/Learn_space/FPT/REL301m/REL301m_mini_capstone_GROUP4/Helper function")
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from cutting_stock_env_2DCSP_S_Q import CuttingStockEnv
from Q_learning import QLearningAgent

def load_agent(filename, env):
    # Load trọng số đã lưu từ file .pkl
    with open(filename, "rb") as f:
        weights = pickle.load(f)
    # Khởi tạo agent mới với cùng môi trường test
    agent = QLearningAgent(env)
    # Gán trọng số đã học cho agent
    agent.weights = weights
    # Đặt epsilon=0 để sử dụng chính sách greedy
    agent.epsilon = 0.0
    return agent

def visualize_and_save_patterns(env, save_dir, episode_num):
    """
    Gọi hàm visualize_patterns() của môi trường, nhưng thay vì hiển thị,
    lưu hình ảnh vào file trong thư mục save_dir với tên file chứa số episode.
    """

    plt.ioff()  # Tắt chế độ tương tác
    env.visualize_patterns()  # Hàm này vẽ các patch lên figure

    fig = plt.gcf()  # Lấy figure hiện hành
    fig.canvas.draw()  # Buộc matplotlib vẽ xong

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"episode_{episode_num+1}_{timestamp}.png")
    fig.savefig(filename)

    plt.close(fig)  # Đóng figure
    plt.ion()  # Bật lại chế độ tương tác nếu cần


def test_agent(agent, episodes=10, visualize=False):
    # Tạo thư mục lưu kết quả với tên theo thời gian hiện tại
    base_save_path = r"E:\Learn_space\FPT\REL301m\REL301m_mini_capstone_GROUP4\Test\result"
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_save_path, timestamp_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    for episode in range(episodes):
        state = agent.env.reset()
        done = False
        step = 0
        print(f"\n=== Episode {episode + 1} ===")
        print(f"Khởi tạo môi trường với Products: {agent.env.products}, Pattern Size: {agent.env.pattern_size}")
        while not done:
            # Chọn hành động theo chính sách greedy (dựa trên Q-values đã học)
            action = agent.get_action(state)
            next_state, reward, done, info = agent.env.step(action)
            
            # In thông tin chi tiết sau mỗi bước
            print(f"\n--- Step {step + 1} ---")
            print(f"Action: {action} (action_type={action[0]}, product_idx={action[1]})")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"State trước: {state}")
            print(f"State sau: {next_state}")
            print(f"Current Demand: {agent.env.current_demand}")
            print(f"Current Pattern: {agent.env.current_pattern}")
            print(f"Number of Confirmed Patterns: {len(agent.env.patterns)}")
            print(f"Remaining Area: {agent.env.remaining_area}")
            
            state = next_state
            step += 1
        
        total_cost = info.get("cost", None)
        print(f"\nEpisode {episode + 1} kết thúc sau {step} bước. Total cost: {total_cost}")
        print(f"Danh sách pattern đã xác nhận: {agent.env.patterns}")
        
        if visualize:
            visualize_and_save_patterns(agent.env, save_dir, episode)

if __name__ == '__main__':
    # Tạo môi trường test với các thông số tương tự như đã train (có thể điều chỉnh theo thực tế)
    test_env = CuttingStockEnv(
        product_ranges=[
            (40, 40, 30, 30, 3, 5),
            (50, 50, 40, 40, 5, 10)
        ],
        pattern_size_range=(150, 200, 150, 200),
        max_steps=200,
        max_patterns=10
    )
    
    # Load agent từ file model.pkl (đảm bảo file này đã được lưu sau quá trình train)
    agent = load_agent(
        "E:\\Learn_space\\FPT\\REL301m\\REL301m_mini_capstone_GROUP4\\model\\qlearning_model.pkl", 
        test_env
    )
    
    # Chạy agent trên môi trường test trong một số episode và visualize kết quả
    test_agent(agent, episodes=30, visualize=True)
