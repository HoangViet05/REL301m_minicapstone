import gym
from gym import spaces
import numpy as np
from scipy.optimize import linprog
from Best_fit import can_pack_all_products_guillotine, GuillotineStockSheet

# ===============================
# Environment: CuttingStockEnv
# ===============================
class CuttingStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, products, stock_size, C_su=20, C_ss=1, max_steps=100, max_patterns=10):
        super(CuttingStockEnv, self).__init__()

        # Thông số bài toán
        self.products = products  # [(w1, h1, d1), (w2, h2, d2), ...]
        self.stock_size = stock_size  # (width, height)
        self.C_su = C_su  # Setup cost per pattern
        self.C_ss = C_ss  # Stock sheet cost

        # Không gian hành động: [action_type, product_idx]
        self.action_space = spaces.MultiDiscrete([3, len(products)])

        # Không gian trạng thái: Demand + Pattern hiện tại + Diện tích trống
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(len(products) + len(products) + 1,),  # [demand, current_pattern, remaining_area]
            dtype=np.float32
        )

        # Các tham số môi trường
        self.max_steps = max_steps
        self.max_patterns = max_patterns

        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_demand = np.array([p[2] for p in self.products], dtype=np.int32)
        self.current_pattern = np.zeros(len(self.products), dtype=np.int32)
        self.patterns = []
        self.remaining_area = self.stock_size[0] * self.stock_size[1]
        return self._get_state()

    def step(self, action):
        """
        action = (action_type, product_idx)
        action_type: 0 = Thêm sản phẩm, 1 = Xóa sản phẩm, 2 = Xác nhận pattern
        """
        self.current_step += 1
        action_type, product_idx = action

        reward = 0.0
        done = False

        # (SỬA) Tạo biến info để trả cost ra agent
        info = {"cost": 0.0}

        # -------------------------
        # Thực hiện hành động
        # -------------------------
        if action_type == 0:  # Thêm sản phẩm
            if self.current_demand[product_idx] > 0:
                temp_pattern = self.current_pattern.copy()
                temp_pattern[product_idx] += 1
                packable, _ = self._check_packing(temp_pattern)
                if packable:
                    self.current_pattern[product_idx] += 1
                    self.current_demand[product_idx] -= 1
                    # (SỬA DQN) Thưởng nhỏ khi thêm sp thành công
                    reward += 5
                    utilization = (self.stock_size[0]*self.stock_size[1] - self.remaining_area) / (self.stock_size[0]*self.stock_size[1])
                    reward += 50 * utilization
                else:
                    reward -= 2
        elif action_type == 1:  # Xóa sản phẩm
            if self.current_pattern[product_idx] > 0:
                self.current_pattern[product_idx] -= 1
                self.current_demand[product_idx] += 1
                reward -= 1
        elif action_type == 2:  # Xác nhận pattern
            if len(self.patterns) >= self.max_patterns:
                reward -= 50
            if np.sum(self.current_pattern) > 0:
                self.patterns.append(self.current_pattern.copy())
                self.current_pattern = np.zeros(len(self.products), dtype=np.int32)
                self.remaining_area = self.stock_size[0]*self.stock_size[1]
                reward += 20
            else:
                reward -= 10

        # -------------------------
        # Kiểm tra điều kiện kết thúc
        # -------------------------
        if np.all(self.current_demand == 0):
            total_cost = self._calculate_total_cost()
            reward += 1000
            reward -= np.log(1 + total_cost)
            info["cost"] = total_cost
            done = True
        else:
            if self.current_step >= self.max_steps:
                reward -= 50
                total_cost = self._calculate_total_cost()
                reward -= np.log(1 + total_cost)
                info["cost"] = total_cost
                done = True
            else:
                reward -= 0.1

        return self._get_state(), reward, done, info

    def _get_state(self):
        demand_normalized = self.current_demand / np.array([p[2] for p in self.products])
        pattern_normalized = self.current_pattern / (np.max(self.current_pattern) + 1e-5)
        utilization = (self.stock_size[0]*self.stock_size[1] - self.remaining_area) / (self.stock_size[0]*self.stock_size[1])
        return np.concatenate([demand_normalized, pattern_normalized, [utilization]])

    def _check_packing(self, pattern):
        products_to_pack = []
        for i in range(len(pattern)):
            if pattern[i] > 0:
                products_to_pack.append((self.products[i][0], self.products[i][1], pattern[i]))
        return can_pack_all_products_guillotine(self.stock_size, products_to_pack)

    def _calculate_total_cost(self):
        """
        Tính cost = (số pattern * C_su) + (số tờ in * C_ss)
        Dùng LP (linprog) để tìm x_j tối ưu, chỉ xét các sản phẩm có mặt.
        """
        if not self.patterns:
            return 0

        A_list = []
        b_list = []
        for i in range(len(self.products)):
            total_i = sum(p[i] for p in self.patterns)
            if total_i > 0:
                A_list.append([-p[i] for p in self.patterns])
                b_list.append(-self.products[i][2])
        if len(A_list) == 0:
            return 0

        A_ub = np.array(A_list)
        b_ub = np.array(b_list)
        c = [self.C_ss] * len(self.patterns)

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            x = np.ceil(res.x)
            total_cost = len(self.patterns)*self.C_su + np.sum(x)*self.C_ss
        else:
            total_cost = 999999
        return total_cost

    def render(self, mode='human'):
        print("Demand:", self.current_demand)
        print("Current Pattern:", self.current_pattern)
        print("Remaining Area:", self.remaining_area)

    def close(self):
        pass