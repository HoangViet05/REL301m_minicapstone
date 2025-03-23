import gym
from gym import spaces
import numpy as np
from scipy.optimize import linprog
from Best_fit import can_pack_all_products_guillotine, GuillotineStockSheet

class CuttingStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, products, pattern_size, C_su=20, C_ss=1, max_steps=100, max_patterns=10):
        super(CuttingStockEnv, self).__init__()
        
        # Thông số bài toán
        self.products = products  # [(w1, h1, d1), (w2, h2, d2), ...]
        self.pattern_size = pattern_size  # (width, height)
        self.C_su = C_su  # Setup cost per pattern
        self.C_ss = C_ss  # Stock sheet cost

        """
            "spaces" là một module cung cấp classes để đinh nghĩa observation space và action space
                + Discrete: Dùng để định nghĩa không gian rời rạc (ví dụ: số lượng các hành động có thể thực hiện).
                + Box: Dùng để định nghĩa không gian liên tục với các giá trị nằm trong một khoảng nhất định (ví dụ: vị trí, vận tốc,...).
                + MultiDiscrete: Dùng để định nghĩa không gian có nhiều biến rời rạc.
                + Tuple: Dùng để kết hợp nhiều không gian lại với nhau thành một không gian phức hợp.
                + Sequence: Dùng để định nghĩa các chuỗi các phần tử theo một không gian nhất định.
        """

        """
        - Không gian hành động: [action_type, product_idx]
        - Tổng số hành động sẽ bằng "3 * len(products)"
        - action_type gồm 3 hành động ứng với mô tả
            + 0: Thêm sản phẩm vào pattern
            + 1: Xóa sản phẩm khỏi pattern
            + 2: Xác nhận pattern
        - product_idx dùng trong các hành động như thêm và xóa sản phẩm
        
        - Không gian rời rạc (Discrete Space) trong reinforcement learning được dùng để mô tả tập hợp các lựa chọn hữu hạn. 
        """
        self.action_space = spaces.MultiDiscrete([3, len(products)])
        
        """
        - Không gian trạng thái: Demand + Pattern hiện tại + Diện tích trống
        - Được thiết dưới dạng 1 vector liên tục có kích thước len(products) + len(products) + 1
        - Trong đó gồm những thông tin mà agent cần quan sát như:
            + Demand: Gồm "len(products)" phần từ => Cho agent biết cần sán xuất bao nhiêu tấm vs mỗi product
            + Current Pattern: Gồm "len(products)" phần từ, mỗi phần tử là số lượng tấm đã được thêm vào patten hiện tại của mỗi product
                            => Cho biết tình trạng hiện tại của pattern để đưa ra quyết định thêm product hoặc không
            + Remaining Area (Diện tích trống còn lại của pattern) => Cho agent đánh giá được khả năng chứa của patterns từ đó đưa ra quyết định có thêm product hay không
        """
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
        self.remaining_area = self.pattern_size[0] * self.pattern_size[1]
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
        info = {"cost": 0.0}  # Mặc định cost = 0, sẽ cập nhật khi done
        
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
                    # (SỬA) Thưởng nhỏ khi thêm sản phẩm thành công
                    reward += 5  
                    # Thêm thưởng theo utilization (khuyến khích lấp đầy)
                    utilization = (self.pattern_size[0]*self.pattern_size[1] - self.remaining_area) / (self.pattern_size[0]*self.pattern_size[1])
                    reward += 10 * utilization
                else:
                    reward -= 2
        elif action_type == 1:  # Xóa sản phẩm
            if self.current_pattern[product_idx] > 0:
                self.current_pattern[product_idx] -= 1
                self.current_demand[product_idx] += 1
                reward -= 1
        elif action_type == 2:  # Xác nhận pattern
            if len(self.patterns) >= self.max_patterns:
                reward -= 50  # phạt nếu vượt quá số pattern
            if np.sum(self.current_pattern) > 0:
                self.patterns.append(self.current_pattern.copy())
                self.current_pattern = np.zeros(len(self.products), dtype=np.int32)
                self.remaining_area = self.pattern_size[0]*self.pattern_size[1]
                reward += 20  # thưởng khi xác nhận pattern
            else:
                reward -= 10
        
        # -------------------------
        # Kiểm tra điều kiện kết thúc
        # -------------------------
        if np.all(self.current_demand == 0):
            # (SỬA) Tính cost khi hoàn thành
            total_cost = self._calculate_total_cost()
            # (SỬA) Thưởng cơ bản khi hoàn thành, sau đó trừ penalty cost theo log
            reward += 100  
            reward -= np.log(1 + total_cost)
            info["cost"] = total_cost
            done = True
        else:
            if self.current_step >= self.max_steps:
                reward -= 50  # phạt nặng nếu hết bước mà chưa xong
                total_cost = self._calculate_total_cost()
                reward -= np.log(1 + total_cost)
                info["cost"] = total_cost
                done = True
            else:
                reward -= 1  # phạt nhẹ mỗi bước
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        # Chuẩn hóa demand, pattern và tính utilization của pattern hiện tại
        demand_normalized = self.current_demand / np.array([p[2] for p in self.products])
        pattern_normalized = self.current_pattern / (np.max(self.current_pattern) + 1e-5)
        utilization = (self.pattern_size[0]*self.pattern_size[1] - self.remaining_area) / (self.pattern_size[0]*self.pattern_size[1])
        return np.concatenate([demand_normalized, pattern_normalized, [utilization]])
    
    def _check_packing(self, pattern):
        # Gọi best-fit guillotine để kiểm tra khả năng xếp sản phẩm
        products_to_pack = []
        for i in range(len(pattern)):
            if pattern[i] > 0:
                products_to_pack.append((self.products[i][0], self.products[i][1], pattern[i]))
        return can_pack_all_products_guillotine(self.pattern_size, products_to_pack)
    
    def _calculate_total_cost(self):
        """
        Tính cost = (số pattern * C_su) + (số tờ in * C_ss)
        Dùng LP (linprog) để tìm x_j tối ưu, chỉ xét các sản phẩm có mặt.
        """
        if not self.patterns:
            return 0  # Nếu chưa có pattern nào, cost = 0
        
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
            total_cost = 999999  # nếu LP thất bại
        return total_cost
    
    def render(self, mode='human'):
        print("Demand:", self.current_demand)
        print("Current Pattern:", self.current_pattern)
        print("Remaining Area:", self.remaining_area)

    def close(self):
        pass