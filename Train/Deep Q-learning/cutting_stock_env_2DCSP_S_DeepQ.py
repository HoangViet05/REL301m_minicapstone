# ===============================
# Environment: CuttingStockEnv
# ===============================
class CuttingStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, product_ranges, pattern_size_range, C_su=20, C_ss=1, max_steps=100, max_patterns=10):
        super(CuttingStockEnv, self).__init__()

        # Thông số bài toán
        # self.products = products  # [(w1, h1, d1), (w2, h2, d2), ...]
        # self.stock_size = stock_size  # (width, height)
        self.C_su = C_su  # Setup cost per pattern
        self.C_ss = C_ss  # Stock sheet cost

        self.product_ranges = product_ranges  # [(w_min, w_max, h_min, h_max, d_min, d_max), ...]
        self.pattern_size_range = pattern_size_range  # (min_w, max_w, min_h, max_h)
        self._generate_dynamic_config()
        
        # Không gian hành động: [action_type, product_idx]
        self.action_space = spaces.MultiDiscrete([3, len(self.products)])
        
        # Không gian trạng thái: Demand + Pattern hiện tại + Diện tích trống
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(len(self.products)*2 + 1 + 2,),  # Thêm 2 cho pattern_size_normalized
            dtype=np.float32
        )
        
        # Các tham số môi trường
        self.max_steps = max_steps
        self.max_patterns = max_patterns
        
        self.reset()

    def _generate_dynamic_config(self):
        # Sinh sản phẩm và pattern size ngẫu nhiên
        self.products = [
            (
                np.random.randint(w_min, w_max + 1),
                np.random.randint(h_min, h_max + 1),
                np.random.randint(d_min, d_max + 1)
            ) for (w_min, w_max, h_min, h_max, d_min, d_max) in self.product_ranges
        ]
        self.pattern_size = (
            np.random.randint(self.pattern_size_range[0], self.pattern_size_range[1] + 1),
            np.random.randint(self.pattern_size_range[2], self.pattern_size_range[3] + 1)
        )

    def reset(self):
        self._generate_dynamic_config()
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
                    product_area = self.products[product_idx][0] * self.products[product_idx][1]
                    self.remaining_area = max(self.remaining_area - product_area, 0)
                    reward += 5
                    utilization = (self.pattern_size[0]*self.pattern_size[1] - self.remaining_area) / (self.pattern_size[0]*self.pattern_size[1])
                    reward += 50 * utilization
                else:
                    reward -= 2
        elif action_type == 1:  # Xóa sản phẩm
            if self.current_pattern[product_idx] > 0:
                self.current_pattern[product_idx] -= 1
                self.current_demand[product_idx] += 1
                product_area = self.products[product_idx][0] * self.products[product_idx][1]
                self.remaining_area += product_area
                reward -= 1
        elif action_type == 2:  # Xác nhận pattern
            if len(self.patterns) >= self.max_patterns:
                reward -= 50
            if np.sum(self.current_pattern) > 0:
                self.patterns.append(self.current_pattern.copy())
                self.current_pattern = np.zeros(len(self.products), dtype=np.int32)
                self.remaining_area = self.pattern_size[0] * self.pattern_size[1]
                reward += 50 * (1 - len(self.patterns)/self.max_patterns)
            else:
                reward -= 10

        # -------------------------
        # Kiểm tra điều kiện kết thúc
        # -------------------------
        if np.all(self.current_demand == 0):
            if np.sum(self.current_pattern) > 0:
                self.patterns.append(self.current_pattern.copy())
                self.current_pattern = np.zeros(len(self.products), dtype=np.int32)
                self.remaining_area = self.pattern_size[0] * self.pattern_size[1]
            total_cost = self._calculate_total_cost()
            reward += 1000
            reward -= 0.1 * total_cost
            info["cost"] = total_cost
            done = True
        else:
            if self.current_step >= self.max_steps:
                missing_demand = np.sum(self.current_demand)
                total_cost = self._calculate_total_cost()
                reward -= (100 + 5 * missing_demand)
                info["cost"] = total_cost
                done = True
            else:
                reward -= 0.11

        return self._get_state(), reward, done, info

    def _get_state(self):
        demand_normalized = self.current_demand / np.array([p[2] for p in self.products])
        pattern_normalized = self.current_pattern / (np.max(self.current_pattern) + 1e-5)
        utilization = (self.pattern_size[0] * self.pattern_size[1] - self.remaining_area) / (self.pattern_size[0] * self.pattern_size[1])
        pattern_size_normalized = [self.pattern_size[0] / 300, self.pattern_size[1] / 300]
        return np.concatenate([demand_normalized, pattern_normalized, [utilization], pattern_size_normalized])

    def _check_packing(self, pattern):
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

    def visualize_patterns(self):
        """
        Vẽ trực quan các pattern đã xác nhận bằng cách sử dụng thuật toán best-fit
        để xác định vị trí các sản phẩm trên pattern.
        """
        if not self.patterns:
            print("Không có pattern nào được xác nhận để hiển thị.")
            return
        
        n_patterns = len(self.patterns)
        fig, axes = plt.subplots(1, n_patterns, figsize=(5 * n_patterns, 5))
        if n_patterns == 1:
            axes = [axes]
        
        for idx, pattern in enumerate(self.patterns):
            ax = axes[idx]
            ax.set_title(f"Pattern {idx+1}\nSize: {self.pattern_size}")
            ax.set_xlim(0, self.pattern_size[0])
            ax.set_ylim(0, self.pattern_size[1])
            ax.invert_yaxis()  # Đảo trục y để mô phỏng hệ tọa độ trên giấy
            ax.set_aspect('equal')
            
            # Tạo danh sách sản phẩm có trong pattern theo dạng (width, height, quantity, product_idx)
            products_to_pack = []
            for i in range(len(pattern)):
                if pattern[i] > 0:
                    products_to_pack.append((self.products[i][0], self.products[i][1], pattern[i], i))
            # Sử dụng GuillotineStockSheet để xếp sản phẩm
            gs = GuillotineStockSheet(self.pattern_size[0], self.pattern_size[1])
            try:
                placements = gs.place_products(products_to_pack)
            except AttributeError:
                print("Method 'place_products' không tồn tại trong GuillotineStockSheet. Sử dụng giải pháp tạm thời.")
                # Tạo giải pháp tạm: đặt các sản phẩm liền nhau theo thứ tự
                placements = []
                x, y = 0, 0
                max_row_height = 0
                for prod in products_to_pack:
                    w, h, qty, prod_idx = prod
                    for _ in range(qty):
                        # Nếu vượt quá width pattern, chuyển hàng
                        if x + w > self.pattern_size[0]:
                            x = 0
                            y += max_row_height
                            max_row_height = 0
                        placements.append({
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "product_idx": prod_idx
                        })
                        x += w
                        if h > max_row_height:
                            max_row_height = h

            # Vẽ các hình chữ nhật theo placements
            from matplotlib.patches import Rectangle
            for placement in placements:
                rect = Rectangle(
                    (placement["x"], placement["y"]),
                    placement["width"],
                    placement["height"],
                    edgecolor="black",
                    facecolor=np.random.rand(3,),
                    alpha=0.6
                )
                ax.add_patch(rect)
                ax.text(
                    placement["x"] + placement["width"] / 2,
                    placement["y"] + placement["height"] / 2,
                    f"P{placement['product_idx']}",
                    color="black",
                    weight="bold",
                    ha="center",
                    va="center"
                )
        plt.tight_layout()
        plt.show()

    def close(self):
        pass
