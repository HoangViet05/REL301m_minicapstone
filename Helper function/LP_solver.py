import numpy as np
from scipy.optimize import linprog

"""
- Ý tưởng
Ý tưởng của tôi là sẽ chỉ tính toán số lượng in có pattern đó nếu nó có product được đưa vào,
và chỉ tính toán dựa trên demand của toàn bộ những product được thêm vào trong pattern đó,
những product nào không được thêm vào pattern đang tính toán hiện tại thì bỏ qua chứ không đưa vào ma trận để tính toán.
"""

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