import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image
import sys
sys.stdout.reconfigure(encoding='utf-8')

def generate_colors(n):
    colors = []
    cmap = plt.get_cmap('tab20')
    for i in range(n):
        colors.append(cmap(i % 20))
    return colors

class GuillotineStockSheet:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Ban đầu, toàn bộ tấm stock là một vùng trống duy nhất: (x, y, w, h)
        self.free_rectangles = [(0, 0, width, height)]
        self.placements = []  # Các sản phẩm đã được đặt, mỗi phần tử (x, y, w, h)

    def find_free_rect_for(self, rect_w, rect_h):
        """
        Duyệt qua danh sách các vùng trống để tìm vùng có thể chứa hình chữ nhật (rect_w, rect_h)
        với tiêu chí best-fit (diện tích dư thừa nhỏ nhất).
        Trả về (index, free_rect) nếu tìm thấy, ngược lại None.
        """
        best_index = None
        best_fit_area = float('inf')
        for i, (fx, fy, fw, fh) in enumerate(self.free_rectangles):
            if rect_w <= fw and rect_h <= fh:
                area_diff = (fw * fh) - (rect_w * rect_h)
                if area_diff < best_fit_area:
                    best_fit_area = area_diff
                    best_index = i
        if best_index is not None:
            return best_index, self.free_rectangles[best_index]
        return None

    def split_free_rect(self, free_rect, placed_rect):
        """
        Tách vùng trống free_rect khi đặt placed_rect vào bên trong free_rect,
        theo phương pháp Guillotine (chia theo đường cắt chạy toàn bộ).
        free_rect: (fx, fy, fw, fh)
        placed_rect: (px, py, pw, ph) – nằm trong free_rect.
        Trả về danh sách các vùng trống mới (có diện tích > 0).
        """
        fx, fy, fw, fh = free_rect
        px, py, pw, ph = placed_rect
        new_rects = []
        # Vùng bên phải của placed_rect:
        right_w = (fx + fw) - (px + pw)
        if right_w > 0:
            new_rects.append((px + pw, py, right_w, ph))
        # Vùng bên trên placed_rect:
        top_h = (fy + fh) - (py + ph)
        if top_h > 0:
            new_rects.append((px, py + ph, fw, top_h))
        return new_rects

    def place_rect(self, w, h):
        """
        Thử đặt hình chữ nhật với kích thước (w, h) vào tấm stock.
        Hỗ trợ xoay 90°: hàm sẽ thử cả (w, h) và (h, w) nếu chúng khác nhau.
        Sử dụng chiến lược best-fit dựa trên danh sách free_rectangles.
        Nếu thành công, cập nhật placements và free_rectangles (theo Guillotine split) và trả về True.
        Nếu không tìm được vùng phù hợp cho bất kỳ hướng nào, trả về False.
        """
        candidates = []

        # Thử đặt với hướng ban đầu (w, h)
        res1 = self.find_free_rect_for(w, h)
        if res1 is not None:
            index1, free_rect1 = res1
            fx1, fy1, fw1, fh1 = free_rect1
            area_diff1 = (fw1 * fh1) - (w * h)
            candidates.append((area_diff1, index1, free_rect1, w, h))

        # Nếu sản phẩm không vuông, thử xoay 90°: (h, w)
        if w != h:
            res2 = self.find_free_rect_for(h, w)
            if res2 is not None:
                index2, free_rect2 = res2
                fx2, fy2, fw2, fh2 = free_rect2
                area_diff2 = (fw2 * fh2) - (w * h)  # diện tích sản phẩm vẫn bằng w*h
                candidates.append((area_diff2, index2, free_rect2, h, w))

        if not candidates:
            return False

        # Chọn ứng viên với diện tích dư thừa nhỏ nhất
        candidates.sort(key=lambda x: x[0])
        _, index, free_rect, used_w, used_h = candidates[0]
        fx, fy, fw, fh = free_rect
        placed_rect = (fx, fy, used_w, used_h)
        self.placements.append(placed_rect)
        del self.free_rectangles[index]
        new_rects = self.split_free_rect(free_rect, placed_rect)
        for r in new_rects:
            if r[2] > 0 and r[3] > 0:
                self.free_rectangles.append(r)
        return True

def can_pack_all_products_guillotine(stock_size, products):
    """
    Kiểm tra xem với một tấm stock duy nhất có kích thước stock_size (width, height)
    có thể xếp được tất cả các sản phẩm theo yêu cầu Guillotine hay không.
    
    Input:
        - stock_size: tuple (width, height) của tấm stock.
        - products: danh sách các sản phẩm, mỗi sản phẩm là tuple (w, h, demand)
                với w, h là kích thước sản phẩm và demand là số lượng cần xếp.
    
    Output:
        - Trả về tuple (packable, waste_area) trong đó:
            + packable là True nếu tất cả sản phẩm được xếp theo yêu cầu Guillotine, ngược lại False.
            + waste_area là diện tích phần lãng phí trên tấm stock nếu xếp được, hoặc None nếu không xếp được.
    """
    stock_width, stock_height = stock_size

    # Kiểm tra nhanh: nếu kích thước sản phẩm vượt quá kích thước stock (cả khi xoay 90°) thì không thể xếp.
    for w, h, demand in products:
        if (w > stock_width or h > stock_height) and (h > stock_width or w > stock_height):
            return (False, None)

    # Sắp xếp sản phẩm theo diện tích giảm dần để đặt những sản phẩm lớn trước.
    sorted_products = sorted(products, key=lambda x: x[0] * x[1], reverse=True)
    sheet = GuillotineStockSheet(stock_width, stock_height)
    for w, h, demand in sorted_products:
        for _ in range(demand):
            if not sheet.place_rect(w, h):
                return (False, None)

    # Nếu đã xếp được tất cả, tính diện tích lãng phí:
    used_area = sum(w * h for (_, _, w, h) in sheet.placements)
    total_area = stock_width * stock_height
    waste_area = total_area - used_area
    return (True, waste_area)

# ------------------ PHẦN TEST ------------------

if __name__ == "__main__":
    # Kích thước của tấm stock duy nhất (ví dụ: 200 x 150)
    stock_size = (200, 150)
    
    # Danh sách sản phẩm: mỗi sản phẩm là (width, height, demand)
    # Ví dụ: (50, 40, 2) nghĩa là sản phẩm có kích thước 50x40 cần xếp 2 lần.
    products = [
        (50, 40, 2),
        (60, 30, 2),
        (40, 40, 3),
        (70, 50, 2),
        (10, 80, 9),
        (30, 20, 1)
    ]
    
    packable, waste = can_pack_all_products_guillotine(stock_size, products)
    print("Có thể xếp tất cả sản phẩm theo Guillotine (với hỗ trợ xoay 90°) không?", packable)
    if packable:
        print("Diện tích lãng phí:", waste)
    else:
        print("Không thể xếp được tất cả sản phẩm.")