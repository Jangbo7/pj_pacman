import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_and_mark_head(img_rgb, target_color=[228, 111, 111], tolerance = 1):
    target_color = np.array(target_color)
    lower_bound = np.array([max(0, target_color[0] - tolerance), max(0, target_color[1] - tolerance), max(0, target_color[2] - tolerance)])
    upper_bound = np.array([min(255, target_color[0] + tolerance), min(255, target_color[1] + tolerance), min(255, target_color[2] + tolerance)])
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        print(f"警告: 未找到头部 (颜色: {target_color})")
        return img_rgb.copy(), None, points
    center = np.mean(points, axis=0).astype(int)
    center_y, center_x = center
    center_y=center_y+17
    center=(center_x, center_y)
    rag = []
    y_max = np.max(points[:, 0]).astype(int)
    x_max = np.max(points[:, 1]).astype(int)
    y_min = np.min(points[:, 0]).astype(int)
    x_min = np.min(points[:, 1]).astype(int)
    rag.append([y_max, x_max, y_min, x_min])

    
    marked_img = img_rgb.copy()
    cv2.rectangle(marked_img, (center_x-4, center_y-23), (center_x+4, center_y-4), (0, 255, 0), 1)#绿框
    
    return marked_img, center, points

# def find_and_mark_head(img_rgb, target_color=[210, 75, 75], tolerance=[20, 15, 15]):
#     """
#     修改后的头部检测函数，使用新的颜色范围
#     R: 210±20 (190-230)
#     G: 75±15 (60-90)
#     B: 75±15 (60-90)
#     """
#     target_color = np.array(target_color)
    
#     # 分别计算每个通道的上下界
#     lower_bound = np.array([
#         max(0, target_color[0] - tolerance[0]),
#         max(0, target_color[1] - tolerance[1]),
#         max(0, target_color[2] - tolerance[2])
#     ])
    
#     upper_bound = np.array([
#         min(255, target_color[0] + tolerance[0]),
#         min(255, target_color[1] + tolerance[1]),
#         min(255, target_color[2] + tolerance[2])
#     ])
    
#     mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
#     points = np.column_stack(np.where(mask > 0))
    
#     if len(points) == 0:
#         print(f"警告: 未找到头部 (颜色范围: R{lower_bound[0]}-{upper_bound[0]}, G{lower_bound[1]}-{upper_bound[1]}, B{lower_bound[2]}-{upper_bound[2]})")
#         return img_rgb.copy(), None, points
    
#     # 计算中心
#     center = np.mean(points, axis=0).astype(int)
#     center_y, center_x = center
#     center_y=center_y+40
#     center = (center_x, center_y)
    
#     # 计算边界
#     y_max = np.max(points[:, 0]).astype(int)
#     x_max = np.max(points[:, 1]).astype(int)
#     y_min = np.min(points[:, 0]).astype(int)
#     x_min = np.min(points[:, 1]).astype(int)
    
#     # 标记图像
#     marked_img = img_rgb.copy()
#     cv2.rectangle(marked_img, (center_x-2, center_y-6), (center_x+6, center_y+13), (0, 255, 0), 1)
    
#     # # 可选：添加一个小圆点标记实际中心
#     # cv2.circle(marked_img, (center_x, center_y), 1, (255, 0, 0), -1)
    
#     print(f"找到头部: ({center_x}, {center_y}), 像素点: {len(points)}个")
    
#     return marked_img, center, points

def find_and_mark_sword(img_rgb, target_color=[214, 214, 214], tolerance = 1):
    target_color = np.array(target_color)
    lower_bound = np.array([max(0, target_color[0] - tolerance), max(0, target_color[1] - tolerance), max(0, target_color[2] - tolerance)])
    upper_bound = np.array([min(255, target_color[0] + tolerance), min(255, target_color[1] + tolerance), min(255, target_color[2] + tolerance)])
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    points = np.column_stack(np.where(mask > 0))
    if (len(points) == 0):
        print("未找到匹配颜色的点")
        return img_rgb, None, None
    
    center = np.mean(points, axis=0).astype(int)
    center_y, center_x = center
    
    marked_img = img_rgb.copy()
    cv2.rectangle(marked_img, (center_x-3, center_y-8), (center_x+3, center_y+8), (255,0, 0), 1)#红框
    
    return marked_img, center, points

def is_rects_adjacent(rect1, rect2, threshold=20):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    cx1, cy1 = x1 + w1//2, y1 + h1//2
    cx2, cy2 = x2 + w2//2, y2 + h2//2
    distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return distance < threshold

def cluster_black_rects(valid_black_rects, threshold=20):
    clusters = []
    if not valid_black_rects:
        return clusters
    clusters.append([valid_black_rects[0]])
    for rect in valid_black_rects[1:]:
        clustered = False
        for cluster in clusters:
            for c_rect in cluster:
                if is_rects_adjacent(rect, c_rect, threshold):
                    cluster.append(rect)
                    clustered = True
                    break
            if clustered:
                break
        if not clustered:
            clusters.append([rect])
    return clusters

def find_and_mark_ladder(img_rgb, 
                         color1_low=[0, 0, 0], color1_high=[30, 30, 30], 
                         color2=[66, 158, 130],
                         ladder_w_min=4, ladder_w_max=10,
                         ladder_h_min=4, ladder_h_max=10,
                         adjacent_threshold=20,
                         large_black_min_size=50):
    """
    识别梯子并返回梯子的顶部和底部中心点
    """
    # 1. 提取黑色（含嵌套小轮廓）和绿色掩码
    mask_black = cv2.inRange(img_rgb, np.array(color1_low), np.array(color1_high))
    mask_green = cv2.inRange(img_rgb, np.array(color2), np.array(color2))
    
    # 2. 提取所有层级的黑色轮廓（包括大黑块内部的小轮廓）
    contours, hierarchy = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_black_rects = []
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        is_small = (ladder_w_min <= w <= ladder_w_max) and (ladder_h_min <= h <= ladder_h_max)
        has_large_parent = False
        
        if hierarchy[0][i][3] != -1:  # 存在父轮廓
            parent_idx = hierarchy[0][i][3]
            parent_cnt = contours[parent_idx]
            px, py, pw, ph = cv2.boundingRect(parent_cnt)
            if pw * ph >= large_black_min_size:  # 父轮廓是大黑块
                has_large_parent = True
        
        # 有效条件：小尺寸 且（是独立轮廓 或 嵌套在大黑块内）
        if is_small and (has_large_parent or hierarchy[0][i][3] == -1):
            valid_black_rects.append((x, y, w, h))
    
    if not valid_black_rects:
        return img_rgb.copy(), None
    
    # 3. 聚类
    ladder_black_clusters = cluster_black_rects(valid_black_rects, adjacent_threshold)
    if not ladder_black_clusters:
        return img_rgb.copy(), None
    
    marked_img = img_rgb.copy()
    all_ladder_info = []
    
    for cluster_idx, black_rects in enumerate(ladder_black_clusters):
        # 计算黑色聚类的外接矩形
        cluster_xs = [x for x, y, w, h in black_rects]
        cluster_ys = [y for x, y, w, h in black_rects]
        cluster_ws = [w for x, y, w, h in black_rects]
        cluster_hs = [h for x, y, w, h in black_rects]
        cluster_x = min(cluster_xs)
        cluster_y = min(cluster_ys)
        cluster_w = max(cluster_xs) + max(cluster_ws) - cluster_x
        cluster_h = max(cluster_ys) + max(cluster_hs) - cluster_y
        
        # 提取外接矩形内的绿色区域
        cluster_green_mask = np.zeros_like(mask_green)
        cluster_green_mask[cluster_y:cluster_y+cluster_h, cluster_x:cluster_x+cluster_w] = \
            mask_green[cluster_y:cluster_y+cluster_h, cluster_x:cluster_x+cluster_w]
        
        green_contours, _ = cv2.findContours(cluster_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not green_contours:
            continue
        
        ladder_contour = max(green_contours, key=cv2.contourArea)
        x_ladder, y_ladder, w_ladder, h_ladder = cv2.boundingRect(ladder_contour)
        
        # 计算梯子中心线
        center_x = x_ladder + w_ladder // 2
        
        # 使用绿色梯子轮廓计算顶部和底部
        ladder_points = ladder_contour.reshape(-1, 2)
        
        # 找到梯子最上方和最下方的点
        top_points = ladder_points[ladder_points[:, 1] == np.min(ladder_points[:, 1])]
        bottom_points = ladder_points[ladder_points[:, 1] == np.max(ladder_points[:, 1])]
        
        # 取平均得到顶部和底部中心
        top_center_y = int(np.mean(top_points[:, 1]))
        bottom_center_y = int(np.mean(bottom_points[:, 1]))
        
        # 使用黑色横档辅助修正
        if black_rects:
            black_top_y = min([y for x, y, w, h in black_rects])
            black_bottom_y = max([y + h for x, y, w, h in black_rects])
            
            # 如果黑色横档的范围更准确，使用它
            if black_top_y < top_center_y:
                top_center_y = black_top_y
            if black_bottom_y > bottom_center_y:
                bottom_center_y = black_bottom_y
        
        # 定义顶部和底部点（在中心线上）
        top_point = (center_x, top_center_y)
        bottom_point = (center_x, bottom_center_y)
        
        # 计算中心点
        center_y = (top_center_y + bottom_center_y) // 2
        
        # 标记梯子主体
        cv2.rectangle(marked_img, (x_ladder, y_ladder), 
                     (x_ladder + w_ladder, y_ladder + h_ladder), (0, 255, 0), 1)
        
        # 标记顶部和底部
        cv2.circle(marked_img, top_point, 4, (255, 0, 0), -1)  # 蓝色：顶部
        cv2.circle(marked_img, bottom_point, 4, (0, 0, 255), -1)  # 红色：底部
        
        # 标记中心
        cv2.circle(marked_img, (center_x, center_y), 3, (0, 255, 0), -1)  # 绿色：中心
        
        all_ladder_info.append({
            'ladder_id': cluster_idx,
            'center': (center_x, center_y),
            'top': top_point,      # 正上方中心点 (x, y)
            'bottom': bottom_point,  # 正下方中心点 (x, y)
            'bounds': (x_ladder, y_ladder, w_ladder, h_ladder),
            'black_rects': black_rects
        })
    
    return marked_img, all_ladder_info if all_ladder_info else None

# def find_and_mark_ladder(img_rgb, 
#                          color1_low=[0, 0, 0], color1_high=[30, 30, 30], 
#                          color2=[66, 158, 130],
#                          ladder_w_min=4, ladder_w_max=10,
#                          ladder_h_min=4, ladder_h_max=10,
#                          adjacent_threshold=20,
#                          large_black_min_size=50):  # 大黑块的最小尺寸（过滤干扰）
#     # 1. 提取黑色（含嵌套小轮廓）和绿色掩码
#     mask_black = cv2.inRange(img_rgb, np.array(color1_low), np.array(color1_high))
#     mask_green = cv2.inRange(img_rgb, np.array(color2), np.array(color2))
    
#     # 2. 提取所有层级的黑色轮廓（包括大黑块内部的小轮廓）
#     # 改用RETR_TREE提取所有轮廓，获取层级关系
#     contours, hierarchy = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     valid_black_rects = []
    
#     for i, cnt in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(cnt)
#         # 判断是否是“嵌套在大黑块内的小轮廓”：
#         # - 自身是小尺寸（梯子横档）
#         # - 父轮廓是大尺寸（大黑块）
#         is_small = (ladder_w_min <= w <= ladder_w_max) and (ladder_h_min <= h <= ladder_h_max)
#         has_large_parent = False
        
#         if hierarchy[0][i][3] != -1:  # 存在父轮廓
#             parent_idx = hierarchy[0][i][3]
#             parent_cnt = contours[parent_idx]
#             px, py, pw, ph = cv2.boundingRect(parent_cnt)
#             if pw * ph >= large_black_min_size:  # 父轮廓是大黑块
#                 has_large_parent = True
        
#         # 有效条件：小尺寸 且（是独立轮廓 或 嵌套在大黑块内）
#         if is_small and (has_large_parent or hierarchy[0][i][3] == -1):
#             valid_black_rects.append((x, y, w, h))
    
#     if not valid_black_rects:
#         return img_rgb.copy(), None
    
#     # 3. 聚类+后续逻辑不变
#     ladder_black_clusters = cluster_black_rects(valid_black_rects, adjacent_threshold)
#     if not ladder_black_clusters:
#         return img_rgb.copy(), None
    
#     marked_img = img_rgb.copy()
#     all_ladder_info = []
    
#     for cluster_idx, black_rects in enumerate(ladder_black_clusters):
#         # 计算黑色聚类的外接矩形
#         cluster_xs = [x for x, y, w, h in black_rects]
#         cluster_ys = [y for x, y, w, h in black_rects]
#         cluster_ws = [w for x, y, w, h in black_rects]
#         cluster_hs = [h for x, y, w, h in black_rects]
#         cluster_x = min(cluster_xs)
#         cluster_y = min(cluster_ys)
#         cluster_w = max(cluster_xs) + max(cluster_ws) - cluster_x
#         cluster_h = max(cluster_ys) + max(cluster_hs) - cluster_y
        
#         # 提取外接矩形内的绿色区域
#         cluster_green_mask = np.zeros_like(mask_green)
#         cluster_green_mask[cluster_y:cluster_y+cluster_h, cluster_x:cluster_x+cluster_w] = \
#             mask_green[cluster_y:cluster_y+cluster_h, cluster_x:cluster_x+cluster_w]
        
#         green_contours, _ = cv2.findContours(cluster_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not green_contours:
#             continue
        
#         ladder_contour = max(green_contours, key=cv2.contourArea)
#         x_ladder, y_ladder, w_ladder, h_ladder = cv2.boundingRect(ladder_contour)
        
#         # 合并点集计算顶端/底端
#         ladder_points = ladder_contour.reshape(-1, 2)
#         for (x, y, w, h) in black_rects:
#             ladder_points = np.vstack([ladder_points, [[x, y], [x+w, y+h]]])
        
#         top_point = ladder_points[np.argmin(ladder_points[:, 1])]
#         bottom_point = ladder_points[np.argmax(ladder_points[:, 1])]
#         center_x = x_ladder + w_ladder // 2
#         center_y = y_ladder + h_ladder // 2
        
#         # 标记
#         cv2.rectangle(marked_img, (center_x-4, center_y-6), (center_x+4, center_y+13), (0, 255, 0), 1)
#         cv2.circle(marked_img, tuple(top_point), 3, (255, 0, 0), -1)
#         cv2.circle(marked_img, tuple(bottom_point), 3, (0, 0, 255), -1)
        
#         all_ladder_info.append({
#             'ladder_id': cluster_idx,
#             'center': (center_x, center_y),
#             'top': tuple(top_point),
#             'bottom': tuple(bottom_point),
#             'bounds': (x_ladder, y_ladder, w_ladder, h_ladder),
#             'black_rects': black_rects
#         })
#         print(all_ladder_info)
    
#     return marked_img, all_ladder_info if all_ladder_info else None
def find_and_mark_rope(img_rgb, rope_color=[232, 204, 99], min_length=25):
    """
    查找绳子像素点，只检测长度（厚度可能为0）
    
    返回:
        marked_img: 标记后的图像
        center: 绳子中心点 (x, y)
        points: 绳子像素点
    """
    target_color = np.array(rope_color)
    
    # 精确匹配绳子颜色
    lower_bound = np.array([target_color[0], target_color[1], target_color[2]])
    upper_bound = np.array([target_color[0], target_color[1], target_color[2]])
    
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    # 使用轮廓检测
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return img_rgb.copy(), None, None
    
    # 寻找最长的轮廓
    rope_contour = None
    max_length = 0
    
    for contour in contours:
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        # 取较长的边作为长度
        length = max(width, height)
        
        if length > max_length and length > min_length:
            max_length = length
            rope_contour = contour
    
    if rope_contour is None:
        return img_rgb.copy(), None, None
    
    # 获取该轮廓的所有像素点
    mask_filtered = np.zeros_like(mask)
    cv2.drawContours(mask_filtered, [rope_contour], 0, 255, -1)
    points = np.column_stack(np.where(mask_filtered > 0))
    
    if len(points) == 0:
        return img_rgb.copy(), None, None
    
    # 计算绳子中心
    center = np.mean(points, axis=0).astype(int)
    center_y, center_x = center
    
    # 计算边界
    y_max = np.max(points[:, 0]).astype(int)
    x_max = np.max(points[:, 1]).astype(int)
    y_min = np.min(points[:, 0]).astype(int)
    x_min = np.min(points[:, 1]).astype(int)
    
    # 计算长度（使用像素边界框，可能更准确）
    length = max(y_max - y_min, x_max - x_min)
    
    print(f"找到绳子，长度: {length:.1f}，中心: ({center_x}, {center_y})")
    
    # 标记图像
    marked_img = img_rgb.copy()
    cv2.rectangle(marked_img, (center_x-4, center_y-6), (center_x+4, center_y+13), (0, 255, 0), 1)
    
    # # 可选：绘制绳子的边界框
    # cv2.rectangle(marked_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    
    return marked_img, (center_x, center_y), points


def find_and_mark_key(img_rgb, target_color=[232, 204, 99], tolerance=20, min_black_area=5):
    """
    识别被黄色圈包围的纯黑色区域（钥匙）
    现在环境中只有一个钥匙，返回单个坐标
    
    参数:
    - img_rgb: RGB图像
    - target_color: 黄色圈的颜色 [R, G, B]
    - tolerance: 颜色容差
    - min_black_area: 最小黑色区域面积
    
    返回:
    - marked_img: 标记后的图像
    - center: 钥匙中心坐标 [y, x] 或 None
    - contours: 找到的轮廓列表
    """
    # 创建图像副本
    marked_img = img_rgb.copy()
    
    # 1. 创建黄色掩码
    target_color = np.array(target_color)
    lower_bound = np.array([
        max(0, target_color[0] - tolerance),
        max(0, target_color[1] - tolerance), 
        max(0, target_color[2] - tolerance)
    ])
    upper_bound = np.array([
        min(255, target_color[0] + tolerance),
        min(255, target_color[1] + tolerance),
        min(255, target_color[2] + tolerance)
    ])
    
    yellow_mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    # 2. 查找黄色轮廓
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("未找到黄色轮廓")
        return marked_img, None, None
    
    # 3. 筛选符合要求的轮廓
    valid_centers = []
    
    for i, contour in enumerate(contours):
        # 计算轮廓面积和边界矩形
        area = cv2.contourArea(contour)
        
        # 忽略太小的轮廓
        if area < 5:
            continue
        
        # 获取轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 创建轮廓掩码
        contour_mask = np.zeros_like(yellow_mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
        # 4. 检查黄色轮廓内部是否有足够的纯黑色区域
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([30, 30, 30])
        black_mask = cv2.inRange(img_rgb, black_lower, black_upper)
        
        # 只保留黄色轮廓内部的黑色区域
        black_inside_yellow = cv2.bitwise_and(black_mask, black_mask, mask=contour_mask)
        
        # 计算黑色区域的面积
        black_pixels = np.count_nonzero(black_inside_yellow)
        
        # 检查黑色区域是否足够大
        if black_pixels >= min_black_area:
            # 计算黑色区域的中心
            black_points = np.column_stack(np.where(black_inside_yellow > 0))
            
            if len(black_points) > 0:
                # 计算黑色区域中心
                center_y = int(np.mean(black_points[:, 0]))
                center_x = int(np.mean(black_points[:, 1]))
                
                valid_centers.append([center_x, center_y])
                
                # 绘制标记
                cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 蓝色矩形框
                cv2.circle(marked_img, (center_x, center_y), 3, (255, 255, 0), -1)  # 中心点
    
    if len(valid_centers) == 0:
        print("未找到符合要求的钥匙（黄色圈包围纯黑色区域）")
        return marked_img, None, None
    
    print(f"找到 {len(valid_centers)} 个钥匙候选区域")
    
    # 如果只有一个钥匙，返回单个坐标
    if len(valid_centers) == 1:
        print(f"钥匙位置: y={valid_centers[0][0]}, x={valid_centers[0][1]}")
        return marked_img, valid_centers[0], contours
    else:
        # 如果有多个，可以选择最大的那个
        # 或者保持原样返回列表，但建议统一格式
        print(f"注意：找到 {len(valid_centers)} 个可能的钥匙，返回第一个")
        return marked_img, valid_centers[0], contours
# def find_and_mark_key(img_rgb, target_color=[232, 204, 99], tolerance=20, min_black_area=5):
#     """
#     识别被黄色圈包围的纯黑色区域（钥匙）
    
#     参数:
#     - img_rgb: RGB图像
#     - target_color: 黄色圈的颜色 [R, G, B]
#     - tolerance: 颜色容差
#     - min_black_area: 最小黑色区域面积
    
#     返回:
#     - marked_img: 标记后的图像
#     - centers: 钥匙中心坐标列表
#     - contours: 找到的轮廓列表
#     """
#     # 创建图像副本
#     marked_img = img_rgb.copy()
    
#     # 1. 创建黄色掩码
#     target_color = np.array(target_color)
#     lower_bound = np.array([
#         max(0, target_color[0] - tolerance),
#         max(0, target_color[1] - tolerance), 
#         max(0, target_color[2] - tolerance)
#     ])
#     upper_bound = np.array([
#         min(255, target_color[0] + tolerance),
#         min(255, target_color[1] + tolerance),
#         min(255, target_color[2] + tolerance)
#     ])
    
#     yellow_mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
#     # 2. 查找黄色轮廓
#     contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if len(contours) == 0:
#         print("未找到黄色轮廓")
#         return marked_img, None, None
    

    
#     # 3. 筛选符合要求的轮廓
#     valid_keys = []
#     key_centers = []
    
#     for i, contour in enumerate(contours):
#         # 计算轮廓面积和边界矩形
#         area = cv2.contourArea(contour)
        
#         # 忽略太小的轮廓
#         if area < 5:
#             continue
        
#         # 获取轮廓的边界矩形
#         x, y, w, h = cv2.boundingRect(contour)
        
#         # 创建轮廓掩码
#         contour_mask = np.zeros_like(yellow_mask)
#         cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
#         # 4. 检查黄色轮廓内部是否有足够的纯黑色区域
#         # 纯黑色的RGB阈值：通常RGB都小于30
#         black_lower = np.array([0, 0, 0])
#         black_upper = np.array([30, 30, 30])
#         black_mask = cv2.inRange(img_rgb, black_lower, black_upper)
        
#         # 只保留黄色轮廓内部的黑色区域
#         black_inside_yellow = cv2.bitwise_and(black_mask, black_mask, mask=contour_mask)
        
#         # 计算黑色区域的面积
#         black_pixels = np.count_nonzero(black_inside_yellow)
        
#         # 检查黑色区域是否足够大（至少占轮廓面积的10%）
#         if black_pixels >= min_black_area:
#             # 计算黑色区域的中心
#             black_points = np.column_stack(np.where(black_inside_yellow > 0))
            
#             if len(black_points) > 0:
#                 # 计算黑色区域中心
#                 center_y = int(np.mean(black_points[:, 0]))
#                 center_x = int(np.mean(black_points[:, 1]))
                
#                 valid_keys.append(contour)
#                 key_centers.append([center_y, center_x])
                
#                 # 绘制标记
#                 cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 蓝色矩形框
#                 cv2.circle(marked_img, (center_x, center_y), 3, (255, 255, 0), -1)  # 中心点
                
#                 # # 在图像上添加文字标注
#                 # cv2.putText(marked_img, f"Key {i+1}", (x, y-5), 
#                 #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
#     if len(valid_keys) == 0:
#         print("未找到符合要求的钥匙（黄色圈包围纯黑色区域）")
#         return marked_img, None, None
    
#     print(f"找到 {len(valid_keys)} 把有效的钥匙")
#     return marked_img, key_centers, valid_keys

def find_and_mark_bone(img_rgb, n_clusters=1, target_color=[236, 236, 236], tolerance=1):
    target_color = np.array(target_color)
    lower_bound = np.array([max(0, target_color[0] - tolerance), max(0, target_color[1] - tolerance), max(0, target_color[2] - tolerance)])
    upper_bound = np.array([min(255, target_color[0] + tolerance), min(255, target_color[1] + tolerance), min(255, target_color[2] + tolerance)])
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    # 创建一个屏蔽上方的掩码
    height, width = mask.shape[:2]
    top_mask = np.ones((height, width), dtype=np.uint8) * 255
    top_mask[0:43, :] = 0  # 将前43行设置为0（屏蔽）
    
    # 应用屏蔽：只保留43行以下的区域
    filtered_mask = cv2.bitwise_and(mask, mask, mask=top_mask)
    
    points = np.column_stack(np.where(filtered_mask > 0))
    
    if len(points) == 0:
        print("未找到匹配颜色的点")
        return img_rgb, None, None

    # 计算所有匹配点的中心点
    center_y = np.mean(points[:, 0]).astype(int)
    center_x = np.mean(points[:, 1]).astype(int)
    
    # 计算边界框
    y_max = np.max(points[:, 0]).astype(int)
    x_max = np.max(points[:, 1]).astype(int)
    y_min = np.min(points[:, 0]).astype(int)
    x_min = np.min(points[:, 1]).astype(int)
    
    # 标记图像
    marked_img = img_rgb.copy()
    cv2.rectangle(marked_img, (x_min, y_min), (x_max, y_max), (0, 200, 200), 1)  # 青色
    
    # # 在屏蔽区域上画线，显示屏蔽范围
    # cv2.line(marked_img, (0, 43), (width, 43), (255, 0, 0), 1)  # 蓝色线表示屏蔽边界
    # cv2.putText(marked_img, "Shield Area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    print(f"屏蔽了前43行像素 (y < 43)")
    print(f"找到的有效点数量: {len(points)}")
    
    return marked_img, (center_x, center_y+17), points
# def find_and_mark_bone(img_rgb, n_clusters=1, target_color=[236, 236, 236], tolerance=1):
#     target_color = np.array(target_color)
#     lower_bound = np.array([max(0, target_color[0] - tolerance), max(0, target_color[1] - tolerance), max(0, target_color[2] - tolerance)])
#     upper_bound = np.array([min(255, target_color[0] + tolerance), min(255, target_color[1] + tolerance), min(255, target_color[2] + tolerance)])
#     mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
#     points = np.column_stack(np.where(mask > 0))
#     if len(points) == 0:
#         print("未找到匹配颜色的点")
#         return img_rgb, None, None

#     # 计算所有匹配点的中心点
#     center_y = np.mean(points[:, 0]).astype(int)
#     center_x = np.mean(points[:, 1]).astype(int)
    
#     # 计算边界框
#     y_max = np.max(points[:, 0]).astype(int)
#     x_max = np.max(points[:, 1]).astype(int)
#     y_min = np.min(points[:, 0]).astype(int)
#     x_min = np.min(points[:, 1]).astype(int)
    
#     # 标记图像
#     marked_img = img_rgb.copy()
#     cv2.rectangle(marked_img, (x_min, y_min), (x_max, y_max), (0, 200, 200), 1)  # 青色
    
#     # 返回单个中心点坐标（格式为tuple而不是列表）
#     return marked_img, (center_x, center_y+41), points

def find_and_mark_gate(img_rgb, n_clusters=1, target_color=[101, 111, 228], tolerance=1):
    target_color = np.array(target_color)
    lower_bound = np.array([max(0, target_color[0] - tolerance), max(0, target_color[1] - tolerance), max(0, target_color[2] - tolerance)])
    upper_bound = np.array([min(255, target_color[0] + tolerance), min(255, target_color[1] + tolerance), min(255, target_color[2] + tolerance)])
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        print("未找到匹配颜色的点")
        return img_rgb, None, None
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(points[:, 1].reshape(-1, 1))
    centers = []
    rag = []
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            center_y = np.mean(cluster_points[:, 0]).astype(int)
            center_x = np.mean(cluster_points[:, 1]).astype(int)
            centers.append([center_y, center_x])
            y_max = np.max(cluster_points[:, 0]).astype(int)
            x_max = np.max(cluster_points[:, 1]).astype(int)
            y_min = np.min(cluster_points[:, 0]).astype(int)
            x_min = np.min(cluster_points[:, 1]).astype(int)
            rag.append([y_max, x_max, y_min, x_min])
    
    marked_img = img_rgb.copy()
    
    for i, center in enumerate(centers):
        center_y, center_x = center
        y_max, x_max, y_min, x_min = rag[i]
        cv2.rectangle(marked_img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)#紫色
    
    return marked_img, centers, points