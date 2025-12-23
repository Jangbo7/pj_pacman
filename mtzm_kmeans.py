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
    rag = []
    y_max = np.max(points[:, 0]).astype(int)
    x_max = np.max(points[:, 1]).astype(int)
    y_min = np.min(points[:, 0]).astype(int)
    x_min = np.min(points[:, 1]).astype(int)
    rag.append([y_max, x_max, y_min, x_min])

    
    marked_img = img_rgb.copy()
    cv2.rectangle(marked_img, (center_x-4, center_y-6), (center_x+4, center_y+13), (0, 255, 0), 1)#绿框
    
    return marked_img, center, points

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

def find_and_mark_ladder(img_rgb, color1=[0, 0, 0], color2=[66, 158, 130]):
    # 分别检测两种颜色的独立色块
    mask1 = cv2.inRange(img_rgb, np.array(color1), np.array(color1))
    mask2 = cv2.inRange(img_rgb, np.array(color2), np.array(color2))
    
    # 合并两种颜色的掩码，用于检测整个梯子图形
    mask_combined = cv2.bitwise_or(mask1, mask2)
    
    # 找到合并后的所有轮廓
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return img_rgb.copy(), None
    
    # 找到最大的轮廓（假设最大的就是梯子）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 获取整个梯子图形的边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 计算整个梯子的中心
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 计算最顶端和最底端（使用轮廓的所有点）
    points = largest_contour.reshape(-1, 2)
    top_point = points[np.argmin(points[:, 1])]  # y最小
    bottom_point = points[np.argmax(points[:, 1])]  # y最大
    
    # 标记图像
    marked_img = img_rgb.copy()
    cv2.rectangle(marked_img, (center_x-4, center_y-6), (center_x+4, center_y+13), (0, 255, 0), 1)
    cv2.circle(marked_img, tuple(top_point), 3, (255, 0, 0), -1)
    cv2.circle(marked_img, tuple(bottom_point), 3, (0, 0, 255), -1)
    
    ladder_info = {
        'center': (center_x, center_y),
        'top': tuple(top_point),
        'bottom': tuple(bottom_point),
        'bounds': (x, y, w, h)
    }
    
    return marked_img, ladder_info
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
    
    参数:
    - img_rgb: RGB图像
    - target_color: 黄色圈的颜色 [R, G, B]
    - tolerance: 颜色容差
    - min_black_area: 最小黑色区域面积
    
    返回:
    - marked_img: 标记后的图像
    - centers: 钥匙中心坐标列表
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
    valid_keys = []
    key_centers = []
    
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
        # 纯黑色的RGB阈值：通常RGB都小于30
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([30, 30, 30])
        black_mask = cv2.inRange(img_rgb, black_lower, black_upper)
        
        # 只保留黄色轮廓内部的黑色区域
        black_inside_yellow = cv2.bitwise_and(black_mask, black_mask, mask=contour_mask)
        
        # 计算黑色区域的面积
        black_pixels = np.count_nonzero(black_inside_yellow)
        
        # 检查黑色区域是否足够大（至少占轮廓面积的10%）
        if black_pixels >= min_black_area:
            # 计算黑色区域的中心
            black_points = np.column_stack(np.where(black_inside_yellow > 0))
            
            if len(black_points) > 0:
                # 计算黑色区域中心
                center_y = int(np.mean(black_points[:, 0]))
                center_x = int(np.mean(black_points[:, 1]))
                
                valid_keys.append(contour)
                key_centers.append([center_y, center_x])
                
                # 绘制标记
                cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 蓝色矩形框
                cv2.circle(marked_img, (center_x, center_y), 3, (255, 255, 0), -1)  # 中心点
                
                # # 在图像上添加文字标注
                # cv2.putText(marked_img, f"Key {i+1}", (x, y-5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    if len(valid_keys) == 0:
        print("未找到符合要求的钥匙（黄色圈包围纯黑色区域）")
        return marked_img, None, None
    
    print(f"找到 {len(valid_keys)} 把有效的钥匙")
    return marked_img, key_centers, valid_keys

def find_and_mark_bone(img_rgb, n_clusters=1, target_color=[236, 236, 236], tolerance=1):
    target_color = np.array(target_color)
    lower_bound = np.array([max(0, target_color[0] - tolerance), max(0, target_color[1] - tolerance), max(0, target_color[2] - tolerance)])
    upper_bound = np.array([min(255, target_color[0] + tolerance), min(255, target_color[1] + tolerance), min(255, target_color[2] + tolerance)])
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        print("未找到匹配颜色的点")
        return img_rgb, None, None

    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)
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
        cv2.rectangle(marked_img, (x_min, y_min), (x_max, y_max), (0, 200, 200), 1)#青色
    
    return marked_img, centers, points

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