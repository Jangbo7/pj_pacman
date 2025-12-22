import cv2
import numpy as np

class cv_supervisor:
    def __init__(self, env_img,args,iter,epoch):
        self.env_img = env_img
        self.args = args
        self.iter = iter
        self.epoch = epoch

    def preprocess_env_img(self):
        """
       return: processed_env_img
       
        """
        processed_env_img = cv2.resize(self.env_img, (self.args.size, self.args.size))
        if self.args.capture:
            cv2.imwrite(f"env_img_epoch_{self.epoch}_iter_{self.iter}.png", processed_env_img)
        return processed_env_img
    
    def analyze_env_img(self):
        """
        分析图像中的前10种主要颜色并生成色卡
        return: color analysis result 包括主要颜色RGB值和色卡图像
        """
        # 如果没有提供预处理后的图像，则使用原始图像并预处理
        processed_env_img = self.preprocess_env_img()
        
        # 将图像转换为RGB格式（如果当前是BGR格式）
        if len(processed_env_img.shape) == 3 and processed_env_img.shape[2] == 3:
            img_rgb = cv2.cvtColor(processed_env_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = processed_env_img
        
        # 获取图像中所有唯一的颜色
        if len(img_rgb.shape) == 3:
            # 彩色图像
            pixels = img_rgb.reshape(-1, 3)
        else:
            # 灰度图像
            pixels = img_rgb.reshape(-1, 1)
        
        # 找到唯一的颜色及其出现次数
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # 按出现次数排序
        sorted_indices = np.argsort(counts)[::-1]
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]
        
        # 获取前10种主要颜色（如果没有那么多就取所有颜色）
        num_main_colors = min(10, len(unique_colors))
        main_colors = unique_colors[:num_main_colors]
        main_counts = counts[:num_main_colors]
        
        # 构建分析结果
        analysis_result = {
            'unique_colors': unique_colors,
            'color_counts': counts,
            'main_colors': main_colors,
            'main_counts': main_counts,
            'processed_image': processed_env_img
        }
        
        # 生成色卡
        color_chart = self._generate_color_chart(main_colors, main_counts)
        analysis_result['color_chart'] = color_chart
        
        # 如果启用捕获功能，保存色卡
        if hasattr(self.args, 'capture') and self.args.capture:
            # 检查色卡是否为空
            if color_chart is not None and color_chart.size > 0:
                try:
                    cv2.imwrite(f"color_chart_result_{self.epoch}_iter_{self.iter}.png", 
                               cv2.cvtColor(color_chart, cv2.COLOR_RGB2BGR))
                except cv2.error:
                    # 如果转换失败，直接保存RGB图像
                    cv2.imwrite(f"color_chart_result_{self.epoch}_iter_{self.iter}.png", 
                               color_chart)
        
        return analysis_result
    
    def classify_clusters_by_area(self, clusters, area_ranges=None):
        """
        根据面积大小对聚类进行分类
        
        Args:
            clusters: 聚类信息列表
            area_ranges: 面积范围定义 [(min_area, max_area, class_name), ...]
                        如果为None，则使用默认分类
            
        Returns:
            classified_clusters: 按类别分组的聚类信息
        """
        # 默认面积分类规则
        if area_ranges is None:
            area_ranges = [
                (0, 50, "small"),
                (50, 150, "medium"),
                (150, 500, "large")
            ]
        
        # 初始化分类结果
        classified_clusters = {}
        for _, _, class_name in area_ranges:
            classified_clusters[class_name] = []
        
        # 为每个聚类分配类别
        for cluster in clusters:
            area = cluster['area']
            assigned = False
            
            for min_area, max_area, class_name in area_ranges:
                if min_area <= area < max_area:
                    cluster['class'] = class_name
                    cluster['class_id'] = list(dict.fromkeys([name for _, _, name in area_ranges])).index(class_name)
                    classified_clusters[class_name].append(cluster)
                    assigned = True
                    break
            
            # 如果没有匹配任何类别，则归为"default"类
            if not assigned:
                cluster['class'] = 'default'
                cluster['class_id'] = len(area_ranges)
                if 'default' not in classified_clusters:
                    classified_clusters['default'] = []
                classified_clusters['default'].append(cluster)
        
        return classified_clusters
    
    def classify_game_objects(self, clusters):
        """
        根据对象大小对游戏对象进行分类
        最大的对象归为ghost类，第二大的对象归类为pacman
        
        Args:
            clusters: 聚类信息列表
            
        Returns:
            classified_clusters: 添加了类别信息的聚类列表
        """
        # 按面积大小排序聚类
        sorted_clusters = sorted(clusters, key=lambda x: x['area'], reverse=True)
        
        # 定义类别映射
        class_mapping = {0: 'ghost', 1: 'pacman'}
        
        # 为每个聚类分配类别
        for i, cluster in enumerate(sorted_clusters):
            # 根据索引分配类别
            if i < 2:  # 只为前两个对象分配特殊类别
                cluster['object_class'] = class_mapping[i]
                cluster['class_index'] = i
            else:  # 其他对象标记为普通对象
                cluster['object_class'] = 'other'
                cluster['class_index'] = 2
                
        return sorted_clusters
    
    def extract_multiple_colors_clusters(self, target_colors, min_area=8, max_area=400, classify_objects=False):
        """
        提取并框出多种指定颜色的聚类
        
        Args:
            target_colors: 目标颜色列表，每个颜色为(R, G, B)或灰度值
            min_area: 最小聚类面积
            max_area: 最大聚类面积
            classify_objects: 是否对游戏对象进行分类
            
        Returns:
            annotated_image: 带有边界框标记的图像
            clusters: 聚类信息列表
        """
        # 预处理图像
        processed_env_img = self.preprocess_env_img()
        
        # 使用原始BGR图像进行处理
        img_bgr = processed_env_img
        
        # 存储所有聚类信息
        all_clusters = []
        
        # 为每种颜色创建掩码并查找聚类
        for color_idx, target_color in enumerate(target_colors):
            # 为指定颜色创建掩码（注意OpenCV使用BGR格式）
            if len(img_bgr.shape) == 3 and hasattr(target_color, '__len__') and len(target_color) == 3:
                # 转换RGB颜色为BGR
                bgr_color = np.array([target_color[2], target_color[1], target_color[0]], dtype=np.uint8)
                mask = cv2.inRange(img_bgr, bgr_color, bgr_color)
            else:
                mask = cv2.inRange(img_bgr, target_color, target_color)
            
            # 查找轮廓（即聚类）使用完整层级结构以处理带孔洞的对象
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤轮廓基于面积
            filtered_contours = []
            filtered_hierarchy = []
            
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    filtered_contours.append(cnt)
                    if hierarchy is not None:
                        filtered_hierarchy.append(hierarchy[0][i])
                    else:
                        filtered_hierarchy.append(None)
            
            # 为每个轮廓计算边界框
            for i, contour in enumerate(filtered_contours):
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                cluster_info = {
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': cv2.contourArea(contour),
                    'color_index': color_idx,  # 记录颜色索引
                    'color': target_color      # 记录颜色值
                }
                
                # 添加层级信息（如果有）
                if i < len(filtered_hierarchy) and filtered_hierarchy[i] is not None:
                    cluster_info['hierarchy'] = filtered_hierarchy[i]
                    
                all_clusters.append(cluster_info)
        
        # 在原始预处理图像上绘制边界框，保留原始图像背景
        if classify_objects:
            # 如果需要分类对象，则先进行分类再绘制
            classified_clusters = self.classify_game_objects(all_clusters)
            annotated_image = self._draw_classified_clusters_on_image(processed_env_img, classified_clusters)
        else:
            # 否则使用原有的绘制方法
            annotated_image = self._draw_multiple_clusters_on_image(processed_env_img, all_clusters)
        
        # 如果启用捕获功能，保存结果
        if hasattr(self.args, 'capture') and self.args.capture:
            filename = f"classified_objects_{self.epoch}_iter_{self.iter}.png" if classify_objects else f"selected_multiple_colors_clusters_{self.epoch}_iter_{self.iter}.png"
            cv2.imwrite(filename, annotated_image)
        
        # 返回聚类信息，如果分类了对象则返回分类后的结果
        if classify_objects:
            return annotated_image, classified_clusters
        else:
            return annotated_image, all_clusters
    
    def detect_pills(self, target_color, min_area=1, max_area=20, min_count=5):
        """
        检测图片中数量多的小型聚类（用于寻找吃豆人游戏中的豆子）
        
        Args:
            target_color: 目标颜色 (R, G, B) 或灰度值
            min_area: 最小聚类面积
            max_area: 最大聚类面积
            min_count: 最少聚类数量阈值
            
        Returns:
            pill_positions: 豆子中心位置列表 [(x, y), ...]
            clusters_count: 检测到的聚类总数
        """
        # 预处理图像
        processed_env_img = self.preprocess_env_img()
        
        # 使用原始BGR图像进行处理
        img_bgr = processed_env_img
        
        # 为指定颜色创建掩码（注意OpenCV使用BGR格式）
        if len(img_bgr.shape) == 3 and hasattr(target_color, '__len__') and len(target_color) == 3:
            # 转换RGB颜色为BGR
            bgr_color = np.array([target_color[2], target_color[1], target_color[0]], dtype=np.uint8)
            mask = cv2.inRange(img_bgr, bgr_color, bgr_color)
        else:
            mask = cv2.inRange(img_bgr, target_color, target_color)
        
        # 查找轮廓（即聚类）使用完整层级结构
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓基于面积
        filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
        
        # 如果聚类数量不够多，认为这不是豆子
        if len(filtered_contours) < min_count:
            return [], 0
        
        # 计算每个聚类的中心位置
        pill_positions = []
        for contour in filtered_contours:
            # 计算轮廓的矩
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                # 计算中心点坐标
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                pill_positions.append((cx, cy))
        
        # 如果启用了捕获功能，保存结果图像
        if hasattr(self.args, 'capture') and self.args.capture:
            # 在图像上绘制豆子位置
            annotated_image = self._draw_pill_positions(processed_env_img, pill_positions)
            cv2.imwrite(f"pill_detection_result_{self.epoch}_iter_{self.iter}.png", annotated_image)
        
        return pill_positions, len(filtered_contours)
    
    def _draw_clusters_on_image(self, image, clusters, target_color):
        """
        在图像上绘制聚类边界框
        
        Args:
            image: 输入图像（将保留原始背景）
            clusters: 聚类信息列表
            target_color: 目标颜色
            
        Returns:
            annotated_image: 带有边界框标记的图像，包含原始图像背景
        """
        # 确保输出图像为BGR三通道格式用于绘制
        if len(image.shape) == 2:
            # 灰度图转BGR
            annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            # 假设输入是BGR（来自OpenCV读取），直接复制
            annotated_image = image.copy()
        else:
            raise ValueError("Unsupported image format")

        # 定义红色（BGR格式）
        red_bgr = (0, 0, 255)

        # 遍历每个聚类并绘制边界框和编号
        for i, cluster in enumerate(clusters):
            x, y, w, h = cluster['bbox']
            # 绘制矩形框
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), red_bgr, 2)
            # 设置字体参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            # 计算文本位置，防止越界
            text_x = max(x, 0)
            text_y = max(y - 5, 10)
            text_position = (text_x, text_y)
            # 在框上方标注序号
            cv2.putText(annotated_image, str(i+1), text_position, font, font_scale, red_bgr, thickness)

        return annotated_image
    
    def _draw_multiple_clusters_on_image(self, image, clusters):
        """
        在图像上绘制多种颜色聚类的边界框
        
        Args:
            image: 输入图像（将保留原始背景）
            clusters: 聚类信息列表
            
        Returns:
            annotated_image: 带有边界框标记的图像，包含原始图像背景
        """
        # 确保图像为彩色格式以绘制彩色边界框
        if len(image.shape) == 2:
            annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated_image = image.copy()  # 使用 BGR 格式进行 OpenCV 绘制
        
        # 为不同颜色的聚类使用不同颜色的边界框（但都保持红色系以便醒目）
        colors_bgr = [
            (0, 0, 255),    # 红色
            (0, 165, 255),  # 橙色
            (0, 255, 255),  # 黄色
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (255, 0, 255),  # 紫色
            (128, 128, 128) # 灰色
        ]
        
        for i, cluster in enumerate(clusters):
            x, y, w, h = cluster['bbox']
            color_idx = cluster.get('color_index', 0)
            box_color = colors_bgr[color_idx % len(colors_bgr)]
            
            # 绘制矩形边界框
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), box_color, 2)
            
            # 绘制序号和颜色索引
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_position = (max(x, 0), max(y - 5, 10))
            
            # 显示聚类序号和颜色索引
            label = f"{i+1}"
            cv2.putText(annotated_image, label, text_position, font, font_scale, box_color, thickness)
        
        return annotated_image
    
    def _draw_classified_clusters_on_image(self, image, clusters):
        """
        在图像上绘制已分类的聚类边界框，显示对象类别而不是序号
        
        Args:
            image: 输入图像（将保留原始背景）
            clusters: 已分类的聚类信息列表
            
        Returns:
            annotated_image: 带有边界框标记和类别标签的图像，包含原始图像背景
        """
        # 确保图像为彩色格式以绘制彩色边界框
        if len(image.shape) == 2:
            annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated_image = image.copy()  # 使用 BGR 格式进行 OpenCV 绘制
        
        # 为不同类别的聚类使用不同颜色的边界框
        colors_bgr = [
            (0, 0, 255),    # 红色 - ghost
            (0, 165, 255),  # 橙色 - pacman
            (255, 0, 0)     # 蓝色 - other
        ]
        
        # 遍历每个聚类并绘制边界框和类别标签
        for i, cluster in enumerate(clusters):
            x, y, w, h = cluster['bbox']
            class_index = cluster.get('class_index', 2)  # 默认索引为2 (other)
            if class_index >= len(colors_bgr):
                class_index = 2  # 超出范围时也使用other的颜色
            box_color = colors_bgr[class_index]
            
            # 绘制矩形边界框
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), box_color, 2)
            
            # 绘制类别标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_position = (max(x, 0), max(y - 5, 10))
            
            # 显示对象类别
            label = cluster.get('object_class', f"obj{i+1}")
            cv2.putText(annotated_image, label, text_position, font, font_scale, box_color, thickness)
        
        return annotated_image
    
    def _generate_color_chart(self, main_colors, main_counts):
        """
        生成主要颜色的色卡
        
        Args:
            main_colors: 主要颜色数组
            main_counts: 对应的颜色出现次数
            
        Returns:
            color_chart: 色卡图像 (RGB格式)
        """
        # 创建色卡图像 (300x200 pixels)
        chart_width = 300
        chart_height = 200
        color_chart = np.ones((chart_height, chart_width, 3), dtype=np.uint8) * 255
        
        # 白色背景
        color_chart[:] = 255
        
        num_colors = len(main_colors)
        if num_colors == 0:
            return color_chart
            
        # 计算每个色块的宽度
        swatch_width = chart_width // num_colors
        swatch_height = chart_height - 40  # 留出空间显示颜色信息
        
        # 绘制每个颜色的色块
        for i, color in enumerate(main_colors):
            # 确定色块位置
            x_start = i * swatch_width
            x_end = min((i + 1) * swatch_width, chart_width)
            
            # 创建颜色区域 (RGB格式)
            if len(color) == 3:  # 彩色
                color_chart[10:10+swatch_height, x_start:x_end] = color
            else:  # 灰度
                color_chart[10:10+swatch_height, x_start:x_end] = [color[0], color[0], color[0]]
            
            # 在色块下方绘制颜色信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            text_color = (0, 0, 0)  # 黑色文字
            
            # 显示颜色索引和RGB值
            if len(color) == 3:
                color_text = f"{i+1}: ({color[0]},{color[1]},{color[2]})"
            else:
                color_text = f"{i+1}: ({color[0]})"
                
            # 计算文字位置（居中）
            ((text_width, text_height), _) = cv2.getTextSize(color_text, font, font_scale, thickness)
            text_x = x_start + (swatch_width - text_width) // 2
            text_y = 10 + swatch_height + 20
            
            # 确保文字位置在图像范围内
            text_x = max(0, min(text_x, chart_width - text_width))
            
            # 绘制文字
            cv2.putText(color_chart, color_text, (text_x, text_y), font, font_scale, text_color, thickness)
        
    def _draw_pill_positions(self, image, pill_positions):
        """
        在图像上绘制豆子位置
        
        Args:
            image: 输入图像
            pill_positions: 豆子位置列表 [(x, y), ...]
            
        Returns:
            annotated_image: 带有豆子位置标记的图像
        """
        # 复制图像以避免修改原图
        annotated_image = image.copy()
        
        # 用绿色小圆圈标记每个豆子位置
        for (x, y) in pill_positions:
            cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
        
        return annotated_image
