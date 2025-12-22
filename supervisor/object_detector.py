"""
Game Object Detector for Atari games using computer vision techniques.
"""
import cv2
import numpy as np
from .detector import CVSupervisor


class ObjectDetector(CVSupervisor):
    """
    Object Detector for Atari Games
    """

    def classify_clusters_by_area(self, clusters, area_ranges=None):
        """
        Classify clusters by area size
        
        Args:
            clusters: List of cluster information
            area_ranges: Area range definitions [(min_area, max_area, class_name), ...]
                        If None, use default classification
            
        Returns:
            classified_clusters: Cluster information grouped by category
        """
        # Default area classification rules
        if area_ranges is None:
            area_ranges = [
                (0, 50, "small"),
                (50, 150, "medium"),
                (150, 500, "large")
            ]
        
        # Initialize classification results
        classified_clusters = {}
        for _, _, class_name in area_ranges:
            classified_clusters[class_name] = []
        
        # Assign categories to each cluster
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
            
            # If no category matches, assign to "default" class
            if not assigned:
                cluster['class'] = 'default'
                cluster['class_id'] = len(area_ranges)
                if 'default' not in classified_clusters:
                    classified_clusters['default'] = []
                classified_clusters['default'].append(cluster)
        
        return classified_clusters

    def classify_game_objects(self, clusters, target_colors):
        """
        Classify game objects by size
        First target color as ghost class, second target color as pacman
        
        Args:
            clusters: List of cluster information
            target_colors: Target color list used for limiting object classification
            
        Returns:
            classified_clusters: Cluster list with category information added
        """
        # Initialize all clusters as 'other' class
        for cluster in clusters:
            cluster['object_class'] = 'other'
            cluster['class_index'] = 2
            
        # For each target color, find the largest cluster with that color
        for i, target_color in enumerate(target_colors[:2]):  # Only process first two colors
            # Filter clusters by color (find clusters matching the target color)
            color_matching_clusters = [
                cluster for cluster in clusters 
                if np.array_equal(cluster['color'], target_color)
            ]
            
            # If we have matching clusters, select the largest one
            if color_matching_clusters:
                # Sort by area size to find the largest
                largest_cluster = max(color_matching_clusters, key=lambda x: x['area'])
                # Assign class based on target color index
                if i == 0:  # First color -> ghost
                    largest_cluster['object_class'] = 'ghost'
                    largest_cluster['class_index'] = 0
                elif i == 1:  # Second color -> pacman
                    largest_cluster['object_class'] = 'pacman'
                    largest_cluster['class_index'] = 1
                
        return clusters

    # def extract_multiple_colors_clusters(self, target_colors, min_area=36, max_area=400, classify_objects=False):
    
    def extract_multiple_colors_clusters(self, target_colors, min_area=36, max_area=80, classify_objects=False):
        """
        Extract and outline clusters of multiple specified colors
        
        Args:
            target_colors: Target color list, each color is (R, G, B) or grayscale value
            min_area: Minimum cluster area
            max_area: Maximum cluster area
            classify_objects: Whether to classify game objects
            
        Returns:
            annotated_image: Image with bounding boxes marked
            clusters: Cluster information list
        """
        # Preprocess image
        processed_env_img = self.preprocess_env_img()
        
        # Use original BGR image for processing
        img_bgr = processed_env_img
        
        # Store all cluster information
        all_clusters = []
        
        # Create masks and find clusters for each color
        for color_idx, target_color in enumerate(target_colors):
            # Create mask for specified color (note OpenCV uses BGR format)
            if len(img_bgr.shape) == 3 and hasattr(target_color, '__len__') and len(target_color) == 3:
                # Convert RGB color to BGR
                bgr_color = np.array([target_color[2], target_color[1], target_color[0]], dtype=np.uint8)
                mask = cv2.inRange(img_bgr, bgr_color, bgr_color)
            else:
                mask = cv2.inRange(img_bgr, target_color, target_color)
            
            # Find contours (i.e. clusters) with full hierarchy to handle objects with holes
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
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
            
            # Calculate bounding box for each contour
            for i, contour in enumerate(filtered_contours):
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cluster_info = {
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': cv2.contourArea(contour),
                    'color_index': color_idx,  # Record color index
                    'color': target_color      # Record color value
                }
                
                # Add hierarchy information if available
                if i < len(filtered_hierarchy) and filtered_hierarchy[i] is not None:
                    cluster_info['hierarchy'] = filtered_hierarchy[i]
                    
                all_clusters.append(cluster_info)
        
        # Draw bounding boxes on original preprocessed image, keeping original image background
        if classify_objects:
            # If object classification is needed, classify first then draw
            classified_clusters = self.classify_game_objects(all_clusters, target_colors)
            annotated_image = self._draw_classified_clusters_on_image(processed_env_img, classified_clusters)
        else:
            # Otherwise use the original drawing method
            annotated_image = self._draw_multiple_clusters_on_image(processed_env_img, all_clusters)
        
        # Save results if capture is enabled
        if hasattr(self.args, 'capture') and self.args.capture:
            filename = f"classified_objects_{self.epoch}_iter_{self.iter}.png" if classify_objects else f"selected_multiple_colors_clusters_{self.epoch}_iter_{self.iter}.png"
            cv2.imwrite(filename, annotated_image)
        
        # Return cluster information, if objects are classified return classified results
        if classify_objects:
            return annotated_image, classified_clusters
        else:
            return annotated_image, all_clusters

    def _draw_multiple_clusters_on_image(self, image, clusters):
        """
        Draw bounding boxes for multiple color clusters on image
        
        Args:
            image: Input image (will retain original background)
            clusters: Cluster information list
            
        Returns:
            annotated_image: Image with bounding boxes marked, containing original image background
        """
        # Ensure image is in color format to draw colored bounding boxes
        if len(image.shape) == 2:
            annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated_image = image.copy()  # Use BGR format for OpenCV drawing
        
        # Use different colors for bounding boxes of different color clusters (but keep red tone for prominence)
        colors_bgr = [
            (0, 0, 255),    # Red
            (0, 165, 255),  # Orange
            (0, 255, 255),  # Yellow
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (255, 0, 255),  # Purple
            (128, 128, 128) # Gray
        ]
        
        for i, cluster in enumerate(clusters):
            x, y, w, h = cluster['bbox']
            color_idx = cluster.get('color_index', 0)
            box_color = colors_bgr[color_idx % len(colors_bgr)]
            
            # Draw rectangular bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw serial number and color index
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_position = (max(x, 0), max(y - 5, 10))
            
            # Display cluster serial number and color index
            label = f"{i+1}"
            cv2.putText(annotated_image, label, text_position, font, font_scale, box_color, thickness)
        
        return annotated_image

    def _draw_classified_clusters_on_image(self, image, clusters):
        """
        Draw bounding boxes for classified clusters on image, showing object category instead of serial number
        
        Args:
            image: Input image (will retain original background)
            clusters: Classified cluster information list
            
        Returns:
            annotated_image: Image with bounding boxes and category labels marked, containing original image background
        """
        # Ensure image is in color format to draw colored bounding boxes
        if len(image.shape) == 2:
            annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated_image = image.copy()  # Use BGR format for OpenCV drawing
        
        # Use different colors for bounding boxes of different categories
        colors_bgr = [
            (0, 0, 255),    # Red - ghost
            (0, 165, 255),  # Orange - pacman
            (255, 0, 0)     # Blue - other
        ]
        
        # Traverse each cluster and draw bounding boxes and category labels
        for i, cluster in enumerate(clusters):
            x, y, w, h = cluster['bbox']
            class_index = cluster.get('class_index', 2)  # Default index is 2 (other)
            if class_index >= len(colors_bgr):
                class_index = 2  # Also use other's color when out of range
            box_color = colors_bgr[class_index]
            
            # Draw rectangular bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw category label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_position = (max(x, 0), max(y - 5, 10))
            
            # Display object category
            label = cluster.get('object_class', f"obj{i+1}")
            cv2.putText(annotated_image, label, text_position, font, font_scale, box_color, thickness)
        
        return annotated_image
    
    def extract_complex_objects_with_holes(self, target_colors, min_area=36, max_area=400):
        """
        提取具有内部空洞的复杂对象（如鬼魂），正确处理外轮廓和内轮廓关系
        
        Args:
            target_colors: 目标颜色列表，每个颜色为(R, G, B)或灰度值
            min_area: 最小对象面积
            max_area: 最大对象面积
            
        Returns:
            annotated_image: 带有边界框标记的图像
            objects: 复杂对象信息列表，包含外轮廓和内轮廓信息
        """
        # 预处理图像
        processed_env_img = self.preprocess_env_img()
        
        # 使用原始BGR图像进行处理
        img_bgr = processed_env_img
        
        # 存储所有复杂对象信息
        all_objects = []
        
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
            
            # 处理轮廓层级结构
            if hierarchy is not None:
                hierarchy = hierarchy[0]  # 取第一层索引
                
            # 分析轮廓层级，识别外轮廓和对应的内轮廓（孔洞）
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # 检查面积是否符合要求
                if not (min_area <= area <= max_area):
                    continue
                    
                # 检查是否为外轮廓（父轮廓为-1）
                is_outer_contour = hierarchy is None or hierarchy[i][3] == -1
                
                if is_outer_contour:
                    # 创建复杂对象信息
                    x, y, w, h = cv2.boundingRect(contour)
                    object_info = {
                        'bbox': (x, y, w, h),
                        'outer_contour': contour,
                        'inner_contours': [],  # 内轮廓（孔洞）列表
                        'area': area,
                        'color_index': color_idx,
                        'color': target_color
                    }
                    
                    # 查找与此外轮廓关联的内轮廓（孔洞）
                    if hierarchy is not None:
                        # 遍历所有轮廓，查找子轮廓
                        for j, child_contour in enumerate(contours):
                            # 检查此轮廓是否为当前外轮廓的子轮廓
                            if hierarchy[j][3] == i:  # hierarchy[j][3] 是父轮廓索引
                                child_area = cv2.contourArea(child_contour)
                                # 添加符合条件的内轮廓
                                if child_area <= area:  # 内轮廓应该比外轮廓小
                                    object_info['inner_contours'].append(child_contour)
                    
                    all_objects.append(object_info)
        
        # 在原始预处理图像上绘制边界框，保留原始图像背景
        annotated_image = self._draw_complex_objects_on_image(processed_env_img, all_objects)
        
        # 如果启用捕获功能，保存结果
        if hasattr(self.args, 'capture') and self.args.capture:
            cv2.imwrite(f"complex_objects_with_holes_{self.epoch}_iter_{self.iter}.png", annotated_image)
        
        return annotated_image, all_objects
    
    def _draw_complex_objects_on_image(self, image, objects):
        """
        在图像上绘制复杂对象（带孔洞）的边界框
        
        Args:
            image: 输入图像（将保留原始背景）
            objects: 复杂对象信息列表
            
        Returns:
            annotated_image: 带有边界框标记的图像，包含原始图像背景
        """
        # 确保图像为彩色格式以绘制彩色边界框
        if len(image.shape) == 2:
            annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated_image = image.copy()  # 使用 BGR 格式进行 OpenCV 绘制
        
        # 为不同颜色的对象使用不同颜色的边界框
        colors_bgr = [
            (0, 0, 255),    # 红色
            (0, 165, 255),  # 橙色
            (0, 255, 255),  # 黄色
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (255, 0, 255),  # 紫色
            (128, 128, 128) # 灰色
        ]
        
        # 遍历每个对象并绘制外轮廓和内轮廓
        for i, obj in enumerate(objects):
            color_idx = obj.get('color_index', 0)
            box_color = colors_bgr[color_idx % len(colors_bgr)]
            
            # 绘制外轮廓
            x, y, w, h = obj['bbox']
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), box_color, 2)
            
            # 绘制外轮廓的序号
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_position = (max(x, 0), max(y - 5, 10))
            label = f"{i+1}"
            cv2.putText(annotated_image, label, text_position, font, font_scale, box_color, thickness)
            
            # 可选：绘制内轮廓
            # for inner_contour in obj['inner_contours']:
            #     cv2.drawContours(annotated_image, [inner_contour], -1, box_color, 1)
        
        return annotated_image
