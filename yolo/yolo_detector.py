import cv2
import os
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="runs/detect/yolov8n_custom_training/weights/best.pt"):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 训练好的YOLO模型路径
        """
        self.model_path = model_path
        self.model = None
        self.class_names = {0: 'pacman', 1: 'ghost'}
        
    def load_model(self):
        """
        加载训练好的YOLO模型
        """
        try:
            self.model = YOLO(self.model_path)
            print(f"成功加载模型: {self.model_path}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def preprocess_frame(self, frame, target_size=(256, 256)):
        """
        预处理游戏画面帧
        
        Args:
            frame: 原始游戏画面帧
            target_size: 目标尺寸 (width, height)
            
        Returns:
            processed_frame: 处理后的帧
        """
        # 调整图像大小以匹配训练时的尺寸
        processed_frame = cv2.resize(frame, target_size)
        return processed_frame
    
    def detect_objects(self, frame, conf_threshold=0.5):
        """
        使用YOLO模型检测游戏画面中的对象
        
        Args:
            frame: 游戏画面帧
            conf_threshold: 置信度阈值
            
        Returns:
            results: 检测结果
            processed_frame: 预处理后的帧
        """
        if self.model is None:
            print("模型未加载")
            return None, None
            
        try:
            # 预处理帧
            processed_frame = self.preprocess_frame(frame)
            
            # 进行检测
            results = self.model.predict(
                source=processed_frame,
                conf=conf_threshold,
                verbose=False  # 静默模式
            )
            return results, processed_frame
        except Exception as e:
            print(f"检测过程中出错: {e}")
            return None, None
    
    def visualize_detections(self, frame, results, frame_id=None):
        """
        可视化检测结果并打印详细信息
        
        Args:
            frame: 游戏画面帧
            results: 检测结果
            frame_id: 帧ID（可选）
            
        Returns:
            annotated_frame: 带有标注的帧
        """
        # 复制帧以避免修改原始帧
        annotated_frame = frame.copy()
        frame = cv2.COLOR_BGR2RGB(annotated_frame)
        
        # 获取检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 获取类别名称
                    class_name = self.class_names.get(class_id, f"Class {class_id}")
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 打印检测结果
                if frame_id is not None:
                    print(f"帧 {frame_id} 检测结果:")
                    print(f"  检测到 {len(boxes)} 个对象")
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.class_names.get(class_id, f"Class {class_id}")
                        print(f"    {i+1}. {class_name} (置信度: {confidence:.2f})")
                else:
                    # 如果没有提供帧ID，仍然打印检测到的对象信息
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.class_names.get(class_id, f"Class {class_id}")
                        print(f"  {i+1}. {class_name} (置信度: {confidence:.2f})")
        
        return annotated_frame
    
    def save_detection_result(self, frame, results, output_dir, filename):
        """
        保存检测结果
        
        Args:
            frame: 游戏画面帧
            results: 检测结果
            output_dir: 输出目录
            filename: 文件名
            
        Returns:
            save_path: 保存路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 可视化检测结果
        annotated_frame = self.visualize_detections(frame, results)
        
        # 保存检测结果
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, annotated_frame)
        
        return save_path