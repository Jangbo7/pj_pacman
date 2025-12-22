import gymnasium as gym
import numpy as np
import cv2
import time
import os
from ultralytics import YOLO
from game_utils import create_pacman_environment


class GameRunner:
    def __init__(self, model_path="runs/detect/yolov8n_custom_training2/weights/best.pt"):
        """
        初始化游戏运行器
        
        Args:
            model_path: 训练好的YOLO模型路径
        """
        self.model_path = model_path
        self.model = None
        self.env = None
        
        # 类别名称映射
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
    
    def preprocess_frame(self, frame):
        """
        预处理游戏画面帧
        
        Args:
            frame: 原始游戏画面帧
            
        Returns:
            processed_frame: 处理后的帧
        """
        # 调整图像大小以匹配训练时的尺寸
        processed_frame = cv2.resize(frame, (256, 256))
        return processed_frame
    
    def detect_objects(self, frame):
        """
        使用YOLO模型检测游戏画面中的对象
        
        Args:
            frame: 游戏画面帧
            
        Returns:
            results: 检测结果
        """
        if self.model is None:
            print("模型未加载")
            return None
            
        try:
            # 预处理帧
            processed_frame = self.preprocess_frame(frame)
            
            # 进行检测
            results = self.model.predict(
                source=processed_frame,
                conf=0.5,  # 置信度阈值
                verbose=False  # 静默模式
            )
            return results, processed_frame
        except Exception as e:
            print(f"检测过程中出错: {e}")
            return None, None
    
    def visualize_detections(self, frame, results, frame_id):
        """
        可视化检测结果
        
        Args:
            frame: 游戏画面帧
            results: 检测结果
            frame_id: 帧ID
            
        Returns:
            annotated_frame: 带有标注的帧
        """
        # 复制帧以避免修改原始帧
        annotated_frame = frame.copy()
        
        # 将BGR格式转换为RGB格式
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
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
                print(f"帧 {frame_id} 检测结果:")
                print(f"  检测到 {len(boxes)} 个对象")
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names.get(class_id, f"Class {class_id}")
                    print(f"    {i+1}. {class_name} (置信度: {confidence:.2f})")
        
        return annotated_frame
    
    def run_game_with_detection(self, max_detections=5):
        """
        运行游戏并进行对象检测
        
        Args:
            max_detections: 最大检测次数
        """
        # 创建结果保存目录
        results_dir = "game_detections"
        os.makedirs(results_dir, exist_ok=True)
        
        # 初始化环境
        self.env = create_pacman_environment()
        observation, info = self.env.reset()
        
        print("开始游戏并进行对象检测...")
        print("=" * 40)
        
        detection_count = 0
        step_count = 0
        
        try:
            while detection_count < max_detections:
                # 随机选择一个动作
                action = self.env.action_space.sample()
                
                # 执行动作并获取结果
                observation, reward, terminated, truncated, info = self.env.step(action)
                step_count += 1
                
                # 每隔一定步数进行一次检测
                if step_count % 1 == 0:  # 每30步检测一次
                    detection_count += 1
                    print(f"\n--- 检测 {detection_count}/{max_detections} ---")
                    
                    # 使用YOLO模型检测对象
                    results, processed_frame = self.detect_objects(observation)
                    
                    if results is not None:
                        # 可视化检测结果
                        annotated_frame = self.visualize_detections(processed_frame, results, detection_count)
                        
                        # 保存检测结果
                        result_path = os.path.join(results_dir, f"detection_{detection_count:02d}.png")
                        cv2.imwrite(result_path, annotated_frame)
                        print(f"  检测结果已保存到: {result_path}")
                    else:
                        print("  检测失败")
                    
                    # 等待一段时间再继续
                    time.sleep(1)
                
                # 检查游戏是否结束
                if terminated or truncated:
                    print("游戏结束，重新开始...")
                    observation, info = self.env.reset()
                    step_count = 0
        
        except KeyboardInterrupt:
            print("\n游戏已被用户中断")
        finally:
            # 关闭环境
            if self.env:
                self.env.close()
            print(f"\n游戏结束！总共进行了 {detection_count} 次检测")
            print(f"检测结果保存在 {results_dir} 目录中")


def main():
    print("Pac-Man 游戏对象检测演示")
    print("=" * 40)
    
    # 初始化游戏运行器
    runner = GameRunner()
    
    # 加载模型
    if not runner.load_model():
        print("无法加载模型，退出程序")
        return
    
    # 运行游戏并进行检测
    runner.run_game_with_detection(max_detections=50)


if __name__ == "__main__":
    main()