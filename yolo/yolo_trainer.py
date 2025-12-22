import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import yaml
import random
import glob

class YoloTrainer:
    def __init__(self, model_name='yolov8n.pt'):
        """
        初始化YOLO训练器
        
        Args:
            model_name: 预训练模型名称
        """
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """
        加载预训练的YOLO模型
        """
        try:
            self.model = YOLO(self.model_name)
            print(f"成功加载模型: {self.model_name}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
        return True
    
    def train_custom_model(self, data_yaml, epochs=100, imgsz=640, batch_size=16):
        """
        训练自定义模型
        
        Args:
            data_yaml: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 输入图像尺寸
            batch_size: 批次大小
            
        Returns:
            results: 训练结果
        """
        if self.model is None:
            print("请先加载模型")
            return None
            
        try:
            # 开始训练
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                name='yolov8n_custom_training'
            )
            print("训练完成")
            return results
        except Exception as e:
            print(f"训练过程中出错: {e}")
            return None
    
    def detect_objects(self, image_path, conf_threshold=0.5):
        """
        使用训练好的模型检测图像中的对象
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            
        Returns:
            results: 检测结果
        """
        if self.model is None:
            print("请先加载模型")
            return None
            
        try:
            # 进行检测
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                save=True  # 保存检测结果
            )
            print(f"检测完成，结果已保存到 {image_path}")
            return results
        except Exception as e:
            print(f"检测过程中出错: {e}")
            return None
    
    def evaluate_model(self, data_yaml):
        """
        评估模型性能
        
        Args:
            data_yaml: 数据集配置文件路径
            
        Returns:
            metrics: 评估指标
        """
        if self.model is None:
            print("请先加载模型")
            return None
            
        try:
            # 进行验证
            metrics = self.model.val(data=data_yaml)
            print("模型评估完成")
            return metrics
        except Exception as e:
            print(f"评估过程中出错: {e}")
            return None

def create_sample_data_yaml(dataset_path, class_names):
    """
    创建示例数据集配置文件
    
    Args:
        dataset_path: 数据集根目录
        class_names: 类别名称列表
        
    Returns:
        yaml_path: 配置文件路径
    """
    yaml_content = f"""
path: {dataset_path}
train: images/train
val: images/val

names:
"""
    
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"
    
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    return yaml_path

def prepare_dataset_structure(dataset_path):
    """
    准备数据集目录结构
    
    Args:
        dataset_path: 数据集根目录
    """
    # 创建必要的目录
    dirs = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(dataset_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"创建目录: {full_path}")

def visualize_validation_results(model_path, data_yaml, num_samples=5):
    """
    在验证集上可视化检测结果
    
    Args:
        model_path: 训练好的模型路径
        data_yaml: 数据集配置文件路径
        num_samples: 可视化样本数量
    """
    # 加载训练好的模型
    model = YOLO(model_path)
    
    # 读取数据集配置
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 获取验证集图像路径
    val_images_path = os.path.join(data_config['path'], data_config['val'])
    val_images = glob.glob(os.path.join(val_images_path, "*.png")) + \
                  glob.glob(os.path.join(val_images_path, "*.jpg"))
    
    # 随机选择样本
    if len(val_images) > num_samples:
        selected_images = random.sample(val_images, num_samples)
    else:
        selected_images = val_images
    
    # 创建结果保存目录
    results_dir = "validation_visualization"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"在验证集上可视化 {len(selected_images)} 个样本的检测结果...")
    
    # 对每个选定的图像进行检测并保存结果
    for i, image_path in enumerate(selected_images):
        # 进行检测
        results = model.predict(source=image_path, conf=0.5, save=False)
        
        # 获取原始图像
        image = cv2.imread(image_path)
        
        # 绘制检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # 绘制边界框
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 获取类别名称
                    class_name = data_config['names'].get(class_id, f"Class {class_id}")
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存结果图像
        filename = os.path.basename(image_path)
        result_path = os.path.join(results_dir, f"result_{filename}")
        cv2.imwrite(result_path, image)
        print(f"保存检测结果: {result_path}")
    
    print(f"验证集检测结果已保存到 {results_dir} 目录")

def train_yolo_model():
    """
    训练YOLO模型的主函数
    """
    # 初始化训练器
    trainer = YoloTrainer('yolov8n.pt')
    
    # 加载模型
    if not trainer.load_model():
        print("模型加载失败，退出训练")
        return
    
    # 数据集配置文件路径
    dataset_yaml = "datas/yolo_data2/dataset.yaml"
    
    # 检查数据集配置文件是否存在
    if not os.path.exists(dataset_yaml):
        print(f"数据集配置文件不存在: {dataset_yaml}")
        print("请先运行数据转换脚本生成YOLO格式数据")
        return
    
    print(f"使用数据集配置文件: {dataset_yaml}")
    
    # 开始训练
    print("开始训练YOLO模型...")
    results = trainer.train_custom_model(
        data_yaml=dataset_yaml,
        epochs=80,         # 训练50轮
        imgsz=256,         # 图像尺寸与生成数据一致
        batch_size=16      # 批次大小
    )
    
    if results:
        print("模型训练成功完成!")
        model_path = "runs2/detect/yolov8n_custom_training/weights/best.pt"
        print(f"模型权重保存在: {model_path}")
        
        # 在验证集上可视化检测结果
        print("\n开始在验证集上可视化检测结果...")
        visualize_validation_results(model_path, dataset_yaml, num_samples=5)
    else:
        print("模型训练失败!")

if __name__ == "__main__":
    # 示例用法
    print("YOLOv8n 训练器")
    print("=" * 30)
    
    # 直接调用训练函数
    train_yolo_model()