#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Data Transformation Script

This script converts generated data to YOLO format and splits it into train/val sets.
"""

import os
import cv2
import shutil
import random
from pathlib import Path

# 配置目录路径
input_data_dir = 'datas/generated_data2'
output_data_dir = 'datas/yolo_data2'

def convert_to_yolo_format(input_dir, output_dir, train_ratio=0.8):
    """
    将生成的数据转换为YOLO格式并分割为训练集和验证集
    
    Args:
        input_dir: 输入数据目录
        output_dir: 输出数据目录
        train_ratio: 训练集占比
    """
    # 创建输出目录结构
    train_images_dir = os.path.join(output_dir, "images", "train")
    train_labels_dir = os.path.join(output_dir, "labels", "train")
    val_images_dir = os.path.join(output_dir, "images", "val")
    val_labels_dir = os.path.join(output_dir, "labels", "val")
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取所有数据文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    image_files.sort()  # 保证顺序一致
    
    # 打乱数据并分割
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    print(f"总数据量: {len(image_files)}")
    print(f"训练集: {len(train_files)}")
    print(f"验证集: {len(val_files)}")
    
    # 处理训练集
    process_dataset(input_dir, train_files, train_images_dir, train_labels_dir)
    
    # 处理验证集
    process_dataset(input_dir, val_files, val_images_dir, val_labels_dir)
    
    # 创建数据集配置文件
    create_yaml_config(output_dir)
    
    print("数据转换完成!")

def process_dataset(input_dir, file_list, images_dir, labels_dir):
    """
    处理数据集，转换为YOLO格式
    
    Args:
        input_dir: 输入目录
        file_list: 文件列表
        images_dir: 图像输出目录
        labels_dir: 标签输出目录
    """
    for image_file in file_list:
        # 获取对应的标签文件名
        label_file = image_file.replace('.png', '.txt')
        
        # 构建完整路径
        image_path = os.path.join(input_dir, image_file)
        label_path = os.path.join(input_dir, label_file)
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            continue
            
        # 加载图像以获取尺寸
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            continue
            
        height, width = image.shape[:2]
        
        # 复制图像文件
        dst_image_path = os.path.join(images_dir, image_file)
        shutil.copy2(image_path, dst_image_path)
        
        # 处理标签文件
        dst_label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            convert_label_file(label_path, dst_label_path, width, height)
        else:
            # 创建空标签文件
            with open(dst_label_path, 'w') as f:
                pass

def convert_label_file(src_label_path, dst_label_path, image_width, image_height):
    """
    转换单个标签文件为YOLO格式
    
    Args:
        src_label_path: 源标签文件路径
        dst_label_path: 目标标签文件路径
        image_width: 图像宽度
        image_height: 图像高度
    """
    with open(src_label_path, 'r') as src_file, open(dst_label_path, 'w') as dst_file:
        for line in src_file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id, x, y, w, h = map(float, parts)
            
            # 转换为YOLO格式 (归一化坐标)
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            norm_width = w / image_width
            norm_height = h / image_height
            
            # 写入YOLO格式标签
            yolo_line = f"{int(class_id)} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n"
            dst_file.write(yolo_line)

def create_yaml_config(output_dir):
    """
    创建YOLO训练所需的yaml配置文件
    
    Args:
        output_dir: 输出目录
    """
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: pacman
  1: ghost
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"配置文件已创建: {yaml_path}")

if __name__ == "__main__":
    convert_to_yolo_format(input_data_dir, output_data_dir)