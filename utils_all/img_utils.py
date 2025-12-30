import numpy as np
import cv2

def pad_image_to_size(image, target_size):
    """
    将图像填充到目标大小，保持原始图像居中
    
    Args:
        image: 输入图像
        target_size: 目标大小 (width, height)
        
    Returns:
        padded_image: 填充后的图像
        padding_info: 填充信息，包含左、上、右、下填充的像素数
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算需要填充的像素数
    pad_w = max(0, target_w - w)
    pad_h = max(0, target_h - h)
    
    # 计算左、右、上、下的填充量，使原始图像居中
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # 使用背景色填充（假设背景色是图像的边缘颜色）
    # 获取图像的边缘像素作为填充颜色
    if w > 0 and h > 0:
        # 使用图像四个角的平均颜色作为填充色
        corner_colors = np.array([
            image[0, 0],          # 左上角
            image[0, -1],         # 右上角
            image[-1, 0],         # 左下角
            image[-1, -1]         # 右下角
        ])
        background_color = corner_colors.mean(axis=0).astype(np.uint8)
    else:
        # 默认黑色
        background_color = [0, 0, 0]
    
    # 执行填充
    padded_image = cv2.copyMakeBorder(
        image, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, 
        value=background_color.tolist()
    )
    
    padding_info = {
        'left': pad_left,
        'top': pad_top,
        'right': pad_right,
        'bottom': pad_bottom,
        'original_width': w,
        'original_height': h
    }
    
    return padded_image, padding_info