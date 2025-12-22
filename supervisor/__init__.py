# 导入基础计算机视觉监督器
from .cv_supervise import cv_supervisor

# 导入对象检测器
from .object_detector import ObjectDetector

# 导入药丸检测器
from .pill_detector import PillDetector

# 为了向后兼容，保留原来的名字
CVSupervisor = cv_supervisor

__all__ = [
    'CVSupervisor',
    'cv_supervisor',
    'ObjectDetector',
    'PillDetector'
]