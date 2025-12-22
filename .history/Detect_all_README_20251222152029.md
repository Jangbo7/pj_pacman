detect_all功能：
用于提取每一帧的游戏信息，可在每一次游戏step前调用，用于提供信息给强化学习或vlm

关于参数：
1 注意区分iter以及epoch，详见注释
2 注意args是配置类标准形式如下：（各参数意义，详见注释）
class MockArgs:
        def __init__(self):
            self.size = 256
            self.visualize_save = True
            self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"
            self.your_mission_name = "MissionName"
            
args = MockArgs()

关于其调用到的decision_utils以及info_utils:
封装在了utils_all里面。
utils_all文件的功能是存储大部分主函数相关的组件，（有需要如添加新的组件时，比如添加分数识别，可添加到utils_all中）

关于返回：
整体会得到一个大字典，如果visualize_save = True会自动保存每帧检测结果（包含图片与txt文件）到detection_results/your_mission_name中。

关于示例：
参考：detect_all_experiment.py