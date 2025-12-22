detect_all：
用于提取每一帧的游戏信息，可在每一次游戏step前调用，用于提供信息给强化学习或vlm

关于参数：
1 注意区分iter以及epoch，详见注释
2 注意args是配置类标准形式如下各参数意义，详见注释：
class MockArgs:
        def __init__(self):
            self.size = 256
            self.visualize_save = True
            self.path = "runs/detect/yolov8n_custom_training2/weights/best.pt"
            self.your_mission_name = "MissionName"
            
args = MockArgs()
3 

