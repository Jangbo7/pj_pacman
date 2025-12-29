## RL Prat
1. 第64帧才开始有loss

测试完成！结果已保存到 detection_results/lyd文件夹中。
第 63 帧，奖励: 0.00, 损失: None, 探索率: 1.0000

0: 256x256 1 ghost, 113.8ms
Speed: 2.3ms preprocess, 113.8ms inference, 21.1ms postprocess per image at shape (1, 3, 256, 256)
min_distance10.0
Caught
已处理帧 65/∞

测试完成！结果已保存到 detection_results/lyd文件夹中。
第 64 帧，奖励: 0.00, 损失: 115.9916, 探索率: 0.9950

2. rewr 始终为0；现在没有加episodes，只能训练一句游戏（4条命）
0: 256x256 1 pacman, 1 ghost, 19.0ms
Speed: 1.1ms preprocess, 19.0ms inference, 3.1ms postprocess per image at shape (1, 3, 256, 256)
min_distance11.0
已处理帧 90/∞

测试完成！结果已保存到 detection_results/lyd文件夹中。
第 89 帧，奖励: 0.00, 损失: 4.2562, 探索率: 0.8778

0: 256x256 1 pacman, 1 ghost, 18.7ms
Speed: 1.2ms preprocess, 18.7ms inference, 3.3ms postprocess per image at shape (1, 3, 256, 256)
min_distance3.1622776601683795
已处理帧 91/∞
游戏结束，重新开始... （到这就退出了）