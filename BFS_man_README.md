# BFS_man.py 说明文档

## 概述

`BFS_man.py` 是一个基于 BFS（广度优先搜索）和曼哈顿距离的 Pac-Man 游戏智能决策模块。根据游戏中豆子的数量自动选择不同的寻路策略，并具备防止回头路和危险逃跑功能。

---

## 动作编号映射

| 编号 | 动作 |
|------|------|
| 0 | NOOP（静止） |
| 1 | UP（上） |
| 2 | RIGHT（右） |
| 3 | LEFT（左） |
| 4 | DOWN（下） |

---

## 类说明

### 1. `GameArgs` - 游戏配置参数类

存储游戏运行所需的配置参数。

| 属性 | 类型 | 说明 |
|------|------|------|
| `size` | int | 图片大小，固定为256 |
| `visualize_save` | bool | 是否保存可视化结果 |
| `path` | str | YOLO模型路径 |
| `your_mission_name` | str | 任务名称，用于保存结果文件夹命名 |
| `game_name` | str | 游戏名称 |
| `vlm` | str | VLM模型名称 |
| `ghost_danger_threshold` | int | Ghost危险距离阈值（曼哈顿距离） |
| `superpill_chase_threshold` | int | 大力丸追逐距离阈值，Pacman与大力丸距离小于此值时考虑追逐 |
| `superpill_safe_margin` | int | 大力丸追逐安全边际，Ghost距离需要比大力丸距离多出此值才会追逐 |
| `ghost_chase_threshold` | int | 追击Ghost距离阈值，吃掉大力丸后Ghost距离小于此值时主动追击 |

---

### 2. `GameState` - 游戏状态存储类

存储从 `detect_all_in_one` 获取的游戏状态信息，用于 BFS 路径规划和决策。

#### 类常量

| 常量 | 说明 |
|------|------|
| `OPPOSITE_DIRECTION` | 反方向映射字典，用于排除回头路 |
| `ACTION_TO_DIRECTION` | 动作编号到方向名的映射 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `pacman_boxes` | list | Pacman边界框列表 |
| `pacman_centers` | list | Pacman中心点列表 |
| `pacman_position` | tuple | 当前Pacman位置 (x, y) |
| `ghosts_boxes` | list | 所有Ghost边界框 |
| `ghosts_centers` | list | 所有Ghost中心点 |
| `four_ghosts_boxes` | list | 4个Ghost的边界框（算法使用） |
| `four_ghosts_centers` | list | 4个Ghost的中心点（算法使用） |
| `ghost_num` | int | Ghost数量 |
| `pill_centers` | list | 所有豆子中心点 |
| `pill_num` | int | 豆子数量 |
| `superpill_boxes` | list | 大力丸边界框 |
| `superpill_centers` | list | 大力丸中心点 |
| `superpill_info` | dict | 大力丸完整信息 |
| `door_centers` | list | 传送门中心点 |
| `obstacles_mask` | ndarray | 障碍物掩码（二值图像） |
| `pacman_decision` | dict | 可行动方向 `{'up': 1/0, ...}` |
| `legal_action_num` | int | 可行动方向数量 |
| `last_action` | int | 上一步执行的动作编号 |
| `last_direction` | str | 上一步的方向名 |
| `score` | int | 当前得分 |
| `HP` | int | 当前生命值 |
| `state` | str | 游戏状态: `'init'`/`'run'`/`'chase'` |
| `frame` | int | 当前帧数 |
| `epoch` | int | 游戏轮次 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `set_last_action(action)` | `action`: 动作编号 | None | 记录上一步执行的动作 |
| `get_opposite_direction()` | 无 | str/None | 获取上一步动作的反方向（需要被排除的方向） |
| `get_legal_actions_no_backtrack()` | 无 | list | 获取排除回头路后的合法动作列表 |
| `update_from_detect_all(all_game_info, frame, epoch)` | 检测信息字典, 帧数, 轮次 | None | 从detect_all_in_one的返回值更新游戏状态 |
| `get_pacman_pos()` | 无 | tuple | 获取Pacman当前位置 |
| `get_ghost_positions()` | 无 | list | 获取所有有效Ghost的位置列表 |
| `get_pill_positions()` | 无 | list | 获取所有豆子位置列表 |
| `get_superpill_positions()` | 无 | list | 获取所有大力丸位置列表 |
| `get_legal_actions()` | 无 | list | 获取当前可执行的动作列表 |
| `is_in_danger(threshold)` | `threshold`: 距离阈值 | (bool, float, tuple) | 判断Pacman是否处于危险状态，返回(是否危险, 最近距离, 最近Ghost位置) |
| `should_chase_superpill(chase_threshold, safe_margin)` | 追逐阈值, 安全边际 | (bool, tuple, float, float) | 判断是否应该追逐大力丸，返回(是否追逐, 最近大力丸位置, 大力丸距离, Ghost距离) |
| `should_chase_ghost(chase_threshold)` | 追击阈值 | (bool, tuple, float) | 判断是否应该追击Ghost（chase状态下），返回(是否追击, 最近Ghost位置, Ghost距离) |
| `check_stuck()` | 无 | (bool, int) | 检测Pacman是否卡住，返回(是否卡住, 已停留帧数) |
| `reset_stuck_detection()` | 无 | None | 重置卡住检测状态 |
| `print_state()` | 无 | None | 打印当前游戏状态（调试用） |

#### 卡住检测属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stuck_position` | tuple | None | 记录可能卡住的位置 |
| `stuck_frames` | int | 0 | 在该位置停留的帧数 |
| `stuck_threshold` | int | 30 | 判定为卡住的帧数阈值 |
| `stuck_distance` | int | 5 | 判定为同一位置的曼哈顿距离阈值 |

---

### 3. `PathFinder` - 路径规划类

根据豆子数量选择不同的寻路策略：
- **豆子 ≤ 15**：使用BFS精确搜索
- **豆子 > 15**：使用曼哈顿距离 + 障碍物感知的启发式算法

#### 类常量

| 常量 | 说明 |
|------|------|
| `DIRECTIONS` | 方向定义列表 `[(dx, dy, 方向名, 动作编号), ...]` |
| `PILL_THRESHOLD` | 豆子数量阈值，默认15 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__(game_state, search_radius)` | GameState对象, 搜索半径 | None | 初始化路径规划器 |
| `find_next_action()` | 无 | (int, tuple, str) | 根据当前游戏状态决定下一步动作，返回(动作编号, 目标位置, 策略名) |
| `_bfs_find_path(start_pos, target_positions)` | 起始位置, 目标位置列表 | (int, tuple, str) | BFS搜索最短路径到最近的豆子，会自动排除回头路 |
| `_heuristic_find_path(start_pos, target_positions)` | 起始位置, 目标位置列表 | (int, tuple, str) | 启发式算法：曼哈顿距离 + 障碍物惩罚 + Ghost惩罚 |
| `_is_valid_position(x, y, obstacles_mask)` | x坐标, y坐标, 障碍物掩码 | bool | 检查位置是否有效（在边界内且不是障碍物） |
| `_calculate_obstacle_penalty(start, target, obstacles_mask)` | 起点, 目标点, 障碍物掩码 | int | 计算从起点到目标的直线路径上的障碍物惩罚 |
| `_calculate_ghost_penalty(target)` | 目标点 | int | 计算目标点附近Ghost带来的惩罚 |
| `_select_action_towards_target(start, target, legal_actions)` | 起点, 目标点, 合法动作列表 | int | 选择朝向目标的最佳合法动作 |
| `_find_nearest_target(pos, targets)` | 当前位置, 目标列表 | tuple | 找到最近的目标点 |
| `get_strategy_name(pill_count)` | 豆子数量 | str | 根据豆子数量返回将使用的策略名称 |

---

## 函数说明

### 辅助函数

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `manhattan_distance(pos1, pos2)` | 两个坐标点 | int | 计算两点之间的曼哈顿距离 |
| `euclidean_distance(pos1, pos2)` | 两个坐标点 | float | 计算两点之间的欧几里得距离 |
| `save_stuck_detection_image(...)` | 图像, 游戏信息, 状态, 帧数, 轮次, 保存目录 | None | 保存Pacman卡住时的检测图片（含legal action箭头） |

### 动作决策函数

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `decide_next_action(game_state, args)` | GameState对象, 配置参数 | (int, tuple, str, bool) | 决定下一步动作的主函数，返回(动作编号, 目标位置, 策略名, 是否危险) |
| `_get_escape_action(game_state, ghost_pos)` | GameState对象, Ghost位置 | int | 获取逃跑动作（远离Ghost的方向） |

### 游戏控制函数

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `single_action(env, action_num, duration)` | 环境, 动作编号, 持续帧数 | (obs, reward, terminated, truncated, info) | 执行单个动作持续一定帧数 |
| `initialize_game()` | 无 | (env, args, model, game_state) | 初始化游戏环境和相关变量 |
| `update_game_state(env_img, args, epoch, frame, former_all_game_info, model, game_state)` | 当前帧图像, 配置, 轮次, 帧数, 上一帧信息, YOLO模型, 游戏状态 | dict | 调用detect_all_in_one更新游戏状态 |

---

## 策略说明

### 1. BFS精确搜索（豆子 ≤ 15）
- 使用广度优先搜索找到到最近豆子的精确最短路径
- 考虑障碍物掩码，只在可通行区域搜索
- **自动排除回头路**：如果上一步向上，本步会排除"向下"选项

### 2. 启发式搜索（豆子 > 15）
对每个豆子计算综合评分：
```
总分 = 曼哈顿距离 + 障碍物惩罚×2 + Ghost惩罚×3
```
- **障碍物惩罚**：检测直线路径上的障碍物数量
- **Ghost惩罚**：目标点30像素内的Ghost会增加惩罚
- **自动排除回头路**

### 3. 大力丸追逐策略
当满足以下条件时触发：
- Pacman与最近大力丸的距离 < `superpill_chase_threshold`（默认50）
- 最近Ghost的距离 > 大力丸距离 + `superpill_safe_margin`（默认20）

触发后，使用BFS精确搜索直接导航到大力丸位置。

### 4. 追击Ghost策略（吃掉大力丸后）
当满足以下条件时触发：
- 当前处于 `chase` 状态（已吃掉大力丸）
- 最近Ghost的距离 < `ghost_chase_threshold`（默认60）

触发后，使用BFS精确搜索主动追击最近的Ghost。

### 5. 危险逃跑模式
当Ghost距离小于阈值且不处于chase状态时触发：
- 计算远离Ghost的方向
- 优先选择距离Ghost最远的合法动作

---

## 卡住检测功能

当Pacman在同一位置停留超过阈值帧数时，系统会自动保存带有检测信息的图片用于调试。

### 检测条件
- Pacman位置与记录位置的曼哈顿距离 ≤ `stuck_distance`（默认5像素）
- 连续停留帧数 ≥ `stuck_threshold`（默认30帧）

### 保存内容
图片保存到 `stuck_detection/` 目录，包含：
- **Ghost位置**：红色边界框和中心点
- **Pacman位置**：绿色边界框和中心点  
- **大力丸**：青色圆点
- **豆子**：黄色小点
- **Legal Action箭头**：绿色箭头显示可移动方向
- **文字信息**：帧数、停留帧数、位置、合法动作、状态、得分

### 文件命名
`stuck_epoch{轮次}_frame{帧数}_{时间戳}.png`

---

## 防止回头路机制

```
示例：
上一步: UP (向上) 
    ↓
本帧合法动作: [up, down, left, right]
    ↓
排除回头路 (down)
    ↓
有效动作: [up, left, right]
```

**安全机制**：如果排除回头路后没有可选动作，会保留原始合法动作，避免卡死。

---

## 使用示例

```python
# 初始化
env, args, model, game_state = initialize_game()
observation, info = env.reset()

# 游戏循环
frame = 0
former_all_game_info = None

while True:
    # 更新游戏状态
    image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    all_game_info = update_game_state(image_bgr, args, 0, frame, former_all_game_info, model, game_state)
    
    # 决策
    action, target, strategy, is_danger = decide_next_action(game_state, args)
    
    # 记录动作（防止回头路）
    game_state.set_last_action(action)
    
    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)
    
    former_all_game_info = all_game_info
    frame += 1
```

---

## 依赖

- `gymnasium` - 游戏环境
- `ale_py` - Atari环境
- `opencv-python` - 图像处理
- `ultralytics` - YOLO模型
- `numpy` - 数值计算
- `dashscope` - VLM API（预留）
