import numpy as np

def calculate_reward(all_game_info, former_all_game_info, next_pill_list, next_superpill_list,
                     former_pill_list, former_superpill_list, action=None, state=None, next_state=None):
    reward = 0.0    

    current_pill_list = len(next_pill_list)
    former_pill_list = len(former_pill_list)
    reward += 2 * (former_pill_list - current_pill_list)

    current_superpill_list = len(next_superpill_list)
    former_superpill_list = len(former_superpill_list)
    reward += 2 * (former_superpill_list - current_superpill_list)

    

    if state is not None and next_state is not None:
        # 3) 接近豆子进度奖励
        pill_progress = float(state[2] - next_state[2])
        reward += 10.0 * pill_progress

        # 超级豆子进度奖励
        superpill_progress = float(state[8] - next_state[8])
        reward += 20.0 * superpill_progress

        # 贴近超级豆子时给一个小的终点奖励（让“最后一步”更值）
        if float(next_state[8]) < 0.06:
            reward += 0.5

        # 4) 躲避 ghost（结合状态，且在快吃到大力丸时减弱惩罚）
        dG_after = float(next_state[5])
        dS_after = float(next_state[8])
        deltaG = float(state[5] - next_state[5])

        mode = int(state[11])  # 0=run/init, 1=chase
        if mode == 0:
            ghost_penalty_scale = 0.6 if dS_after > 0.10 else 0.2
            reward -= ghost_penalty_scale * (1.0 - dG_after)
        else:
            reward += 0.2 * deltaG

        # 5) 卡住惩罚（阈值放宽，避免检测抖动绕过）
        moved = abs(float(state[0] - next_state[0])) + abs(float(state[1] - next_state[1]))
        if moved < 2e-3:
            reward -= 0.3

    # 4) NOOP + 时间惩罚
    if action is not None and action == 0:
        reward -= 0.05
    reward -= 0.02

    return reward



def normalize(value: float, min_value: float, max_value: float) -> float:
    """Min-Max 归一化函数（带 clip，避免 >1 或 <0）。"""
    if max_value == min_value:
        return 0.0
    v = (value - min_value) / (max_value - min_value)
    return float(np.clip(v, 0.0, 1.0))


def normalize_signed(value: float, max_abs: float) -> float:
    """归一化到 [-1, 1] 并 clip。"""
    if max_abs <= 0:
        return 0.0
    return float(np.clip(value / max_abs, -1.0, 1.0))


def _nearest_point_and_distance(pacman_xy, points, max_distance: float):
    """
    返回 (nearest_point(x,y) or None, min_dist)
    若 points 为空，返回 (None, max_distance)
    """
    if not points:
        return None, float(max_distance)
    pts = np.asarray(points, dtype=np.float32)
    p = np.asarray(pacman_xy, dtype=np.float32)
    dists = np.linalg.norm(pts - p, axis=1)
    idx = int(np.argmin(dists))
    nearest = (float(pts[idx, 0]), float(pts[idx, 1]))
    d = float(dists[idx])
    return nearest, float(min(d, max_distance))



def encode_state(all_game_info, pill_list, superpill_list) -> np.ndarray:
    """
    将游戏信息编码为特征状态向量
    输入：
        all_game_info，包含当前帧所有游戏元素信息的字典，格式如下：
        {
            'pacman_boxes': [[x1, y1, x2, y2], ...],      # pacman边界框(左x,右x,上y,下y)
            'pacman_centers': [[x, y], ...],              # pacman中心点列表
            'ghosts_boxes': [[x1, y1, x2, y2], ...],      # 所有ghosts边界框
            'ghosts_centers': [[x, y], ...],              # 所有ghosts中心点
            'pill_centers': [[x, y], ...],                # pills中心点列表
            'pill_num': [n],                              # pills数量
            'superpill_boxes': [[x1, y1, x2, y2], ...],   # superpills边界框
            'superpill_centers': [[x, y], ...],           # superpills中心点
            'door_centers': [[x, y], ...],                # doors中心点(上下传送门)
            'obstacles_mask': mask,                       # 障碍物掩码(只有在第一帧会更新）
            'pacman_decision': {directions},              # 当前帧pacman可行动方向(合法action空间)
            'ghost_num': n,                               # ghosts数量
            'score': n,                                   # 当前帧得分
            'HP': n,                                      # 当前帧pacman生命值
            'state': 'init' or 'run' or 'chase',  # 当前pacman游戏状态 
        },

    输出：  
        最简 12 维状态（float32），兼顾吃豆子 + 躲 ghost + 吃超级豆子 + 游戏状态：
        [0]  pac_x_norm
        [1]  pac_y_norm

        [2]  d_pill_norm
        [3]  pill_dx_signed
        [4]  pill_dy_signed

        [5]  d_ghost_norm
        [6]  ghost_dx_signed
        [7]  ghost_dy_signed

        [8]  d_super_norm
        [9]  super_dx_signed
        [10] super_dy_signed

        [11] pacman_state
    """
    min_x, max_x = 0, 256
    min_y, max_y = 0, 256
    min_distance, max_distance = 0, 50

    # pacman 坐标
    if all_game_info.get('pacman_centers'):
        pacmanx, pacmany = all_game_info['pacman_centers'][0]
    else:
        pacmanx, pacmany = 0.0, 0.0

    pacman_xy = (pacmanx, pacmany)

    # 最近 ghost / pill / super
    nearest_ghost, min_d_ghost = _nearest_point_and_distance(
        pacman_xy, all_game_info.get('4ghosts_centers', []), max_distance
    )
    nearest_pill, min_d_pill = _nearest_point_and_distance(
        pacman_xy, pill_list or [], max_distance
    )
    nearest_super, min_d_super = _nearest_point_and_distance(
        pacman_xy, superpill_list or [], max_distance
    )

    # 归一化
    pac_x = normalize(pacmanx, min_x, max_x)
    pac_y = normalize(pacmany, min_y, max_y)

    d_pill = normalize(min_d_pill, min_distance, max_distance)
    d_ghost = normalize(min_d_ghost, min_distance, max_distance)
    d_super = normalize(min_d_super, min_distance, max_distance)

    pill_dx = pill_dy = 0.0
    ghost_dx = ghost_dy = 0.0
    super_dx = super_dy = 0.0

    if nearest_pill is not None:
        pill_dx = normalize_signed(nearest_pill[0] - pacmanx, max_x)
        pill_dy = normalize_signed(nearest_pill[1] - pacmany, max_y)

    if nearest_ghost is not None:
        ghost_dx = normalize_signed(nearest_ghost[0] - pacmanx, max_x)
        ghost_dy = normalize_signed(nearest_ghost[1] - pacmany, max_y)

    if nearest_super is not None:
        super_dx = normalize_signed(nearest_super[0] - pacmanx, max_x)
        super_dy = normalize_signed(nearest_super[1] - pacmany, max_y)

    # pacman 状态
    pacman_state = 0
    if all_game_info.get('state') == 'init' or all_game_info.get('state') == 'run':
        pacman_state = 0
    elif all_game_info.get('state') == 'chase':
        pacman_state = 1
    
    state = [
        pac_x, pac_y,
        d_pill, pill_dx, pill_dy,
        d_ghost, ghost_dx, ghost_dy,
        d_super, super_dx, super_dy,
        pacman_state
    ]
    return np.array(state, dtype=np.float32)



def get_legal_actions_mask(all_game_info) -> np.ndarray:
    """
    获取当前帧pacman可行动方向的掩码
    输入：
        all_game_info，包含当前帧所有游戏元素信息的字典，格式如下：
        {
            'pacman_decision': {directions_mask},              # 当前帧pacman可行动方向(合法action空间)
        }
    返回：
        legal_actions_mask: 可行动方向的掩码向量：
        [
           0/1, 0/1, 0/1, 0/1, 0/1
        ]
    """
    legal_actions_mask = np.zeros(5, dtype=bool)
    legal_actions_mask[0] = 1  # stay always legal
    # decision_misorder: up, down, left, right
    decisions_misorder = list(all_game_info['pacman_decision'].values())
    # decision: up, right, left, down
    decisions = [0] * 4
    decisions[0] = decisions_misorder[0]
    decisions[1] = decisions_misorder[3]
    decisions[2] = decisions_misorder[2]
    decisions[3] = decisions_misorder[1]
    for enum, mask in enumerate(decisions):
        if mask > 0.5:
            legal_actions_mask[enum + 1] = 1
    return legal_actions_mask
