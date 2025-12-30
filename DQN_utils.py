# ------ DQN utils ------
import numpy as np

def calculate_reward(all_game_info, former_all_game_info, next_pill_list, next_superpill_list, former_pill_list, former_superpill_list):
    reward = 0.0

    # 1. pacman 死亡奖励
    current_HP = all_game_info['HP']
    former_HP = former_all_game_info['HP'] if former_all_game_info else current_HP
    if current_HP < former_HP:
        reward -= 100.0  # 死亡惩罚
    
    # 2. 豆子相关奖励
    current_num_pill = len(next_pill_list)
    former_num_pill = len(former_pill_list) if former_pill_list else current_num_pill
    num_pill_eaten = former_num_pill - current_num_pill
    
    if num_pill_eaten > 0:
        reward += num_pill_eaten * 10.0  # 普通豆子奖励

    # 3. 超级豆子奖励
    current_num_superpill = len(next_superpill_list)
    former_num_superpill = len(former_superpill_list) if former_superpill_list else current_num_superpill
    num_superpill_eaten = former_num_superpill - current_num_superpill
    
    if num_superpill_eaten > 0:
        reward += num_superpill_eaten * 50.0  # 超级豆子奖励
    
    # 4. 幽灵相关奖励/惩罚
    pacman_state = all_game_info['state']
    ghost_positions = all_game_info['4ghosts_centers']
    pacman_position = all_game_info['pacman_centers'][0] if all_game_info['pacman_centers'] else (0, 0)
    
    if pacman_state == 'chase':
        # 检测是否碰到幽灵
        for ghost_pos in ghost_positions:
            distance = np.linalg.norm(np.array(ghost_pos) - np.array(pacman_position))
            if distance < 3:  
                reward += 100.0  # 碰到幽灵奖励
    elif pacman_state == 'run':
        # 检测是否即将碰到幽灵
        for ghost_pos in ghost_positions:
            distance = np.linalg.norm(np.array(ghost_pos) - np.array(pacman_position))
            if distance < 10:  
                reward -= 20.0  # 即将碰到幽灵的惩罚
            if distance < 3:  
                reward -= 200.0  # 即将碰到幽灵的惩罚
    
    # 4. 移动奖励/惩罚
    legal_actions = list(all_game_info['pacman_decision'].values())
    if legal_actions.count(1) == 0:
        reward -= 1.0  # 静止惩罚
    else:
        reward += 0.1  # 移动奖励（轻微）
    
    # 5. 时间惩罚（防止无限循环）
    reward -= 0.01  # 每帧轻微时间惩罚
    
    return reward

def normalize(value, min_value, max_value):
    """Min-Max 归一化函数"""
    return (value - min_value) / (max_value - min_value)

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

        pill_list, superpill_list                         # 所有pill和superpill只在第一帧更新，后续不再使用图像检测，而是根据main中维护的列表判断是否被吃掉
    返回：
        state: 编码后的状态向量：
        [
            pacmanx, pacmany,                                            # pacman位置
            min_d_ghost, min_d_ghost_gx, min_d_ghost_gy,                 # 最近ghost距离及位置
            min_d_pill, min_d_pill_px, min_d_pill_py,                    # 最近pill距离及位置
            min_d_superpill, min_d_superpill_px, min_d_superpill_py,     # 最近superpill距离及位置
            all_game_info['score'],                                      # 当前帧得分
            all_game_info['HP']                                          # 当前帧生命值
        ]
    """
    # 假设以下为预先定义的归一化范围（根据训练数据统计得到）
    min_x, max_x = 0, 256  # 游戏地图宽度
    min_y, max_y = 0, 256  # 游戏地图高度
    min_distance, max_distance = 0, 50  # 假设最大距离为 50
    min_score, max_score = 0, 4000  # 假设得分最大为 4000
    min_hp, max_hp = 0, 3  # 假设最大生命值为 3

    if all_game_info['pacman_centers']:
        pacmanx, pacmany = all_game_info['pacman_centers'][0]
    else:
        pacmanx, pacmany = 0, 0

    ## 最近 ghost
    if all_game_info['4ghosts_centers']:
        min_d_ghost = 999
        for gx, gy in all_game_info['4ghosts_centers']:
            d_ghost = np.linalg.norm([pacmanx - gx, pacmany - gy])
            if min_d_ghost > d_ghost:
                min_d_ghost = d_ghost
    else:
        min_d_ghost = 0

    ## 最近 pill
    if pill_list:
        min_d_pill = 999
        for px, py in pill_list:
            d_pill = np.linalg.norm([pacmanx - px, pacmany - py])
            if min_d_pill > d_pill:
                min_d_pill = d_pill
    else:
        min_d_pill = 0

    ## 最近 superpill
    if superpill_list:
        min_d_superpill = 999
        for px, py in superpill_list:
            d_superpill = np.linalg.norm([pacmanx - px, pacmany - py])
            if min_d_superpill > d_superpill:
                min_d_superpill = d_superpill
    else:
        min_d_superpill = 0

    ## Pacman 当前状态
    pacman_state = 0  # 0 = 'run', 1 = 'chase'
    if all_game_info['state'] == 'run':
        pacman_state = 0
    elif all_game_info['state'] == 'chase':
        pacman_state = 1

    ## 是否需要逃跑
    is_escape_needed = 1 if pacman_state == 0 and min_d_ghost <= 4 else 0

    ## chase 状态下的 Ghost 距离
    ghost_distance_in_chase = min_d_ghost if pacman_state == 1 else 0

    # 对连续特征进行归一化处理
    normalized_pacmanx = normalize(pacmanx, min_x, max_x)
    normalized_pacmany = normalize(pacmany, min_y, max_y)
    normalized_min_d_ghost = normalize(min_d_ghost, min_distance, max_distance)
    normalized_min_d_pill = normalize(min_d_pill, min_distance, max_distance)
    normalized_min_d_superpill = normalize(min_d_superpill, min_distance, max_distance)
    normalized_score = normalize(all_game_info['score'], min_score, max_score)
    normalized_hp = normalize(all_game_info['HP'], min_hp, max_hp)

    # 编码状态向量
    state = [
        normalized_pacmanx, normalized_pacmany,
        normalized_min_d_ghost, 
        normalized_min_d_pill, 
        normalized_min_d_superpill, 
        normalized_score,
        normalized_hp,
        pacman_state,               # 当前游戏状态 (run/chase)
        is_escape_needed,           # 是否需要逃跑
        ghost_distance_in_chase     # 追击状态下的 Ghost 距离
    ]
    
    return np.array(state)

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
    # 静止是always legal的action
    legal_actions_mask[4] = 1
    decisions = list(all_game_info['pacman_decision'].values())
    for enum, mask in enumerate(decisions):
        if mask > 0.5:
            legal_actions_mask[enum] = 1
    return legal_actions_mask