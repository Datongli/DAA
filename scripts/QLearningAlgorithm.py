"""
简单 Q-learning 演示（离散格点 + 感知半径），并生成 3D 动图展示本机无人机与直线入侵者轨迹。
运行：python d:\workforce\project\DAA-main\scripts\QLearningAlgorithm.py
"""
from __future__ import annotations
import random
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 环境参数
SCENE_X = 1000
SCENE_Y = 1000
SCENE_Z = 20
CELL = 10  # 网格大小（米）
NX = SCENE_X // CELL
NY = SCENE_Y // CELL
NZ = max(1, SCENE_Z // CELL)

UAV_SPEED = 10.0  # m/s -> 每步走一格
INTRUDER_SPEED = 10.0
PERCEPTION_RADIUS = 50.0  # m
SAFE_DISTANCE = 20.0  # 碰撞阈值

START_UAV = (500.0, 0, 10.0)
GOAL = (500.0, 1000.0, 10.0)

# 动作集合（以米为单位，每步等于 UAV_SPEED）
ACTIONS = [
    (0, 1*CELL, 0),    # N
    (0, -1*CELL, 0),   # S
    (1*CELL, 0, 0),    # E
    (-1*CELL, 0, 0),   # W
    (1*CELL, 1*CELL, 0),   # NE
    (-1*CELL, 1*CELL, 0),  # NW
    (1*CELL, -1*CELL, 0),  # SE
    (-1*CELL, -1*CELL, 0), # SW
    (0, 0, CELL),      # UP
    (0, 0, -CELL),     # DOWN
    (0, 0, 0),         # STAY
]
N_ACTIONS = len(ACTIONS)

# 辅助函数
def clamp_pos(pos):
    x = min(max(pos[0], 0), SCENE_X)
    y = min(max(pos[1], 0), SCENE_Y)
    z = min(max(pos[2], 0), SCENE_Z)
    return (x, y, z)

def euclid(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def discretize(pos):
    ix = int(pos[0] // CELL)
    iy = int(pos[1] // CELL)
    iz = int(pos[2] // CELL)
    # 保证索引在范围内
    ix = min(max(ix, 0), NX-1)
    iy = min(max(iy, 0), NY-1)
    iz = min(max(iz, 0), NZ-1)
    return ix, iy, iz

# 状态表示：(u_ix, u_iy, u_iz, rel_dx, rel_dy, rel_dz)
# 若未探测到入侵者，用特殊值 999 表示
SENTINEL = 999

def make_state(uav_pos, intruder_pos):
    u_ix, u_iy, u_iz = discretize(uav_pos)
    dist = euclid(uav_pos, intruder_pos)
    if dist <= PERCEPTION_RADIUS:
        # 感知到：用相对格点表示（相对位置以格点为单位，限制一定范围）
        dx = int(round((intruder_pos[0] - uav_pos[0]) / CELL))
        dy = int(round((intruder_pos[1] - uav_pos[1]) / CELL))
        dz = int(round((intruder_pos[2] - uav_pos[2]) / CELL))
        # 裁剪相对量到可表示范围，例如 [-10,10]
        lim = 10
        dx = max(min(dx, lim), -lim)
        dy = max(min(dy, lim), -lim)
        dz = max(min(dz, lim), -lim)
        return (u_ix, u_iy, u_iz, dx, dy, dz)
    else:
        return (u_ix, u_iy, u_iz, SENTINEL, SENTINEL, SENTINEL)

# Q-table 用 defaultdict 存储 numpy 数组
def make_qtable():
    return defaultdict(lambda: np.zeros(N_ACTIONS, dtype=float))

# 单步环境动态（返回 next_uav_pos, next_intruder_pos, reward, done）
def step_env(uav_pos, intruder_pos, action_idx, intruder_dir):
    # UAV 执行动作
    dx, dy, dz = ACTIONS[action_idx]
    new_uav = (uav_pos[0] + dx, uav_pos[1] + dy, uav_pos[2] + dz)
    new_uav = clamp_pos(new_uav)
    # 入侵者按固定方向直线飞（单位化方向乘速度）
    intr_dx, intr_dy = intruder_dir
    # intruder_dir 是单位方向在 xy 平面（dx, dy），z 固定为 0
    new_intr = (intruder_pos[0] + intr_dx * INTRUDER_SPEED, intruder_pos[1] + intr_dy * INTRUDER_SPEED, intruder_pos[2])
    new_intr = clamp_pos(new_intr)

    # 评分项
    prev_goal_dist = euclid(uav_pos, GOAL)
    new_goal_dist = euclid(new_uav, GOAL)
    prev_intr_dist = euclid(uav_pos, intruder_pos)
    new_intr_dist = euclid(new_uav, new_intr)

    done = False
    reward = 0.0

    # 1) 鼓励向目标靠近：按距离减少量奖励
    reward += (prev_goal_dist - new_goal_dist) * 0.5  # 缩放因子，可调

    # 2) 每步基础开销（鼓励更快到达）
    reward -= 0.1

    # 3) 在感知范围内对入侵者接近程度施加惩罚（越近惩罚越大）
    if new_intr_dist <= SAFE_DISTANCE:
        reward -= 200.0   # 严重碰撞惩罚
        done = True
    elif new_intr_dist <= PERCEPTION_RADIUS:
        # 线性惩罚：距离越小惩罚越大
        reward -= (PERCEPTION_RADIUS - new_intr_dist) / PERCEPTION_RADIUS * 10.0

    # 4) 到达目标的大幅奖励
    if new_goal_dist <= CELL:
        reward += 200.0
        done = True

    # 5) 接近场地边界轻微惩罚，防止贴边走
    margin = 2 * CELL
    if new_uav[0] < margin or new_uav[0] > SCENE_X - margin or new_uav[1] < margin or new_uav[1] > SCENE_Y - margin:
        reward -= 5.0

    # 6) 抑制剧烈垂直机动（小惩罚，鼓励平稳飞行）
    alt_change = abs(new_uav[2] - uav_pos[2])
    reward -= alt_change * 0.05

    return new_uav, new_intr, reward, done

# Q-learning 训练
def train_qlearning(n_episodes=800, max_steps=200, alpha=0.5, gamma=0.9, epsilon_start=0.9, epsilon_end=0.1):
    Q = make_qtable()
    for ep in range(n_episodes):
        uav_pos = START_UAV
        # 入侵者起点和方向：从 (500,1000,10) 向 (500,0,10) 直线飞，方向为 (0, -1)
        intruder_pos = (500.0, 1000.0, 10.0)
        intruder_dir = (0.0, -1.0)
        epsilon = max(epsilon_end, epsilon_start * (1 - ep / n_episodes))
        state = make_state(uav_pos, intruder_pos)
        ep_reward = 0.0
        for step in range(max_steps):
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))
            next_uav, next_intr, reward, done = step_env(uav_pos, intruder_pos, action, intruder_dir)
            next_state = make_state(next_uav, next_intr)
            best_next = 0.0 if len(Q[next_state]) == 0 else np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            state = next_state
            uav_pos, intruder_pos = next_uav, next_intr
            ep_reward += reward
            if done:
                break
        # 每50个episode打印一次进度和该episode回报
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"[train] ep {ep+1}/{n_episodes}, epsilon={epsilon:.3f}, ep_reward={ep_reward:.1f}")
    return Q

# 使用贪婪策略在环境中执行一次，返回轨迹
def run_episode_with_policy(Q, max_steps=400):
    uav_pos = START_UAV
    # 入侵者起点和方向：从 (500,1000,10) 向 (500,0,10) 直线飞，方向为 (0, -1)
    intruder_pos = (500.0, 1000.0, 10.0)
    intruder_dir = (0.0, -1.0)
    traj_uav = [uav_pos]
    traj_intr = [intruder_pos]
    detect_flags = [euclid(uav_pos, intruder_pos) <= PERCEPTION_RADIUS]  # 初始是否侦测到
    done = False
    for step in range(max_steps):
        state = make_state(uav_pos, intruder_pos)
        # 选择 Q 最大动作（若无则随机）
        if state in Q:
            action = int(np.argmax(Q[state]))
        else:
            action = random.randrange(N_ACTIONS)
        uav_pos, intruder_pos, reward, done = step_env(uav_pos, intruder_pos, action, intruder_dir)
        traj_uav.append(uav_pos)
        traj_intr.append(intruder_pos)
        detect_flags.append(euclid(uav_pos, intruder_pos) <= PERCEPTION_RADIUS)
        if done:
            break
    return traj_uav, traj_intr, detect_flags

# 绘图动画函数（平滑插值并展示轨迹与移动点）
def animate_trajectories(traj_uav, traj_intr, detect_flags=None, interp_per_step=8, interval=50):
    # 插值平滑
    def interp_list(pts):
        pts = np.array(pts)
        out = []
        for i in range(len(pts)-1):
            a = pts[i]
            b = pts[i+1]
            for t in np.linspace(0, 1, interp_per_step, endpoint=False):
                out.append(a*(1-t) + b*t)
        out.append(pts[-1])
        return np.array(out)

    uav_smooth = interp_list(traj_uav)
    intr_smooth = interp_list(traj_intr)

    # 将 detect_flags 扩展到插值帧数上，便于在动画中显示
    if detect_flags is None:
        detect_interp = np.zeros(max(len(uav_smooth), len(intr_smooth)), dtype=bool)
    else:
        # detect_flags 长度 = 原始步数，扩展为每段 interp_per_step 个帧
        detect_arr = np.array(detect_flags, dtype=bool)
        # 对每一段（每个原始索引）重复 interp_per_step 次（最后一个点需要单独处理）
        repeated = np.repeat(detect_arr[:-1], interp_per_step)
        detect_interp = np.concatenate([repeated, [detect_arr[-1]]])
        # 若插值长度比 detect_interp 长，pad 最后值
        if len(detect_interp) < max(len(uav_smooth), len(intr_smooth)):
            pad_len = max(len(uav_smooth), len(intr_smooth)) - len(detect_interp)
            detect_interp = np.concatenate([detect_interp, np.full(pad_len, detect_arr[-1], dtype=bool)])
        else:
            detect_interp = detect_interp[:max(len(uav_smooth), len(intr_smooth))]

    total = max(len(uav_smooth), len(intr_smooth))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 固定视图范围
    ax.set_xlim(0, SCENE_X)
    ax.set_ylim(0, SCENE_Y)
    ax.set_zlim(0, SCENE_Z)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')

    # 轨迹线和运动点
    uav_line, = ax.plot([], [], [], 'b-', lw=1, alpha=0.7, label='UAV traj')
    intr_line, = ax.plot([], [], [], 'r-', lw=1, alpha=0.7, label='Intruder traj')
    uav_point, = ax.plot([], [], [], 'bo', ms=6, label='UAV')
    intr_point, = ax.plot([], [], [], 'ro', ms=6, label='Intruder')
    # 在图中标出目标点
    goal_point, = ax.plot([GOAL[0]], [GOAL[1]], [GOAL[2]], marker='*', color='g', markersize=12, linestyle='', label='Goal')
    ax.legend()
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
    # 用于在图上显示侦测警告（右上方）
    action_text = ax.text2D(0.75, 0.90, '', transform=ax.transAxes, color='red', fontsize=12)

    def init():
        uav_line.set_data([], [])
        uav_line.set_3d_properties([])
        intr_line.set_data([], [])
        intr_line.set_3d_properties([])
        uav_point.set_data([], [])
        uav_point.set_3d_properties([])
        intr_point.set_data([], [])
        intr_point.set_3d_properties([])
        # 保证 goal_point 在 init 中也被设置（静态）
        goal_point.set_data([GOAL[0]], [GOAL[1]])
        goal_point.set_3d_properties([GOAL[2]])
        time_text.set_text('')
        action_text.set_text('')
        return uav_line, intr_line, uav_point, intr_point, goal_point, time_text, action_text

    def update(frame):
        idx = min(frame, uav_smooth.shape[0]-1)
        jdx = min(frame, intr_smooth.shape[0]-1)
        uav_line.set_data(uav_smooth[:idx+1,0], uav_smooth[:idx+1,1])
        uav_line.set_3d_properties(uav_smooth[:idx+1,2])
        intr_line.set_data(intr_smooth[:jdx+1,0], intr_smooth[:jdx+1,1])
        intr_line.set_3d_properties(intr_smooth[:jdx+1,2])
        uav_point.set_data(uav_smooth[idx:idx+1,0], uav_smooth[idx:idx+1,1])
        uav_point.set_3d_properties(uav_smooth[idx:idx+1,2])
        intr_point.set_data(intr_smooth[jdx:jdx+1,0], intr_smooth[jdx:jdx+1,1])
        intr_point.set_3d_properties(intr_smooth[jdx:jdx+1,2])
        time_text.set_text(f'frame: {frame}')
        # 若该插值帧对应为侦测到入侵者，显示警告文本
        if frame < len(detect_interp) and detect_interp[frame]:
            action_text.set_text("警告：发现入侵者")
        else:
            action_text.set_text('')
        return uav_line, intr_line, uav_point, intr_point, goal_point, time_text, action_text

    ani = FuncAnimation(fig, update, frames=total, init_func=init, blit=False, interval=interval, repeat=False)
    plt.show()

def main_train_and_show():
    print("开始训练 Q-learning ...")
    Q = train_qlearning(n_episodes=10000, max_steps=400, alpha=0.5, gamma=0.9, epsilon_start=0.9, epsilon_end=0.1)
    print("训练完成，运行一次贪婪策略并动画展示...")
    traj_uav, traj_intr, detect_flags = run_episode_with_policy(Q, max_steps=400)
    animate_trajectories(traj_uav, traj_intr, detect_flags=detect_flags, interp_per_step=10, interval=50)

if __name__ == "__main__":
    main_train_and_show()