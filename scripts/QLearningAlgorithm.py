import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from pathlib import Path


# =========================
# 环境/模型超参数
# =========================
DT = 1.0  # s，仿真步长
WORLD_XY = (0.0, 1000.0)
WORLD_Z = (0.0, 50.0)

OWN_INIT_POS = np.array([500.0, 0.0, 25.0], dtype=float)
OWN_INIT_YAW_DEG = 90.0  # 朝向+Y
OWN_HORIZ_SPEED = 10.0  # m/s，水平恒速
OWN_ALT_SPEEDS = [2.0, 0.0, -2.0]  # 上、平、下

GOAL_POS = np.array([500.0, 1000.0, 25.0], dtype=float)
GOAL_TOL = 5.0  # m

INTR_INIT_POS = np.array([500.0, 1000.0, 25.0], dtype=float)
INTR_TERM_POS = np.array([500.0, 0.0, 25.0], dtype=float)
INTR_YAW_DEG = -90.0  # 朝向 -Y
INTR_HORIZ_SPEED = 10.0  # m/s，水平恒速
INTR_ALT_V = 0.0  # m/s

DETECT_RADIUS = 50.0  # m
FOV_DEG = 180.0  # 前向 180°
FOV_HALF = FOV_DEG / 2.0  # 90°

COLLISION_DIST = 5.0  # m

# Q-learning 超参数
GAMMA = 0.95
ALPHA = 0.2
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 300  # 线性衰减到终值
TRAIN_EPISODES = 4000
MAX_STEPS_PER_EP = 200  # 单回合最多步数

# 状态离散（相对方位角θ、绝对高度差h）
# θ 分桶: [-90,-30), [-30,0), [0,30), [30,90]
# h 分桶: [0,10), [10,20), [20,30], (30,40], (40,50]
THETA_BUCKETS = 4
H_BUCKETS = 5

# 动作集合（二维组合：水平角速度 Δψ，竖直速度 vz）
YAW_RATES = [-30.0, -10.0, 0.0, 10.0, 30.0]  # deg/s
ALT_V_OPTS = [2.0, 0.0, -2.0]  # m/s
ACTIONS: List[Tuple[float, float]] = [(yr, vz) for yr in YAW_RATES for vz in ALT_V_OPTS]
N_ACTIONS = len(ACTIONS)  # 15

# 保存/加载路径（项目根目录下 checkpoints/qtable_uav.npy）
ROOT_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT_DIR / "checkpoints"
QTABLE_PATH = CKPT_DIR / "qtable_uav.npy"


def wrap_angle_deg(a: float) -> float:
    """Wrap angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    # 把 -180 归并到 180 的等价类，便于比较
    if a <= -180.0:
        a += 360.0
    return a


def bearing_deg(vec_xy: np.ndarray) -> float:
    """返回向量在世界坐标系下的方位角（X轴为0°, 朝+Y为+90°）。"""
    return math.degrees(math.atan2(vec_xy[1], vec_xy[0]))


def angle_to_target_deg(own_yaw_deg: float, own_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """返回从自身朝向到目标方向的偏差角（目标方位 - 当前朝向），范围(-180,180]"""
    to_target = target_pos[:2] - own_pos[:2]
    if np.linalg.norm(to_target) < 1e-6:
        return 0.0
    target_bearing = bearing_deg(to_target)
    err = wrap_angle_deg(target_bearing - own_yaw_deg)
    return err


def in_front_fov(own_yaw_deg: float, own_pos: np.ndarray, intr_pos: np.ndarray, radius: float = DETECT_RADIUS) -> Tuple[bool, float, float]:
    """
    判断入侵者是否在前向FOV半圆内（水平面），且返回水平视线夹角与三维距离。
    返回: (detected, rel_theta_deg, distance_3d)
    """
    rel_vec = intr_pos - own_pos
    dist3 = float(np.linalg.norm(rel_vec))
    if dist3 > radius:
        return False, 0.0, dist3

    # 水平夹角
    rel_xy = rel_vec[:2]
    if np.linalg.norm(rel_xy) < 1e-6:
        # 水平重合，角度按0处理
        theta = 0.0
    else:
        rel_bearing = bearing_deg(rel_xy)
        theta = wrap_angle_deg(rel_bearing - own_yaw_deg)

    detected = abs(theta) <= FOV_HALF
    return detected, theta, dist3


def bin_theta(theta_deg: float) -> Optional[int]:
    """θ分桶: [-90,-30)->0, [-30,0)->1, [0,30)->2, [30,90]->3；若超出[-90,90]返回None"""
    if theta_deg < -90.0 or theta_deg > 90.0:
        return None
    if -90.0 <= theta_deg < -30.0:
        return 0
    if -30.0 <= theta_deg < 0.0:
        return 1
    if 0.0 <= theta_deg < 30.0:
        return 2
    # [30,90]
    return 3


def bin_h(h_abs: float) -> int:
    """h分桶: [0,10)->0, [10,20)->1, [20,30]->2, (30,40]->3, (40,50]->4"""
    # 先限制到[0,50]范围
    h = max(0.0, min(50.0, float(h_abs)))
    if 0.0 <= h < 10.0:
        return 0
    if 10.0 <= h < 20.0:
        return 1
    if 20.0 <= h <= 30.0:
        return 2
    if 30.0 < h <= 40.0:
        return 3
    # 40.0 < h <= 50.0
    return 4


@dataclass
class UAVState:
    pos: np.ndarray  # (x, y, z)
    yaw_deg: float   # 航向角，0沿+X，+90沿+Y


class SimpleUAV:
    def __init__(self, pos: np.ndarray, yaw_deg: float, horiz_speed: float, vz: float = 0.0):
        self.state = UAVState(pos=pos.astype(float).copy(), yaw_deg=float(yaw_deg))
        self.horiz_speed = float(horiz_speed)
        self.vz = float(vz)

    def step(self, yaw_rate_deg_s: float, vz_cmd: float, dt: float = DT):
        # 更新航向
        self.state.yaw_deg = wrap_angle_deg(self.state.yaw_deg + yaw_rate_deg_s * dt)

        # 水平移动（恒定水平速度）
        yaw_rad = math.radians(self.state.yaw_deg)
        vx = self.horiz_speed * math.cos(yaw_rad)
        vy = self.horiz_speed * math.sin(yaw_rad)

        # 竖直速度由动作直接指定
        self.vz = float(vz_cmd)

        self.state.pos[0] += vx * dt
        self.state.pos[1] += vy * dt
        self.state.pos[2] = float(np.clip(self.state.pos[2] + self.vz * dt, WORLD_Z[0], WORLD_Z[1]))

        # 软限制边界：若越界则钳制
        self.state.pos[0] = float(np.clip(self.state.pos[0], WORLD_XY[0], WORLD_XY[1]))
        self.state.pos[1] = float(np.clip(self.state.pos[1], WORLD_XY[0], WORLD_XY[1]))

    def los_yaw_error(self, target: np.ndarray) -> float:
        return angle_to_target_deg(self.state.yaw_deg, self.state.pos, target)


class QLearningAgent:
    def __init__(self):
        # Q表: [theta_bucket(4), h_bucket(5), action(15)]
        self.Q = np.zeros((THETA_BUCKETS, H_BUCKETS, N_ACTIONS), dtype=float)

    def epsilon(self, ep: int) -> float:
        if ep >= EPSILON_DECAY_EPISODES:
            return EPSILON_END
        # 线性衰减
        return EPSILON_START - (EPSILON_START - EPSILON_END) * (ep / EPSILON_DECAY_EPISODES)

    def select_action(self, state: Tuple[int, int], ep: int) -> int:
        theta_i, h_i = state
        eps = self.epsilon(ep)
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        q = self.Q[theta_i, h_i]
        # 随机打破平局
        max_q = np.max(q)
        candidates = np.flatnonzero(np.isclose(q, max_q))
        return int(np.random.choice(candidates))

    def update(self, s: Tuple[int, int], a: int, r: float, s_next: Optional[Tuple[int, int]], done: bool):
        theta_i, h_i = s
        if done or s_next is None:
            target = r
        else:
            tn, hn = s_next
            target = r + GAMMA * np.max(self.Q[tn, hn])
        self.Q[theta_i, h_i, a] = (1 - ALPHA) * self.Q[theta_i, h_i, a] + ALPHA * target


class UAVCollisionEnv:
    """
    场景：
    - 自机：起点(500,0,25)，水平恒速10m/s，朝+Y，探测半径50m、前向180°。
    - 目标： (500,1000,25)。
    - 入侵者：起点(500,1000,25)，朝-Y，水平恒速10m/s（对向直线）。
    - Q-learning 仅在探测到入侵者时启用；未探测时自机按航点直飞（限制最大转速）。
    - 注意：脱离冲突后不终止，改为继续飞向目标点（RL退出，转为自动驾驶）。
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.own = SimpleUAV(OWN_INIT_POS.copy(), OWN_INIT_YAW_DEG, OWN_HORIZ_SPEED)
        self.intr = SimpleUAV(INTR_INIT_POS.copy(), INTR_YAW_DEG, INTR_HORIZ_SPEED, INTR_ALT_V)
        self.t = 0
        self.detected_started = False  # 一旦首次探测到入侵者，则进入RL阶段
        self.prev_dist = None  # 上一步自机与入侵者的3D距离
        self.done = False
        self.info = {}
        # 返回初始RL状态（若未探测，则返回None）
        return self._get_state()

    def _get_state(self) -> Optional[Tuple[int, int]]:
        detected, theta_deg, dist3 = in_front_fov(self.own.state.yaw_deg, self.own.state.pos, self.intr.state.pos)
        if not detected:
            return None
        h_abs = abs(self.intr.state.pos[2] - self.own.state.pos[2])
        th_idx = bin_theta(theta_deg)
        if th_idx is None:
            return None
        h_idx = bin_h(h_abs)
        return (th_idx, h_idx)

    def _autopilot_to_goal(self) -> Tuple[float, float]:
        """未探测到入侵者时：朝目标点直飞（限制角速±30°/s），竖直保持高度（0）"""
        err = self.own.los_yaw_error(GOAL_POS)
        yaw_rate = float(np.clip(err, -30.0, 30.0))  # 单步按误差剪裁到最大角速
        return yaw_rate, 0.0

    def _intruder_policy(self) -> Tuple[float, float]:
        """入侵者固定对向直飞（航向保持-90°），竖直速度0"""
        return 0.0, 0.0

    def step(self, action: Optional[int], use_rl: bool) -> Tuple[Optional[Tuple[int, int]], float, bool, dict]:
        """环境推进一步。
        - action: 若 use_rl=True，传入离散动作索引；否则忽略（自驾朝目标飞）
        - use_rl: True 表示开启RL（探测到入侵者后）
        返回: (next_state, reward, done, info)
        """
        # 计算当前探测/距离
        detected_now, theta_deg, dist3 = in_front_fov(self.own.state.yaw_deg, self.own.state.pos, self.intr.state.pos)
        if self.prev_dist is None:
            self.prev_dist = dist3

        # 选择动作
        if use_rl and action is not None and detected_now:
            yaw_rate, vz = ACTIONS[action]
        else:
            yaw_rate, vz = self._autopilot_to_goal()

        # 入侵者固定策略
        intr_yaw_rate, intr_vz = self._intruder_policy()

        # 推进一步
        self.own.step(yaw_rate, vz, DT)
        self.intr.step(intr_yaw_rate, intr_vz, DT)

        self.t += 1

        # 终止检测：碰撞/到达/超时/越界
        dist_own_goal = float(np.linalg.norm(self.own.state.pos - GOAL_POS))
        dist3_new = float(np.linalg.norm(self.own.state.pos - self.intr.state.pos))
        detected_next, theta_next, _ = in_front_fov(self.own.state.yaw_deg, self.own.state.pos, self.intr.state.pos)

        collision = dist3_new <= COLLISION_DIST
        reached = dist_own_goal <= GOAL_TOL
        timeout = self.t >= MAX_STEPS_PER_EP

        # RL阶段进入/退出逻辑
        if detected_now and not self.detected_started:
            self.detected_started = True

        # RL退出条件：一旦进入过探测期，当目标不在FOV且距离>60m则认为脱离冲突
        deconflicted = self.detected_started and (not detected_next) and (dist3_new > 60.0)

        # 奖励
        reward = self._compute_reward(
            yaw_rate=yaw_rate,
            vz=vz,
            prev_dist=self.prev_dist,
            curr_dist=dist3_new,
            collision=collision,
            reached=reached,
            deconflicted=deconflicted,
            use_rl=use_rl and self.detected_started
        )

        # 一次性处理：发生脱离冲突后，退出RL阶段，避免后续重复记为“脱离”
        if deconflicted:
            self.detected_started = False

        # 更新 prev_dist
        self.prev_dist = dist3_new

        # 不再因脱离冲突而终止；仅碰撞/到达/超时终止
        done = collision or reached or timeout
        self.done = done

        # 下一个RL状态（仅在探测为真时提供）
        next_state = None
        if detected_next:
            th_i = bin_theta(theta_next)
            if th_i is not None:
                h_i = bin_h(abs(self.intr.state.pos[2] - self.own.state.pos[2]))
                next_state = (th_i, h_i)

        info = {
            "t": self.t,
            "collision": collision,
            "reached": reached,
            "timeout": timeout,
            "deconflicted": deconflicted,
            "detected": detected_next,
            "own_pos": self.own.state.pos.copy(),
            "intr_pos": self.intr.state.pos.copy(),
            "own_yaw_deg": self.own.state.yaw_deg,
            "dist_to_intr": dist3_new,
            "dist_to_goal": dist_own_goal
        }
        return next_state, reward, done, info

    def _compute_reward(self, yaw_rate: float, vz: float,
                        prev_dist: float, curr_dist: float,
                        collision: bool, reached: bool, deconflicted: bool,
                        use_rl: bool) -> float:
        # 基础步进代价，鼓励尽快解决
        r = -0.5

        if collision:
            return -100.0
        if reached:
            return 30.0
        if deconflicted:
            return 20.0

        # 仅在RL阶段进行避碰相关塑形
        if use_rl:
            # 距离增大奖励 / 减小惩罚
            if curr_dist > prev_dist + 1e-6:
                r += 1.0
            else:
                r -= 0.5

            # 距离越近惩罚越大（范围[0,50]映射到[-0.5,0]）
            r += 0.5 * (curr_dist / DETECT_RADIUS - 1.0)

            # 动作平滑惩罚（过大角速、爬升/下降略惩罚）
            r -= 0.02 * abs(yaw_rate)  # [-0.6,0]
            r -= 0.02 * abs(vz)        # {0, -0.04}

        return float(r)


def train(agent: QLearningAgent, env: UAVCollisionEnv, episodes: int = TRAIN_EPISODES, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    stats = {"collisions": 0, "reached": 0, "deconflicted": 0, "timeouts": 0}

    for ep in range(episodes):
        env.reset()
        s = env._get_state()
        done = False

        while not done:
            use_rl = s is not None  # 仅探测到时启用RL
            if use_rl:
                a = agent.select_action(s, ep)
            else:
                a = None

            s_next, r, done, info = env.step(a, use_rl=use_rl)

            if use_rl:
                agent.update(s, a, r, s_next, done)

            s = s_next

        # 统计
        if info.get("collision"):
            stats["collisions"] += 1
        if info.get("reached"):
            stats["reached"] += 1
        if info.get("deconflicted"):
            stats["deconflicted"] += 1
        if info.get("timeout"):
            stats["timeouts"] += 1

        # 简要进度
        if (ep + 1) % 50 == 0:
            eps = agent.epsilon(ep)
            print(f"[Train] ep={ep+1}/{episodes} eps={eps:.3f} stats={stats}")

    return stats

# === 新增：Q表的保存/加载辅助函数 ===
def save_qtable(agent: QLearningAgent, path: Path = QTABLE_PATH):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, agent.Q)
    print(f"[QTable] saved to: {path}")

def load_qtable_if_exists(agent: QLearningAgent, path: Path = QTABLE_PATH) -> bool:
    if path.exists():
        agent.Q = np.load(path)
        print(f"[QTable] loaded from: {path}")
        return True
    print(f"[QTable] not found, training from scratch: {path}")
    return False


def evaluate(agent: QLearningAgent, env: UAVCollisionEnv, render: bool = False) -> dict:
    env.reset()
    done = False
    s = env._get_state()

    traj_own = []
    traj_intr = []

    while not done:
        traj_own.append(env.own.state.pos.copy())
        traj_intr.append(env.intr.state.pos.copy())

        use_rl = s is not None
        if use_rl:
            # 贪婪策略
            theta_i, h_i = s
            q = agent.Q[theta_i, h_i]
            a = int(np.argmax(q))
        else:
            a = None

        s, r, done, info = env.step(a, use_rl=use_rl)

    result = {
        "final_info": info,
        "own_traj": np.array(traj_own),
        "intr_traj": np.array(traj_intr)
    }

    if render:
        try:
            import matplotlib.pyplot as plt
            own = result["own_traj"]; intr = result["intr_traj"]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(own[:, 0], own[:, 1], own[:, 2], label='UAV (own)')
            ax.plot(intr[:, 0], intr[:, 1], intr[:, 2], label='Intruder')
            ax.scatter([OWN_INIT_POS[0]], [OWN_INIT_POS[1]], [OWN_INIT_POS[2]], c='g', marker='o', s=40, label='Start')
            ax.scatter([GOAL_POS[0]], [GOAL_POS[1]], [GOAL_POS[2]], c='r', marker='*', s=80, label='Goal')
            ax.set_xlim(WORLD_XY); ax.set_ylim(WORLD_XY); ax.set_zlim(WORLD_Z)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title('Q-learning UAV Collision Avoidance')
            ax.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Render skipped: {e}")

    return result

# === 新增：动态3D动画 ===
def animate_episode(traj_own: np.ndarray, traj_intr: np.ndarray, interval_ms: int = 100, save_path: Optional[str] = None):
    """
    动态演示一次仿真过程。
    - interval_ms: 帧间隔（毫秒），100ms≈10fps
    - save_path: 可选，保存为mp4（需要本机安装 ffmpeg）
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    own = np.asarray(traj_own)
    intr = np.asarray(traj_intr)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(WORLD_XY); ax.set_ylim(WORLD_XY); ax.set_zlim(WORLD_Z)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('UAV Collision Avoidance (dynamic)')

    # 轨迹与当前点
    line_own, = ax.plot([], [], [], 'b-', lw=2, label='UAV')
    line_intr, = ax.plot([], [], [], 'r-', lw=2, label='Intruder')
    point_own, = ax.plot([], [], [], 'bo', ms=6)
    point_intr, = ax.plot([], [], [], 'ro', ms=6)

    # 起点/目标
    ax.scatter([OWN_INIT_POS[0]], [OWN_INIT_POS[1]], [OWN_INIT_POS[2]], c='g', marker='o', s=40, label='Start')
    ax.scatter([GOAL_POS[0]], [GOAL_POS[1]], [GOAL_POS[2]], c='r', marker='*', s=80, label='Goal')
    ax.legend()

    def init():
        line_own.set_data([], []); line_own.set_3d_properties([])
        line_intr.set_data([], []); line_intr.set_3d_properties([])
        point_own.set_data([], []); point_own.set_3d_properties([])
        point_intr.set_data([], []); point_intr.set_3d_properties([])
        return line_own, line_intr, point_own, point_intr

    def update(i):
        x1, y1, z1 = own[:i+1, 0], own[:i+1, 1], own[:i+1, 2]
        x2, y2, z2 = intr[:i+1, 0], intr[:i+1, 1], intr[:i+1, 2]

        line_own.set_data(x1, y1); line_own.set_3d_properties(z1)
        line_intr.set_data(x2, y2); line_intr.set_3d_properties(z2)

        point_own.set_data([x1[-1]], [y1[-1]]); point_own.set_3d_properties([z1[-1]])
        point_intr.set_data([x2[-1]], [y2[-1]]); point_intr.set_3d_properties([z2[-1]])

        ax.set_title(f'UAV Collision Avoidance (t={i*DT:.1f}s)')
        return line_own, line_intr, point_own, point_intr

    anim = FuncAnimation(fig, update, frames=len(own), init_func=init, interval=interval_ms, blit=False, repeat=False)

    if save_path:
        try:
            fps = max(1, int(1000 / interval_ms))
            anim.save(save_path, writer='ffmpeg', fps=fps)
        except Exception as e:
            print(f"视频保存失败（可能未安装ffmpeg）：{e}")

    plt.tight_layout()
    plt.show()

def main():
    agent = QLearningAgent()
    env = UAVCollisionEnv(seed=0)

    # 若存在已保存的 Q 表则加载，以便继续训练
    load_qtable_if_exists(agent, QTABLE_PATH)

    print("Start training...")
    stats = train(agent, env, episodes=TRAIN_EPISODES, seed=0)
    print("Training done. stats=", stats)

    # 训练后保存 Q 表
    save_qtable(agent, QTABLE_PATH)

    print("Evaluating greedy policy...")
    result = evaluate(agent, env, render=False)  # 不在这里静态渲染，转用动态动画
    print("Final info:", result["final_info"])

    # 动态3D演示（如需保存视频，提供 save_path='demo.mp4' 并安装 ffmpeg）
    animate_episode(result["own_traj"], result["intr_traj"], interval_ms=100, save_path=None)

    # 如需另存 Q 表副本：
    # np.save(ROOT_DIR / "checkpoints" / "qtable_uav_backup.npy", agent.Q)


if __name__ == "__main__":
    main()