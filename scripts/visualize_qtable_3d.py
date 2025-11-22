import numpy as np
from QLearningAlgorithm import (
    QLearningAgent, UAVCollisionEnv, evaluate, animate_episode, QTABLE_PATH
)
from typing import Optional

def main():
    agent = QLearningAgent()
    # 加载训练好的 Q 表
    if QTABLE_PATH.exists():
        agent.Q = np.load(QTABLE_PATH)
        print(f"[QTable] loaded from: {QTABLE_PATH}")
    else:
        print(f"[QTable] not found: {QTABLE_PATH}")
        return

    env = UAVCollisionEnv(seed=0)
    # 使用贪婪策略跑一次仿真并动态演示
    result = evaluate(agent, env, render=False)
    print("Final info:", result["final_info"])
    animate_episode(result["own_traj"], result["intr_traj"], interval_ms=100, save_path=None)

if __name__ == "__main__":
    main()

def animate_episode(traj_own: np.ndarray, traj_intr: np.ndarray, interval_ms: int = 100, save_path: Optional[str] = None):
    """
    动态演示一次仿真过程，镜头随着无人机移动。
    - interval_ms: 帧间隔（毫秒），100ms≈10fps
    - save_path: 可选，保存为 gif（需安装 pillow：pip install pillow）
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    own = np.asarray(traj_own)
    intr = np.asarray(traj_intr)

    if own.size == 0 or intr.size == 0:
        print("[Animate] empty trajectory, nothing to animate.")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 设置初始视角范围
    margin = 50  # 视角范围的边距
    ax.set_xlim(WORLD_XY)
    ax.set_ylim(WORLD_XY)
    ax.set_zlim(WORLD_Z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
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
        line_own.set_data([], [])
        line_own.set_3d_properties([])
        line_intr.set_data([], [])
        line_intr.set_3d_properties([])
        point_own.set_data([], [])
        point_own.set_3d_properties([])
        point_intr.set_data([], [])
        point_intr.set_3d_properties([])
        return line_own, line_intr, point_own, point_intr

    def update(i):
        # 防止索引越界
        idx = min(i, len(own) - 1)
        # 更新轨迹
        line_own.set_data(own[:idx+1, 0], own[:idx+1, 1])
        line_own.set_3d_properties(own[:idx+1, 2])
        line_intr.set_data(intr[:idx+1, 0], intr[:idx+1, 1])
        line_intr.set_3d_properties(intr[:idx+1, 2])

        # 更新当前点
        point_own.set_data([own[idx, 0]], [own[idx, 1]])
        point_own.set_3d_properties([own[idx, 2]])
        point_intr.set_data([intr[idx, 0]], [intr[idx, 1]])
        point_intr.set_3d_properties([intr[idx, 2]])

        # 动态调整视角范围，使无人机保持在镜头中央
        center_x, center_y, center_z = own[idx]
        ax.set_xlim(center_x - margin, center_x + margin)
        ax.set_ylim(center_y - margin, center_y + margin)
        ax.set_zlim(max(center_z - margin, WORLD_Z[0]), min(center_z + margin, WORLD_Z[1]))

        # 动态调整视角方向，使镜头跟随无人机
        ax.view_init(elev=20, azim=(90 - np.degrees(np.arctan2(own[idx, 1] - intr[idx, 1], own[idx, 0] - intr[idx, 0]))))

        return line_own, line_intr, point_own, point_intr

    fps = max(1, int(1000 / interval_ms)) if interval_ms > 0 else 10
    anim = FuncAnimation(fig, update, frames=len(own), init_func=init, interval=interval_ms, blit=False, repeat=False)

    if save_path:
        # 确保扩展名为 .gif
        if not str(save_path).lower().endswith('.gif'):
            save_path = str(save_path) + '.gif'
        try:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"[Animate] saved GIF: {save_path}")
        except Exception as e:
            print(f"[Animate] failed to save GIF: {e}")

    plt.tight_layout()
    plt.show()
    plt.close(fig)