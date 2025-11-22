import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class IntruderState:
    pos: np.ndarray  # (x,y,z)
    yaw_deg: float

class RandomIntruder:
    """
    随机生成会穿过无人机航线的入侵者。
    - 不依赖 QLearningAlgorithm 模块，避免循环导入。
    - API 与原先使用方式兼容：
        .state.pos  / .state.yaw_deg
        .reset(seed=None)
        .step(yaw_rate_deg_s, vz, dt)
    """
    def __init__(self,
                 horiz_speed: float = 10.0,
                 dt: float = 1.0,
                 world_xy: Tuple[float, float] = (0.0, 1000.0),
                 world_z: Tuple[float, float] = (0.0, 50.0),
                 own_init: Optional[np.ndarray] = None,
                 goal_pos: Optional[np.ndarray] = None,
                 own_horiz_speed: float = 10.0,
                 start_radius_range: Tuple[float, float] = (60.0, 100.0)):
        self.horiz_speed = float(horiz_speed)
        self.dt = float(dt)
        self.world_xy = world_xy
        self.world_z = world_z
        self.own_init = None if own_init is None else np.asarray(own_init, dtype=float).copy()
        self.goal_pos = None if goal_pos is None else np.asarray(goal_pos, dtype=float).copy()
        self.start_radius_range = start_radius_range
        self.own_horiz_speed = float(own_horiz_speed)

        self.state = IntruderState(pos=np.zeros(3), yaw_deg=0.0)
        self._start = np.zeros(3)
        self._end = np.zeros(3)
        self._vel = np.zeros(3)

        self.reset()

    def _clip_to_world(self, p: np.ndarray) -> np.ndarray:
        low = np.array([self.world_xy[0], self.world_xy[0], self.world_z[0]])
        high = np.array([self.world_xy[1], self.world_xy[1], self.world_z[1]])
        return np.clip(p, low, high)

    def _generate_path_crossing_uav(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        若提供了 own_init 和 goal_pos，则生成一条必然穿过其航线的路径：
        - 在自机航线段（20%-80%处）选冲突点 conflict_point
        - 在 conflict_point 周围选一个较远起点，使入侵者飞向并穿过 conflict_point
        """
        if self.own_init is None or self.goal_pos is None:
            # 回退到随机全场起止
            x0 = np.random.uniform(self.world_xy[0], self.world_xy[1])
            y0 = np.random.uniform(self.world_xy[0], self.world_xy[1])
            z0 = np.random.uniform(self.world_z[0], self.world_z[1])
            start = np.array([x0, y0, z0])
            x1 = np.random.uniform(self.world_xy[0], self.world_xy[1])
            y1 = np.random.uniform(self.world_xy[0], self.world_xy[1])
            z1 = np.random.uniform(self.world_z[0], self.world_z[1])
            end = np.array([x1, y1, z1])
            return self._clip_to_world(start), self._clip_to_world(end)

        path_vec = self.goal_pos - self.own_init
        # 为保证入侵者起点在无人机终点平面、终点在无人机起点平面，取冲突点为航线中点
        ratio = 0.5
        conflict = self.own_init + path_vec * ratio
        conflict[2] += np.random.uniform(-5.0, 5.0)

        # 计算无人机到冲突点所需时间，按比例设置入侵者起点半径使其在相近时间到达冲突点
        dist_uav_to_conflict = float(np.linalg.norm(conflict - self.own_init))
        # 估算入侵者起点与冲突点的半径（使 intr 到达时间与 uav 接近）
        if self.own_horiz_speed > 1e-6:
            desired_r = (self.horiz_speed / self.own_horiz_speed) * dist_uav_to_conflict
        else:
            desired_r = float((self.start_radius_range[0] + self.start_radius_range[1]) / 2.0)
        # 添加一些随机扰动并裁剪到合理范围
        r = float(np.clip(desired_r * np.random.uniform(0.8, 1.2), self.start_radius_range[0], max(self.start_radius_range[1], desired_r * 1.2)))
        ang = np.random.uniform(0.0, 2.0 * math.pi)

        # 起点位于无人机终点所在平面（y = goal_pos[1]）
        start_x = conflict[0] + r * math.cos(ang)
        start_y = float(self.goal_pos[1]) if self.goal_pos is not None else float(conflict[1] + r * math.sin(ang))
        start_z = float(conflict[2] + np.random.uniform(-10.0, 10.0))
        start = np.array([start_x, start_y, start_z])

        # 终点设为关于冲突点的对称点，确保路径穿过冲突点
        end = 2.0 * conflict - start
        return self._clip_to_world(start), self._clip_to_world(end)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(int(seed))
        self._start, self._end = self._generate_path_crossing_uav()
        self.state.pos = self._start.copy()
        dir_vec = self._end - self._start
        dist = np.linalg.norm(dir_vec)
        if dist > 1e-6:
            horiz_dir = dir_vec[:2] / np.linalg.norm(dir_vec[:2]) if np.linalg.norm(dir_vec[:2]) > 1e-6 else np.array([1.0, 0.0])
            yaw = math.degrees(math.atan2(horiz_dir[1], horiz_dir[0]))
            self.state.yaw_deg = float(yaw)
            self._vel = (dir_vec / dist) * self.horiz_speed
        else:
            self.state.yaw_deg = 0.0
            self._vel = np.zeros(3)

    def step(self, yaw_rate_deg_s: float = 0.0, vz_cmd: float = 0.0, dt: Optional[float] = None):
        """按速度前进。接口兼容原来的 intr.step(yaw_rate, vz, DT)。"""
        if dt is None:
            dt = self.dt
        # 不用根据 yaw_rate 调整航向（入侵者按直线航行），但保留参数以兼容调用
        # 更新位置
        # 仅按速度向量移动（_vel 已定向指向终点）
        remaining = self._end - self.state.pos
        if np.linalg.norm(remaining) > self.horiz_speed * dt:
            self.state.pos = self.state.pos + self._vel * dt
        else:
            # 到达终点后停在终点
            self.state.pos = self._end.copy()
        # 竖直速度由 vz_cmd 决定
        self.state.pos[2] = float(np.clip(self.state.pos[2] + vz_cmd * dt, self.world_z[0], self.world_z[1]))
        # 保证 XY 范围
        self.state.pos[0] = float(np.clip(self.state.pos[0], self.world_xy[0], self.world_xy[1]))
        self.state.pos[1] = float(np.clip(self.state.pos[1], self.world_xy[0], self.world_xy[1]))