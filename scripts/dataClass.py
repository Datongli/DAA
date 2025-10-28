"""
规定一些数据类
"""
from dataclasses import dataclass
import numpy as np
from enum import Enum


@dataclass
class Position:
    """ENU坐标系中的位置"""
    east: float    # 东向分量 (米)
    north: float   # 北向分量 (米)
    up: float      # 垂直分量 (米)


@dataclass
class Velocity:
    """ENU坐标系中的速度"""
    eastVelocity: float           # 东向速度 (米/秒)
    northVelocity: float          # 北向速度 (米/秒)
    upVelocity: float             # 垂直速度 (米/秒)


@dataclass
class StateEstimate:
    """状态估计（带不确定性）"""
    ID: str| int| float        # ID
    position: Position         # 位置
    velocity: Velocity         # 速度  
    covariance: np.ndarray     # 6x6协方差矩阵 (位置+速度)
    timeStamp: float           # 最后更新时间戳

@dataclass
class Attitude:
    """姿态"""
    yaw: float              # 偏航角 (弧度，逆时针为正，正北为0)
    pitch: float            # 俯仰角 (弧度，上仰为正)
    roll: float             # 横滚角 (弧度，右滚为正)
    

@dataclass
class UWBData:
    """UWB数据类"""
    timeStamp: int| float      # 时间戳(秒)
    range: float               # 距离 (米)
    azimuth: float             # 方位角 (弧度)
    elevation: float           # 俯仰角 (弧度)


@dataclass
class RadarData:
    """雷达数据类"""
    timeStamp: int| float      # 时间戳(秒)
    range: float               # 距离 (米)
    azimuth: float             # 方位角 (弧度)
    elevation: float           # 俯仰角 (弧度)


@dataclass
class TrackData:
    """Track数据类"""
    UAVID: str| int| float  # 无人机自身ID
    timeStamp: int| float   # 时间戳(秒)
    coordinate: Position    # ENU坐标系下三维位置
    velocity: Velocity      # ENU坐标系下速度
    yaw: float              # 偏航角 (弧度，逆时针为正，正北为0)
    pitch: float            # 俯仰角 (弧度，上仰为正)
    roll: float             # 横滚角 (弧度，右滚为正)
    

@dataclass
class CloudData:
    """云平台数据类"""
    UAVID: str| int| float  # 发送的无人机ID
    timeStamp: int| float   # 时间戳(秒)
    coordinate: Position    # ENU坐标系下三维位置
    velocity: Velocity      # ENU坐标系下速度


class HorizontalAction(Enum):
    """水平动作枚举"""
    NONE = 0                      # 无动作
    TURN_LEFT_SLOW = 1            # 左转  至少3度每秒
    TURN_RIGHT_SLOW = 2           # 右转  至少3度每秒
    TURN_LEFT_FAST = 3            # 左转  至少6度每秒
    TURN_RIGHT_FAST = 4           # 右转  至少6度每秒


class VerticalAction(Enum):
    """垂直动作枚举"""
    NONE = 0                 # 无动作
    HOVER = 1                # 悬停
    CLIMB = 2                # 爬升  至少5米每秒
    DESCEND = 3              # 下降  至少5米每秒


class BlendedAction:
    """融合动作类"""
    def __init__(self, horizontal: HorizontalAction, vertical: VerticalAction):
        self.horizontal = horizontal
        self.vertical = vertical
    
    def __str__(self):
        return f"H: {self.horizontal.name}, V: {self.vertical.name}"
    

@dataclass
class TRMState:
    """TRM状态表示"""
    # 水平状态变量
    horizontalRange: float          # 水平距离 r (米)
    relativeTrackAngle: float       # 相对航迹角 ψ (弧度)
    relativeBearing: float          # 相对方位角 θ (弧度)
    relativeHorizontalPosition: np.array  # 相对水平坐标
    ownHorizontalSpeed: float       # 自机水平速度 s0 (米/秒)
    intruderHorizontalSpeed: float  # 入侵机速度 s1 (米/秒)
    timeToVerticalLoss: float       # 垂直分离即将丢失时间 τv (秒)
    # 垂直状态变量
    relativeAltitude: float         # 相对高度 h (米)
    ownVerticalSpeed: float         # 自机垂直速率 dh0 (米/秒)
    intruderVerticalSpeed: float    # 入侵机垂直速率 dh1 (米/秒)
    timeToHorizontalLoss: float     # 水平分离即将丢失时间 τ (秒)


@dataclass
class Action:
    """
    动作类
    """
    east: float  # 东向动作
    north: float  # 北向动作
    up: float  # 高向动作