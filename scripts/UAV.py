"""
定义UAV类
"""
from sensor import *
from STMandTRM import STM, TRM, TrackFile
import math
import numpy as np


class UAV:
    def __init__(self, cfg) -> None:
        """
        初始化无人机
        :param cfg: 配置文件
        """
        self.cfg = cfg
        self.ID: float| int| str = self.cfg.id  # 无人机ID
        self.ownState: StateEstimate | None = None  # 无人机自身状态估计（位置和速度）
        self.targets: Dict[str: StateEstimate] = {}  # 目标状态估计字典
        self.sensors: Dict[str: Sensor] = {}  # 传感器字典
        self.trackFiles: Dict[str: TrackFile] = {}  # 跟踪文件字典
        self._init_sensors()
        self.stm: STM = STM(self.ID, self.sensors)  # 状态管理模块
        self.trm: TRM = TRM(self.cfg)  # 跟踪管理模块
        self.simulateSensors: bool = getattr(self.cfg, "simulateSensors", True)  # 是否模拟传感器数据
        self.actions: BlendedAction | None = None  # 无人机动作（ blended action）
            
    def update(self, data: Dict) -> np.array:
        """
        UAV更新逻辑
        :param data: 传感器数据
        """
        """更新传感器数据"""
        # 先得到轨迹数据
        trackData = data.get("Track")
        if trackData:
            self.sensors["Track"].get_observations(trackData)
        # 如果存在云数据，则更新云、radar、UWB、track数据
        cloudData = data.get("Cloud")  
        if cloudData:
            self.sensors["Cloud"].get_observations(cloudData)
            if self.simulateSensors:
                # 生成模拟数据
                simSensorData = self._simulate_sensor_measurements(cloudData)
                for sensorName, dataList in simSensorData.items():
                    if sensorName in self.sensors:
                        self.sensors[sensorName].get_observations(dataList)
                    else:
                        raise ValueError(f"未知传感器类型: {sensorName}")
        """STM模块逻辑更新"""
        self.ownState, self.targets, self.trackFiles = self.stm.update()
        """TRM模块逻辑更新"""
        self.actions, riskProfile = self.trm.resolve_threat(self.ownState, self.targets)
        return self.actions, riskProfile, self.trackFiles
    
    def _init_sensors(self) -> None:
        """
        初始化传感器
        """
        for sensor in self.cfg.sensors:
            sensorCls = SENSOR_REGISTRY.get(sensor)
            if sensorCls:
                self.sensors[sensor] = sensorCls(self.cfg)
            else:
                raise ValueError(f"未知传感器类型: {sensor}")
            
    def _simulate_sensor_measurements(self, cloudData: Dict) -> Dict[str, list]:
        """
        基于上一时刻自身状态与上一时刻动作，推演当前时刻自身真实状态；再用该状态与 Cloud 入侵者真值生成 Radar/UWB 的量测。
        返回当前时刻的 Track、Radar、UWB 三类数据（各为单条记录的字典）。
        :param cloudData: 云数据
        :return: 模拟传感器数据字典
        """
        # 若 cloudData 为列表/元组，则取第一条记录
        if isinstance(cloudData, (list, tuple)) and cloudData:
            cloudData = cloudData[0]
        """读取Cloud入侵者真值（ENU）"""
        intruderId = self.getAttr(cloudData, "UAVID")  # 获取入侵者ID
        timeStamp = self.getAttr(cloudData, "timeStamp", default=None)  # 获取时间戳
        cloudCoordinate = self.getAttr(cloudData, "coordinate", default={})  # 获取东北高坐标
        cloudVelocity = self.getAttr(cloudData, "velocity", default={})  # 获取东北高速度
        targetEast = float(self.readNumber(cloudCoordinate, "east", "E", "e", default=0.0))  # 获取东坐标
        targetNorth = float(self.readNumber(cloudCoordinate, "north", "N", "n", default=0.0))  # 获取北坐标
        targetUp = float(self.readNumber(cloudCoordinate, "up", "U", "u", default=0.0))  # 获取高坐标
        """计算当前时刻自身真实状态"""
        dt = timeStamp - self.ownState.timeStamp if self.ownState else 1.0  # 计算时间间隔
        gravity = 9.81  # 重力加速度
        # 获取上一时刻自身状态（位置和速度）
        ownStatePrev = self.ownState if self.ownState else None  
        if ownStatePrev is None:
            # 若首次为空，初始化为0，确保计算链路可运行
            ownEastPrev = ownNorthPrev = ownUpPrev = 0.0
            velEastPrev = velNorthPrev = velUpPrev = 0.0
        else:
            ownCoordPrev = self.getAttr(ownStatePrev, "position", default=None)  # 获取上一时刻自身位置
            ownVelPrev = self.getAttr(ownStatePrev, "velocity", default=None)  # 获取上一时刻自身速度
            if ownCoordPrev is None:
                # 若上一时刻自身位置为空，初始化为0，确保计算链路可运行
                ownEastPrev = float(self.readNumber(ownStatePrev, "east", "E", "e", default=0.0))
                ownNorthPrev = float(self.readNumber(ownStatePrev, "north", "N", "n", default=0.0))
                ownUpPrev = float(self.readNumber(ownStatePrev, "up", "U", "u", default=0.0))
            else:
                ownEastPrev = float(self.readNumber(ownCoordPrev, "east", "E", "e", default=0.0))
                ownNorthPrev = float(self.readNumber(ownCoordPrev, "north", "N", "n", default=0.0))
                ownUpPrev = float(self.readNumber(ownCoordPrev, "up", "U", "u", default=0.0))
            if ownVelPrev is None:
                # 若上一时刻自身速度为空，初始化为0，确保计算链路可运行
                velEastPrev = float(self.readNumber(ownStatePrev, "eastVelocity", "ve", "vx", default=0.0))
                velNorthPrev = float(self.readNumber(ownStatePrev, "northVelocity", "vn", "vy", default=0.0))
                velUpPrev = float(self.readNumber(ownStatePrev, "upVelocity", "vu", "vz", default=0.0))
            else:
                velEastPrev = float(self.readNumber(ownVelPrev, "eastVelocity", "ve", "vx", default=0.0))
                velNorthPrev = float(self.readNumber(ownVelPrev, "northVelocity", "vn", "vy", default=0.0))
                velUpPrev = float(self.readNumber(ownVelPrev, "upVelocity", "vu", "vz", default=0.0))
        # 获取上一时刻采取的动作
        horizontalActionObj = self.getAttr(self, "actions", default=None)  # 提取水平动作对象
        horizontalActionObj = self.getAttr(horizontalActionObj, "horizontal", default=None)  # 获取水平动作
        horizontalName = self.readActionName(horizontalActionObj)  # 提取水平动作名称
        verticalActionObj = self.getAttr(self, "actions", default=None)  # 提取垂直动作对象
        verticalActionObj = self.getAttr(verticalActionObj, "vertical", default=None)  # 获取垂直动作
        verticalName = self.readActionName(verticalActionObj)  # 提取垂直动作名称
        deltaYawDeg = self._calculate_yaw_angle(horizontalName)  # 计算偏航角变化
        deltaVelUp = self._calculate_vel_up(verticalName)  # 计算垂直速度变化
        # 保持水平速标不变，旋转水平速度矢量
        horizontalSpeedPrev = math.hypot(velEastPrev, velNorthPrev)
        if horizontalSpeedPrev > 1e-6 and abs(deltaYawDeg) > 0.0:
            deltaYawRad = self.deg2rad(deltaYawDeg)  # 左转为正，右转为负
            cosD, sinD = math.cos(deltaYawRad), math.sin(deltaYawRad)
            velEastCurr = cosD * velEastPrev - sinD * velNorthPrev  # 当前东方向速度
            velNorthCurr = sinD * velEastPrev + cosD * velNorthPrev  # 当前北方向速度
        else:
            velEastCurr, velNorthCurr = velEastPrev, velNorthPrev
        # 垂直速度更新
        velUpCurr = velUpPrev + deltaVelUp
        # 位置积分
        ownEastCurr = ownEastPrev + velEastCurr * dt  # 当前东坐标
        ownNorthCurr = ownNorthPrev + velNorthCurr * dt  # 当前北坐标
        ownUpCurr = ownUpPrev + velUpCurr * dt  # 当前高坐标
        # 姿态更新：yaw/pitch 来自速度指向，roll 采用协调转弯近似
        ownAttitudePrev = getattr(self.stm, "ownAttitude", None)
        yawPrev = float(self.readNumber(ownAttitudePrev, "yaw", "Yaw", default=0.0)) if ownAttitudePrev is not None else 0.0
        pitchPrev = float(self.readNumber(ownAttitudePrev, "pitch", "Pitch", default=0.0)) if ownAttitudePrev is not None else 0.0
        rollPrev = float(self.readNumber(ownAttitudePrev, "roll", "Roll", default=0.0)) if ownAttitudePrev is not None else 0.0
        horizontalSpeedCurr = math.hypot(velEastCurr, velNorthCurr)
        yawCurr = math.atan2(velNorthCurr, velEastCurr) if horizontalSpeedCurr > 1e-6 else yawPrev
        pitchCurr = math.atan2(velUpCurr, max(horizontalSpeedCurr, 1e-3))
        yawRate = self.deg2rad(deltaYawDeg) / dt
        rollCurr = math.atan2(horizontalSpeedCurr * yawRate, gravity) if abs(yawRate) > 1e-6 else rollPrev
        """生成模拟数据"""
        # 生成Track数据
        trackMsg = {
            "UAVID": self.ID,
            "timeStamp": timeStamp,
            "coordinate": {"east": ownEastCurr, "north": ownNorthCurr, "up": ownUpCurr},
            "velocity": {"eastVelocity": velEastCurr, "northVelocity": velNorthCurr, "upVelocity": velUpCurr},
            "yaw": yawCurr,
            "pitch": pitchCurr,
            "roll": rollCurr,
        }
        # 量测模型（Radar/UWB）
        # 入侵者相对位置（ENU）
        relativeEnu = np.array([
            targetEast - ownEastCurr,
            targetNorth - ownNorthCurr,
            targetUp - ownUpCurr
        ], dtype=float)
        # ENU → 机体坐标（x前 y右 z上）
        # 机体到世界的旋转为 Rz(yaw) * Ry(pitch) * Rx(roll)，世界到机体取转置
        rotBodyWorld = self.rotZ(yawCurr) @ self.rotY(pitchCurr) @ self.rotX(rollCurr)
        relativeBody = rotBodyWorld.T @ relativeEnu
        rangeTrue = float(np.linalg.norm(relativeBody))
        xyNorm = max(1e-9, math.hypot(relativeBody[0], relativeBody[1]))
        azimuthTrueRad = math.atan2(relativeBody[1], relativeBody[0])   # 方位角（机体坐标，弧度）
        elevationTrueRad = math.atan2(relativeBody[2], xyNorm)          # 俯仰角（机体坐标，弧度）
        # 读取传感器精度/噪声配置（缺失直接用默认值）
        radarCfg = self.getAttr(self.cfg, "Radar", None)
        uwbCfg = self.getAttr(self.cfg, "UWB", None)
        radarRangeAcc = self.readNumber(radarCfg, "rangeAcc", default=0.1)
        radarRangeStd = self.readNumber(radarCfg, "rangeStd", default=0.5)
        radarAzimuthAcc = self.readNumber(radarCfg, "azimuthAcc", default=1.0)
        radarAzimuthStd = self.readNumber(radarCfg, "azimuthStd", default=0.05)
        radarElevationAcc = self.readNumber(radarCfg, "elevationAcc", default=2.0)
        radarElevationStd = self.readNumber(radarCfg, "elevationStd", default=0.05)
        uwbRangeAcc = self.readNumber(uwbCfg, "rangeAcc", default=0.1)
        uwbRangeStd = self.readNumber(uwbCfg, "rangeStd", default=0.3)
        uwbAzimuthAcc = self.readNumber(uwbCfg, "azimuthAcc", default=1.0)
        uwbAzimuthStd = self.readNumber(uwbCfg, "azimuthStd", default=0.08)
        uwbElevationAcc = self.readNumber(uwbCfg, "elevationAcc", default=2.0)
        uwbElevationStd = self.readNumber(uwbCfg, "elevationStd", default=0.08)
        # 叠加噪声（方位角/俯仰角输出为度）
        radarRange = self.noisy(rangeTrue, radarRangeAcc, radarRangeStd)
        radarAzimuth = self.noisy(self.rad2deg(azimuthTrueRad), radarAzimuthAcc, radarAzimuthStd)
        radarElevation = self.noisy(self.rad2deg(elevationTrueRad), radarElevationAcc, radarElevationStd)
        uwbRange = self.noisy(rangeTrue, uwbRangeAcc, uwbRangeStd)
        uwbAzimuth = self.noisy(self.rad2deg(azimuthTrueRad), uwbAzimuthAcc, uwbAzimuthStd)
        uwbElevation = self.noisy(self.rad2deg(elevationTrueRad), uwbElevationAcc, uwbElevationStd)
        # 生成radar数据
        radarMsg = {
            # "UAVID": self.ID,
            "timeStamp": timeStamp,
            # "targetID": intruderId,
            "range": radarRange,
            "azimuth": radarAzimuth,
            "elevation": radarElevation,
        }
        # 生成uwb数据
        uwbMsg = {
            # "UAVID": self.ID,
            "timeStamp": timeStamp,
            # "targetID": intruderId,
            "range": uwbRange,
            "azimuth": uwbAzimuth,
            "elevation": uwbElevation,
        }
        return {"Track": [trackMsg], "Radar": [radarMsg], "UWB": [uwbMsg]}

    @staticmethod
    def getAttr(obj: object, *names, default=None) -> any:
        """
        从对象或字典中按候选字段名顺序读取属性/键，若均不存在则返回默认值
        :param obj: 对象
        :param names: 属性名列表
        :param default: 默认值
        :return: 属性值
        """
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj:
                return obj[name]
        return default
    
    @staticmethod
    def rotZ(yawRad: float) -> np.ndarray:
        """
        绕 z 轴的旋转矩阵（右手系），角度单位：弧度。
        :param yawRad: 旋转角度（弧度）
        :return: 旋转矩阵
        """
        cosYaw, sinYaw = math.cos(yawRad), math.sin(yawRad)
        return np.array([[cosYaw, -sinYaw, 0.0],
                            [sinYaw,  cosYaw, 0.0],
                            [0.0,     0.0,    1.0]], dtype=float)
    
    @staticmethod
    def rotY(pitchRad: float) -> np.ndarray:
        """
        绕 y 轴的旋转矩阵（右手系），角度单位：弧度。
        :param pitchRad: 旋转角度（弧度）
        :return: 旋转矩阵
        """
        cosPitch, sinPitch = math.cos(pitchRad), math.sin(pitchRad)
        return np.array([[ cosPitch, 0.0, sinPitch],
                            [ 0.0,      1.0, 0.0     ],
                            [-sinPitch, 0.0, cosPitch]], dtype=float)
    
    @staticmethod
    def rotX(rollRad: float) -> np.ndarray:
        """
        绕 x 轴的旋转矩阵（右手系），角度单位：弧度。
        :param rollRad: 旋转角度（弧度）
        :return: 旋转矩阵
        """
        cosRoll, sinRoll = math.cos(rollRad), math.sin(rollRad)
        return np.array([[1.0, 0.0,     0.0    ],
                            [0.0, cosRoll, -sinRoll],
                            [0.0, sinRoll,  cosRoll]], dtype=float)
    
    @staticmethod
    def deg2rad(deg: float) -> float:
        """
        角度转弧度
        :param deg: 角度
        :return: 弧度
        """
        return deg * math.pi / 180.0

    @staticmethod
    def rad2deg(rad: float) -> float:
        """
        弧度转角度
        :param rad: 弧度
        :return: 角度
        """
        return rad * 180.0 / math.pi

    @staticmethod
    def readNumber(container: any, *keys, default=0.0) -> float:
        """
        从 dict 或对象属性中读取数值，若缺失则返回默认值。
        :param container: 容器（dict 或对象）
        :param keys: 键名列表（按顺序查找）
        :param default: 默认值
        :return: 数值
        """
        if isinstance(container, dict):
            for k in keys:
                if k in container:
                    return float(container[k])
        if hasattr(container, "__dict__"):
            for k in keys:
                if hasattr(container, k):
                    return float(getattr(container, k))
        return float(default)
    
    @staticmethod
    def readActionName(actionObj: any) -> str:
        """
        提取动作名称并统一为大写字符串
        :param actionObj: 动作对象
        :return: 动作名称（大写）
        """
        if actionObj is None:
            return "NONE"
        if isinstance(actionObj, str):
            return actionObj.upper()
        name = UAV.getAttr(actionObj, "name", default=None)
        if name:
            return str(name).upper()
        # Enum 或复合对象：尝试常见字段
        for key in ("horizontal", "horizontalAction", "HorizontalAction",
                    "vertical", "verticalAction", "VerticalAction"):
            val = UAV.getAttr(actionObj, key, default=None)
            if val is not None:
                return UAV.readActionName(val)
        return str(actionObj).upper()
    
    @staticmethod
    def _calculate_yaw_angle(horizontalName: any) -> int | float:
        """
        计算偏航角
        :param horizontalName: 水平动作名称
        :return: 偏航角（度）
        """
        # 水平动作对应的偏航角变化（度/秒）
        deltaYawDeg = 0.0
        if "TURN_LEFT_FAST" in horizontalName:
            deltaYawDeg = +6.0
        elif "TURN_RIGHT_FAST" in horizontalName:
            deltaYawDeg = -6.0
        elif "TURN_LEFT_SLOW" in horizontalName:
            deltaYawDeg = +3.0
        elif "TURN_RIGHT_SLOW" in horizontalName:
            deltaYawDeg = -3.0
        # NONE 或未知则为 0
        return deltaYawDeg
    
    @staticmethod
    def _calculate_vel_up(verticalName: any) -> int | float:
        # 垂直动作对应的上升速度变化（m/s）
        deltaVelUp = 0.0
        if "CLIMB" in verticalName:
            deltaVelUp = +5.0
        elif "DESCEND" in verticalName:
            deltaVelUp = -5.0
        # NONE/HOVER 视为 0
        return deltaVelUp
    
    @staticmethod
    def noisy(value: float, acc: float, std: float) -> float:
        """
        对真值叠加均匀偏置与高斯噪声：value + U(-acc, acc) + N(0, std)。
        :param value: 真值
        :param acc: 均匀偏置范围
        :param std: 高斯噪声标准差
        :return: 噪声值
        """
        return float(value + np.random.uniform(-acc, acc) + np.random.normal(0.0, std))




if __name__ == "__main__":
    pass