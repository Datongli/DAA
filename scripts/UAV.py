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
        self.actionENU: Action | None = None
        self.dt = getattr(self.cfg, "dt", 0.1)  # 模拟时间间隔
            
    def update(self, data: Dict, actionENU: Action | None = None) -> np.array:
        """
        UAV更新逻辑
        :param data: 传感器数据
        """
        """更新传入的东北高三维动作"""
        self.actionENU = actionENU
        """更新传感器数据"""
        # 先得到轨迹数据
        trackData = data.get("Track")
        if trackData:
            self.sensors["Track"].get_observations(trackData)
        # 读取Cloud数据（有可能为空）
        cloudData = data.get("Cloud", [])
        if cloudData:
            self.sensors["Cloud"].get_observations(cloudData)  
        # 读取IntruderReal数据（入侵者真实轨迹）
        intruderRealData = data.get("IntruderReal", [])
        if intruderRealData and self.simulateSensors:
            # 基于真实轨迹生成模拟传感器数据
            simSensorData = self._simulate_sensor_measurements(intruderRealData)
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
    
    def get_own_state(self) -> any:
        """
        获取无人机自身状态估计
        :return: 无人机自身状态估计
        """
        # 先用来占位，后续根据需要进行实现
        pass

    def read_action(self, actions: Action | None = None) -> None:
        """
        读取无人机动作
        :param actions: 无人机动作
        """
        # 先用来占位，后续根据需要进行实现
        self.actionENU = actions
    
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
            
    def _simulate_sensor_measurements(self, intruderRealData: List[Dict]) -> Dict[str, list]:
        """
        基于上一时刻自身状态与上一时刻动作，推演当前时刻自身真实状态；再用该状态与 IntruderReal 入侵者真值生成 Radar/UWB 的量测。
        返回当前时刻的 Track、Radar、UWB 三类数据（各为单条记录的字典）。
        :param intruderRealData: 入侵者真实轨迹数据
        :return: 模拟传感器数据字典
        """
        # 若 intruderRealData 为空，返回空数据
        if not intruderRealData:
            return {"Track": [], "Radar": [], "UWB": []}
        # 取第一个入侵者数据（支持多入侵者可在此扩展）
        intruderData = intruderRealData[0] if isinstance(intruderRealData, list) else intruderRealData
        """读取入侵者真实位置（ENU）"""
        intruderId = self.getAttr(intruderData, "UAVID")  # 获取入侵者ID
        timeStamp = self.getAttr(intruderData, "timeStamp", default=None)  # 获取时间戳
        intruderCoordinate = self.getAttr(intruderData, "coordinate", default={})  # 获取东北高坐标
        intruderVelocity = self.getAttr(intruderData, "velocity", default={})  # 获取东北高速度
        targetEast = float(self.readNumber(intruderCoordinate, "east", "E", "e", default=0.0))  # 获取东坐标
        targetNorth = float(self.readNumber(intruderCoordinate, "north", "N", "n", default=0.0))  # 获取北坐标
        targetUp = float(self.readNumber(intruderCoordinate, "up", "U", "u", default=0.0))  # 获取高坐标
        """计算当前时刻自身真实状态"""
        self.dt = timeStamp - self.ownState.timeStamp if self.ownState else 1.0  # 计算时间间隔
        # ---------- 首帧初始化逻辑开始 ----------
        # 如果目前 ownState 为空，说明这是首次调用：用 Track 里的自身数据初始化上一时刻状态
        if self.ownState is None:
            dt = 0.0
            # 从 Track 传感器中读取当前时刻的自身状态（假设已经在 UAV.update 中喂入了 Track 数据）
            trackSensor = self.sensors.get("Track", None)
            if trackSensor is not None and getattr(trackSensor, "data", None):
                # 取最后一条 Track 数据（或根据需要取与 timeStamp 匹配的条目）
                lastTrack = trackSensor.data[-1]
                trackCoord = self.getAttr(lastTrack, "coordinate", default={})
                trackVel = self.getAttr(lastTrack, "velocity", default={})
                ownEastPrev = float(self.readNumber(trackCoord, "east", "E", "e", default=0.0))
                ownNorthPrev = float(self.readNumber(trackCoord, "north", "N", "n", default=0.0))
                ownUpPrev = float(self.readNumber(trackCoord, "up", "U", "u", default=0.0))
                velEastPrev = float(self.readNumber(trackVel, "eastVelocity", "ve", "vx", default=0.0))
                velNorthPrev = float(self.readNumber(trackVel, "northVelocity", "vn", "vy", default=0.0))
                velUpPrev = float(self.readNumber(trackVel, "upVelocity", "vu", "vz", default=0.0))
                self.ownState = StateEstimate(
                    ID=self.ID,
                    position=trackCoord,
                    velocity=trackVel,
                    covariance=None,
                    timeStamp=timeStamp
                )
            else:
                # 如果 Track 里也没有任何数据，只能退回到 0（这种情况一般不应该发生）
                ownEastPrev = ownNorthPrev = ownUpPrev = 0.0
                velEastPrev = velNorthPrev = velUpPrev = 0.0
        else:
            # ---------- 非首帧：按原有逻辑从 self.ownState 取上一时刻状态 ----------
            ownStatePrev = self.ownState
            ownCoordPrev = self.getAttr(ownStatePrev, "position", default=None)  # 获取上一时刻自身位置
            ownVelPrev = self.getAttr(ownStatePrev, "velocity", default=None)  # 获取上一时刻自身速度
            if ownCoordPrev is None:
                ownEastPrev = float(self.readNumber(ownStatePrev, "east", "E", "e", default=0.0))
                ownNorthPrev = float(self.readNumber(ownStatePrev, "north", "N", "n", default=0.0))
                ownUpPrev = float(self.readNumber(ownStatePrev, "up", "U", "u", default=0.0))
            else:
                ownEastPrev = float(self.readNumber(ownCoordPrev, "east", "E", "e", default=0.0))
                ownNorthPrev = float(self.readNumber(ownCoordPrev, "north", "N", "n", default=0.0))
                ownUpPrev = float(self.readNumber(ownCoordPrev, "up", "U", "u", default=0.0))
            if ownVelPrev is None:
                velEastPrev = float(self.readNumber(ownStatePrev, "eastVelocity", "ve", "vx", default=0.0))
                velNorthPrev = float(self.readNumber(ownStatePrev, "northVelocity", "vn", "vy", default=0.0))
                velUpPrev = float(self.readNumber(ownStatePrev, "upVelocity", "vu", "vz", default=0.0))
            else:
                velEastPrev = float(self.readNumber(ownVelPrev, "eastVelocity", "ve", "vx", default=0.0))
                velNorthPrev = float(self.readNumber(ownVelPrev, "northVelocity", "vn", "vy", default=0.0))
                velUpPrev = float(self.readNumber(ownVelPrev, "upVelocity", "vu", "vz", default=0.0))
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
        # 获取动作
        if self.actionENU is None:
            # 采用默认的规定好的动作
            horizontalActionObj = self.getAttr(self, "actions", default=None)  # 提取水平动作对象
            horizontalActionObj = self.getAttr(horizontalActionObj, "horizontal", default=None)  # 获取水平动作
            horizontalName = self.readActionName(horizontalActionObj)  # 提取水平动作名称
            verticalActionObj = self.getAttr(self, "actions", default=None)  # 提取垂直动作对象
            verticalActionObj = self.getAttr(verticalActionObj, "vertical", default=None)  # 获取垂直动作
            verticalName = self.readActionName(verticalActionObj)  # 提取垂直动作名称
            deltaYawDeg = self._calculate_yaw_angle_by_name(horizontalName)  # 计算偏航角变化
            deltaVelUp = self._calculate_vel_up_by_name(verticalName)  # 计算垂直速度变化
            # 保持水平速度不变，旋转水平速度矢量
            horizontalSpeedPrev = math.hypot(velEastPrev, velNorthPrev)
            if horizontalSpeedPrev > 1e-6 and abs(deltaYawDeg) > 0.0:
                deltaYawRad = self.deg2rad(deltaYawDeg)  # 左转（逆时针）为正，右转（顺时针）为负
                cosD, sinD = math.cos(deltaYawRad), math.sin(deltaYawRad)
                velEastCurr = cosD * velEastPrev - sinD * velNorthPrev  # 当前东方向速度
                velNorthCurr = sinD * velEastPrev + cosD * velNorthPrev  # 当前北方向速度
            else:
                velEastCurr, velNorthCurr = velEastPrev, velNorthPrev
            # 垂直速度更新
            velUpCurr = velUpPrev + deltaVelUp
        else:
            velEastCurr, velNorthCurr, velUpCurr = self.actionENU.east, self.actionENU.north, self.actionENU.up
        # 位置积分
        ownEastCurr = ownEastPrev + velEastCurr * self.dt  # 当前东坐标
        ownNorthCurr = ownNorthPrev + velNorthCurr * self.dt  # 当前北坐标
        ownUpCurr = ownUpPrev + velUpCurr * self.dt  # 当前高坐标
        # 姿态更新：简单起见只更新偏航角，俯仰角和滚转角保持不变
        ownAttitudePrev = getattr(self.stm, "ownAttitude", None)
        yawPrev = float(self.readNumber(ownAttitudePrev, "yaw", "Yaw", default=0.0)) if ownAttitudePrev is not None else self.rad2deg(lastTrack.yaw)  # 默认偏航角朝向正北，90度
        pitchPrev = float(self.readNumber(ownAttitudePrev, "pitch", "Pitch", default=0.0)) if ownAttitudePrev is not None else 0.0
        rollPrev = float(self.readNumber(ownAttitudePrev, "roll", "Roll", default=0.0)) if ownAttitudePrev is not None else 0.0
        # 修正：偏航角0度指向正东
        # 从速度矢量计算偏航角：atan2(velNorth, velEast)，0度=正东
        horizontalSpeedCurr = math.hypot(velEastCurr, velNorthCurr)
        yawCurr = math.atan2(velNorthCurr, velEastCurr) if horizontalSpeedCurr > 1e-6 else yawPrev
        # 俯仰角和滚转角保持不变
        pitchCurr = pitchPrev
        rollCurr = rollPrev
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
        # 计算入侵者相对位置（ENU）
        relativeEnu = np.array([
            targetEast - ownEastCurr,
            targetNorth - ownNorthCurr,
            targetUp - ownUpCurr
        ], dtype=float)
        # ENU → 机体坐标（x前 y右 z上）
        # 构建完整的旋转矩阵
        rYaw = self.rotZ(yawCurr)
        rPitch = self.rotY(pitchCurr)
        rRoll = self.rotX(rollCurr)
        # 机体到ENU的基础变换（0度=正东）
        rBody2ENU = np.array([
            [1, 0, 0],    # 机体x(前) -> ENU x(东)
            [0, -1, 0],   # 机体y(右) -> ENU -y(南)
            [0, 0, -1]    # 机体z(下) -> ENU -z(下)
        ])
        rotBodyWorld = rYaw @ rPitch @ rRoll @ rBody2ENU
        relativeBody = rotBodyWorld.T @ relativeEnu
        # 计算极坐标参数
        rangeTrue = float(np.linalg.norm(relativeBody))
        xyNorm = max(1e-9, math.hypot(relativeBody[0], relativeBody[1]))
        azimuthTrueRad = math.atan2(relativeBody[1], relativeBody[0])   # 方位角（机体坐标，弧度）
        elevationTrueRad = math.atan2(relativeBody[2], xyNorm)          # 俯仰角（机体坐标，弧度）
        # 读取传感器精度/噪声配置（缺失直接用默认值）
        radarCfg = self.getAttr(self.cfg, "Radar", None)
        uwbCfg = self.getAttr(self.cfg, "UWB", None)
        # Radar配置
        radarMaxRange = self.readNumber(radarCfg, "maxRange", default=150.0)
        radarMinRange = self.readNumber(radarCfg, "minRange", default=1.0)
        radarMaxAzimuthDeg = self.readNumber(radarCfg, "maxAzimuthDeg", default=120.0) / 2.0
        radarMaxElevationDeg = self.readNumber(radarCfg, "maxElevationDeg", default=60.0) / 2.0
        radarRangeAcc = self.readNumber(radarCfg, "rangeAcc", default=0.1)
        radarRangeStd = self.readNumber(radarCfg, "rangeStd", default=0.5)
        radarAzimuthAcc = self.readNumber(radarCfg, "azimuthAcc", default=1.0)
        radarAzimuthStd = self.readNumber(radarCfg, "azimuthStd", default=0.05)
        radarElevationAcc = self.readNumber(radarCfg, "elevationAcc", default=2.0)
        radarElevationStd = self.readNumber(radarCfg, "elevationStd", default=0.05)
        # UWB配置
        uwbMaxRange = self.readNumber(uwbCfg, "maxRange", default=200.0)
        uwbMinRange = self.readNumber(uwbCfg, "minRange", default=1.0)
        uwbRangeAcc = self.readNumber(uwbCfg, "rangeAcc", default=0.1)
        uwbRangeStd = self.readNumber(uwbCfg, "rangeStd", default=0.3)
        uwbAzimuthAcc = self.readNumber(uwbCfg, "azimuthAcc", default=1.0)
        uwbAzimuthStd = self.readNumber(uwbCfg, "azimuthStd", default=0.08)
        uwbElevationAcc = self.readNumber(uwbCfg, "elevationAcc", default=2.0)
        uwbElevationStd = self.readNumber(uwbCfg, "elevationStd", default=0.08)
        # 检查Radar探测限制
        radarMsg = []
        azimuthTrueDeg = self.rad2deg(azimuthTrueRad)
        elevationTrueDeg = self.rad2deg(elevationTrueRad)
        radarInRange = (radarMinRange <= rangeTrue <= radarMaxRange)
        radarInAzimuth = (abs(azimuthTrueDeg) <= radarMaxAzimuthDeg)
        radarInElevation = (abs(elevationTrueDeg) <= radarMaxElevationDeg)
        if radarInRange and radarInAzimuth and radarInElevation:
            # 在探测范围内，生成带有噪声的数据
            radarRange = self.noisy(rangeTrue, radarRangeAcc, radarRangeStd)
            radarAzimuth = self.noisy(azimuthTrueDeg, radarAzimuthAcc, radarAzimuthStd)
            radarElevation = self.noisy(elevationTrueDeg, radarElevationAcc, radarElevationStd)
            radarMsg = [{
                "timeStamp": timeStamp,
                "range": radarRange,
                "azimuth": radarAzimuth,
                "elevation": radarElevation,
            }]
        # 检查UWB探测限制
        uwbMsg = []
        uwbInRange = (uwbMinRange <= rangeTrue <= uwbMaxRange)
        if uwbInRange:
            # 在探测范围内，生成带噪声的数据
            uwbRange = self.noisy(rangeTrue, uwbRangeAcc, uwbRangeStd)
            uwbAzimuth = self.noisy(azimuthTrueDeg, uwbAzimuthAcc, uwbAzimuthStd)
            uwbElevation = self.noisy(elevationTrueDeg, uwbElevationAcc, uwbElevationStd)
            uwbMsg = [{
                "timeStamp": timeStamp,
                "range": uwbRange,
                "azimuth": uwbAzimuth,
                "elevation": uwbElevation,
            }]
        return {"Track": [trackMsg], "Radar": radarMsg, "UWB": uwbMsg}

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
    def rotX(angle: float) -> np.ndarray:
        """绕X轴旋转矩阵"""
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

    @staticmethod
    def rotY(angle: float) -> np.ndarray:
        """绕Y轴旋转矩阵"""
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

    @staticmethod
    def rotZ(angle: float) -> np.ndarray:
        """绕Z轴旋转矩阵（偏航角，0度=正东）"""
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    
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
    def _calculate_yaw_angle_by_name(horizontalName: any) -> int | float:
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
    def _calculate_vel_up_by_name(verticalName: any) -> int | float:
        """
        根据动作名称计算上升速度变化
        :param verticalName: 垂直动作名称
        :return: 上升速度变化（m/s）
        """
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