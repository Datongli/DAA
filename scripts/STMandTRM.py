"""
定义STM和TRM类
"""
from sensor import *
from typing import Tuple
from dataClass import *
from filterpy.kalman import KalmanFilter
from Algorithm import *


class TrackFile:
    """
    管理单个目标的传感器特定跟踪文件
    """
    def __init__(self, targetID: float| int| str, sensorType: str, initCovariance: np.ndarray) -> None:
        """
        初始化跟踪文件
        :param targetID: 目标ID
        :param sensorType: 传感器类型
        :param initCovariance: 传感器初始协方差
        :return: None
        """
        self.targetID: float| int| str = targetID  # 目标ID
        self.sensorType: str = sensorType  # 传感器类型
        self.initCovariance: np.ndarray = initCovariance  # 传感器初始协方差
        self.kf: KalmanFilter = None  # 卡尔曼滤波器实例
        self.lastUpdate: float| int = -1.0  # 最后更新时间

    def update(self, sensorData: CloudData| RadarData| UWBData, ownState: StateEstimate| None = None,
               ownAttitude: Attitude| None = None) -> None:
        """
        更新跟踪文件状态
        :param sensorData: 传感器数据
        :param ownState: 无人机自身状态
        :param ownAttitude: 无人机自身姿态
        :return: None
        """
        if self.kf is None:
            self._init_filter(sensorData, ownState, ownAttitude)
            return
        dt = sensorData.timeStamp - self.lastUpdate  # 计算时间间隔
        if dt > 0:
            # 更新状态转移矩阵
            self.kf.F[0, 3] = dt
            self.kf.F[1, 4] = dt
            self.kf.F[2, 5] = dt
            self.kf.predict()  # 预测步骤
        if self.sensorType == "Cloud":
            # 观测向量，全状态，包含状态和速度
            z = np.array([
                sensorData.coordinate.east,
                sensorData.coordinate.north,
                sensorData.coordinate.up,
                sensorData.velocity.eastVelocity,
                sensorData.velocity.northVelocity,
                sensorData.velocity.upVelocity
            ])
            H = np.eye(6)  # 观测矩阵
            R = self.initCovariance  # CloudSensor的6x6协方差
        else:
            # Radar/UWB观测（极坐标转ENU位置）
            enuPosition = TrackFile.body_to_enu(
                sensorData.range,
                sensorData.azimuth,
                sensorData.elevation,
                ownState.position,
                ownAttitude.yaw if hasattr(ownAttitude, 'yaw') else 0,
                ownAttitude.pitch if hasattr(ownAttitude, 'pitch') else 0,
                ownAttitude.roll if hasattr(ownAttitude, 'roll') else 0
            )
            z_vec = enuPosition
            H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])
            # 误差传播，得到ENU下的观测协方差
            J = TrackFile.jacobian_body_to_enu(
                sensorData.range,
                sensorData.azimuth,
                sensorData.elevation,
                ownAttitude.yaw if hasattr(ownAttitude, 'yaw') else 0,
                ownAttitude.pitch if hasattr(ownAttitude, 'pitch') else 0,
                ownAttitude.roll if hasattr(ownAttitude, 'roll') else 0
            )
            R = J @ self.initCovariance @ J.T  # 3x3
        # 卡尔曼滤波更新
        self.kf.update(z if self.sensorType == "Cloud" else z_vec, R, H)
        self.lastUpdate = sensorData.timeStamp  # 更新最后更新时间

    def _init_filter(self, sensorData: CloudData| RadarData| UWBData, ownState: StateEstimate| None = None,
                     ownAttitude: Attitude| None = None) -> None:
        """
        初始化卡尔曼滤波器
        :param sensorData: 传感器数据
        :param ownState: 无人机自身状态
        :param ownAttitude: 无人机自身姿态
        :return: None
        """
        if self.sensorType == "Cloud":
            # 直接使用云平台提供的数据
            initState = np.array([
                sensorData.coordinate.east,
                sensorData.coordinate.north,
                sensorData.coordinate.up,
                sensorData.velocity.eastVelocity,
                sensorData.velocity.northVelocity,
                sensorData.velocity.upVelocity
            ])  # 初始状态
            initCovariance = self.initCovariance  # 初始协方差
            self.kf = TrackFile.init_kalman_cloud(initState, initCovariance)
        else:
            # 如果是radar或者uwb，使用考虑姿态角的坐标转换
            # 初始状态
            enuPosition = TrackFile.body_to_enu(
                sensorData.range,
                sensorData.azimuth,
                sensorData.elevation,
                ownState.position,
                ownAttitude.yaw if hasattr(ownAttitude, 'yaw') else 0,
                ownAttitude.pitch if hasattr(ownAttitude, 'pitch') else 0,
                ownAttitude.roll if hasattr(ownAttitude, 'roll') else 0
            )
            initState = np.array([enuPosition[0], enuPosition[1], enuPosition[2], 0, 0, 0])  # 初始状态 (假设速度为零)
            J = TrackFile.jacobian_body_to_enu(
                sensorData.range,
                sensorData.azimuth,
                sensorData.elevation,
                ownAttitude.yaw if hasattr(ownAttitude, 'yaw') else 0,
                ownAttitude.pitch if hasattr(ownAttitude, 'pitch') else 0,
                ownAttitude.roll if hasattr(ownAttitude, 'roll') else 0
            )
            initCovariance = J @ self.initCovariance @ J.T  # 初始协方差
            self.kf = TrackFile.init_kalman_radar_or_uwb(initState, initCovariance)
        self.lastUpdate = sensorData.timeStamp

    @staticmethod
    def init_kalman_cloud(initState: np.ndarray, initCovariance: np.ndarray) -> KalmanFilter:
        """
        静态方法，初始化适用于Cloud的卡尔曼滤波器
        :param initState: 初始状态
        :param initCovariance: 初始协方差
        :return: 初始化后的卡尔曼滤波器对象
        """
        kf = KalmanFilter(dim_x=6, dim_z=6)
        kf.x = initState  # 初始状态
        kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])  # 状态转移矩阵，匀速模型
        kf.H = np.eye(6)  # 观测矩阵，直接观测位置和速度
        kf.P = initCovariance  # 初始协方差
        kf.R = initCovariance  # 观测噪声
        kf.Q = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])  # 过程噪声
        return kf

    @staticmethod
    def init_kalman_radar_or_uwb(initState: np.ndarray, initCovariance: np.ndarray) -> KalmanFilter:
        """
        静态方法，初始化适用于Radar或UWB的卡尔曼滤波器
        :param initState: 初始状态
        :param initCovariance: 初始协方差
        :return: 初始化后的卡尔曼滤波器对象
        """
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = initState  # 初始状态
        kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])  # 状态转移矩阵
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])  # 观测矩阵（只观测位置）
        # 通过雅可比矩阵做误差传播，得到ENU坐标下的初始协方差矩阵
        P = np.zeros((6, 6))
        P[:3, :3] = initCovariance
        P[3:, 3:] = np.diag([10, 10, 10])  # 速度部分协方差，设大些
        kf.P = P  # 初始协方差矩阵
        kf.R =  initCovariance  # 观测噪声, 只用位置部分的协方差
        kf.Q = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]) # 过程噪声
        return kf
    
    @staticmethod
    def jacobian_enu(range: float| int, azimuth: float| int, elevation: float| int) -> np.ndarray:
        """
        静态方法，构造ENU坐标下的雅可比矩阵
        :param range: 距离
        :param azimuth: 方位角
        :param elevation: 仰角
        :return: ENU坐标下的雅可比矩阵
        """
        J = np.zeros((3, 3))
        J[0, 0] = np.cos(elevation) * np.sin(azimuth)           # dx/dr
        J[1, 0] = np.cos(elevation) * np.cos(azimuth)           # dy/dr
        J[2, 0] = np.sin(elevation)                        # dz/dr
        J[0, 1] = range * np.cos(elevation) * np.cos(azimuth)       # dx/daz
        J[1, 1] = -range * np.cos(elevation) * np.sin(azimuth)      # dy/daz
        J[2, 1] = 0                                 # dz/daz
        J[0, 2] = -range * np.sin(elevation) * np.sin(azimuth)      # dx/del
        J[1, 2] = -range * np.sin(elevation) * np.cos(azimuth)      # dy/del
        J[2, 2] = range * np.cos(elevation)                    # dz/del
        return J
    
    @staticmethod
    def calculate_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        静态方法，计算从机体坐标系到导航坐标系的旋转矩阵
        :param yaw: 偏航角 (弧度，逆时针为正，0°=正东)
        :param pitch: 俯仰角 (弧度，上仰为正)
        :param roll: 滚转角 (弧度，右滚为正)
        :return: 从机体坐标系到导航坐标系的旋转矩阵
        """
        # 滚转矩阵（绕x轴旋转）
        rRoll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        # 俯仰矩阵（绕y轴旋转）
        rPitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        # 偏航矩阵（绕z轴旋转，0度=正东）
        rYaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # 从机体坐标系到导航坐标系的旋转矩阵
        rBody2ENU = np.array([
            [1, 0, 0],    # 机体x(前) -> ENU x(东)
            [0, -1, 0],   # 机体y(右) -> ENU -y(南)
            [0, 0, -1]    # 机体z(下) -> ENU -z(下)
        ])
        rTotal = rYaw @ rPitch @ rRoll @ rBody2ENU
        return rTotal

    @staticmethod
    def body_to_enu(range: float, azimuth: float, elevation: float, 
                    ownPosition: Position, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        将机体坐标系下的极坐标转换为ENU坐标系下的位置
        Args:
            range: 距离 (米)
            azimuth: 方位角 (弧度，机体坐标系)
            elevation: 俯仰角 (弧度，机体坐标系)
            ownPosition: 自身位置
            yaw: 偏航角 (弧度，逆时针为正，0°=北)
            pitch: 俯仰角 (弧度，上仰为正)
            roll: 滚转角 (弧度，右滚为正)
        Returns:
            ENU坐标系下的位置 [east, north, up]
        """
        """将极坐标转换成机体坐标系下直角坐标"""
        # 假设机体坐标系：x轴指向机头正前方，y轴指向右侧，z轴指向下方
        xBody = range * np.cos(azimuth) * np.cos(elevation)
        yBody = range * np.sin(azimuth) * np.cos(elevation)
        zBody = - range * np.sin(elevation)
        bodyVector = np.array([xBody, yBody, zBody])  # 入侵者在机体坐标系下的位置
        """构造从机体坐标系到ENU坐标系的旋转矩阵"""
        # 旋转顺序：先滚转，再俯仰，最后偏航 (ZYX欧拉角)
        # 总旋转矩阵
        rTotal = TrackFile.calculate_rotation_matrix(yaw, pitch, roll)
        # 旋转变换
        enuVector = rTotal @ bodyVector
        """加上自身位置偏移"""
        enuPosition = np.array([
            ownPosition.east + enuVector[0],
            ownPosition.north + enuVector[1],
            ownPosition.up + enuVector[2]
        ])
        return enuPosition
    
    @staticmethod
    def jacobian_body_to_enu(range: float, azimuth: float, elevation: float,
                            yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        计算从机体极坐标到ENU坐标的雅可比矩阵
        用于误差传播计算
        
        Args:
            range: 距离 (米)
            azimuth: 方位角 (弧度，机体坐标系)
            elevation: 俯仰角 (弧度，机体坐标系)
            yaw: 偏航角 (弧度，逆时针为正，0°=北)
            pitch: 俯仰角 (弧度，上仰为正)
            roll: 滚转角 (弧度，右滚为正)
        
        Returns:
            3x3雅可比矩阵 [∂(east,north,up)/∂(range,azimuth,elevation)]
        """
        """构造旋转矩阵"""
        rTotal = TrackFile.calculate_rotation_matrix(yaw, pitch, roll)
        """计算偏导数"""
        # 对range的偏导数
        dxDr = np.cos(elevation) * np.cos(azimuth)
        dyDr = np.cos(elevation) * np.sin(azimuth)
        dzDr = - np.sin(elevation)
        # 对azimuth的偏导数
        dxDaz = -range * np.cos(elevation) * np.sin(azimuth)
        dyDaz = range * np.cos(elevation) * np.cos(azimuth)
        dzDaz = 0
        # 对elevation的偏导数
        dxDel = -range * np.sin(elevation) * np.cos(azimuth)
        dyDel = -range * np.sin(elevation) * np.sin(azimuth)
        dzDel = -range * np.cos(elevation)
        """构造雅可比矩阵"""
        jBody = np.array([
            [dxDr, dxDaz, dxDel],
            [dyDr, dyDaz, dyDel],
            [dzDr, dzDaz, dzDel]
        ])
        # 应用旋转矩阵
        J = rTotal @ jBody
        return J

    def get_state_estimate(self) -> StateEstimate:
        """
        获取当前状态估计
        :return: 当前状态估计对象
        """
        if self.kf is None:
            return None
        # 提取状态向量和协方差
        state = self.kf.x  # 6维状态向量，[东, 北, 高, 东向速度, 北向速度, 垂直速度]
        covariance = self.kf.P  # 6x6协方差矩阵
        # 构造StateEstimate对象
        return StateEstimate(
            ID=self.targetID,
            position=Position(east=state[0], north=state[1], up=state[2]),
            velocity=Velocity(eastVelocity=state[3], northVelocity=state[4], upVelocity=state[5]),
            covariance=covariance,
            timeStamp=self.lastUpdate
        )


class STM:
    """STM类"""
    def __init__(self, ownID: float| int| str, sensors: Dict[str, Sensor]) -> None:
        """
        初始化STM对象
        :param ownID: 无人机自身ID
        :param sensors: 无人机搭载的传感器字典
        :return: None
        """
        self.ownID: float| int| str = ownID  # 无人机自身ID
        self.sensors: Dict[str, Sensor] = sensors  # 无人机搭载的传感器字典
        self.trackFiles: Dict[str, Dict[str, TrackFile]] = {}  # 目标ID -> 跟踪文件集合
        self.ownState: None| StateEstimate = None  # 无人机自身状态估计
        self.targets: Dict[str, StateEstimate] = {}  # 目标状态估计字典
        self.ownAttitude: Attitude| None = None  # 无人机自身姿态

    def update(self) -> Tuple[None| StateEstimate, Dict[str, StateEstimate]]:
        """
        执行完整的STM更新周期
        :return: 无人机自身状态估计和所有目标状态估计字典
        """
        self._update_own_state()  # 更新自身状态
        self._update_track_files()  # 更新跟踪文件
        self._associate_targets()  # 目标关联
        return self.ownState, self.targets, self.trackFiles

    def _update_own_state(self) -> None:
        """
        用Track更新自身的状态估计
        :return: None
        """
        # 用Track更新自身状态
        self.ownState = StateEstimate(
            ID=self.ownID,
            position=self.sensors["Track"].data[0].coordinate,
            velocity=self.sensors["Track"].data[0].velocity,
            covariance=self.sensors["Track"].covariance,
            timeStamp=self.sensors["Track"].data[0].timeStamp
        )
        # 保存姿态角信息用于坐标转换
        self.ownAttitude = Attitude(
            yaw=self.sensors["Track"].data[0].yaw,
            pitch=self.sensors["Track"].data[0].pitch,
            roll=self.sensors["Track"].data[0].roll
        )

    def _update_track_files(self) -> None:
        """
        数据关联与更新跟踪文件
        处理三种情况：
        1. 只有Cloud数据
        2. 只有Radar/UWB数据
        3. Cloud和Radar/UWB数据都有
        :return: None
        """
        """第一步：若有，则处理Cloud数据"""
        cloudTargets = set()  # 记录本次时刻Cloud观测到的目标ID
        if "Cloud" in self.sensors and len(self.sensors["Cloud"].data) > 0:
            for cloudData in self.sensors["Cloud"].data:
                # 跳过自身数据
                if cloudData.UAVID == self.ownID:
                    continue
                cloudTargets.add(cloudData.UAVID)
                # 检查是否存在与目标ID相关的跟踪文件
                if cloudData.UAVID not in self.trackFiles:
                    self.trackFiles[cloudData.UAVID] = {}
                # 创建或更新Cloud跟踪文件
                if "Cloud" not in self.trackFiles[cloudData.UAVID]:
                    self.trackFiles[cloudData.UAVID]["Cloud"] = TrackFile(
                        cloudData.UAVID, 
                        "Cloud", 
                        self.sensors["Cloud"].covariance
                    )
                # 更新Cloud跟踪文件
                self.trackFiles[cloudData.UAVID]["Cloud"].update(cloudData)
        """第二步：处理Radar和UWB数据"""
        for sensorType in ["Radar", "UWB"]:
            # 检查传感器是否存在且有数据
            if sensorType not in self.sensors or len(self.sensors[sensorType].data) == 0:
                continue
            if len(cloudTargets) > 0:
                """情况A：有Cloud数据 - 使用Cloud数据进行关联"""
                self._associate_sensor_with_cloud(sensorType, cloudTargets)
            else:
                """情况B：没有Cloud数据 - 使用已有跟踪文件或创建新的"""
                self._associate_sensor_without_cloud(sensorType)
        """第三步：清理过时的跟踪文件"""
        self._cleanup_stale_tracks()

    def _associate_sensor_with_cloud(self, sensorType: str, cloudTargets: set) -> None:
        """
        当有Cloud数据时，将Radar/UWB数据与Cloud数据关联
        :param sensorType: 传感器类型（"Radar"或"UWB"）
        :param cloudTargets: 本次Cloud观测到的目标ID集合
        :return: None
        """
        sensorDataList = self.sensors[sensorType].data  # 获取传感器数据列表
        # 为每个Cloud目标寻找最匹配的传感器数据
        for targetID in cloudTargets:
            if targetID not in self.trackFiles or "Cloud" not in self.trackFiles[targetID]:
                continue
            # 获取Cloud跟踪文件的最新状态
            cloudTrackFile = self.trackFiles[targetID]["Cloud"]
            cloudState = cloudTrackFile.get_state_estimate()
            if cloudState is None:
                continue
            cloudPosition = np.array([
                cloudState.position.east,
                cloudState.position.north,
                cloudState.position.up
            ])
            # 找到与cloud位置最匹配的传感器数据
            bestMatch = None
            bestDistance = float('inf')
            for sensorData in sensorDataList:
                # 转换传感器数据到ENU坐标系
                posInENU = TrackFile.body_to_enu(
                    sensorData.range,
                    sensorData.azimuth,
                    sensorData.elevation,
                    self.ownState.position,
                    self.ownAttitude.yaw,
                    self.ownAttitude.pitch,
                    self.ownAttitude.roll
                )
                # 计算欧式距离（用于初步筛选）
                distance = np.linalg.norm(posInENU - cloudPosition)
                if distance < bestDistance:
                    bestDistance = distance
                    bestMatch = (sensorData, posInENU)
            # 如果找到匹配，进行马氏距离验证
            if bestMatch is not None:
                sensorData, posInENU = bestMatch
                # 计算马氏距离
                cloudCovariance = self.sensors["Cloud"].covariance[0:3, 0:3]
                rowCovariance = self.sensors[sensorType].covariance
                J = TrackFile.jacobian_body_to_enu(
                    sensorData.range,
                    sensorData.azimuth,
                    sensorData.elevation,
                    self.ownAttitude.yaw,
                    self.ownAttitude.pitch,
                    self.ownAttitude.roll
                )
                sensorCovariance = J @ rowCovariance @ J.T
                mahalanobisDistance = self.mahalanobis_distance(
                    cloudPosition,
                    posInENU,
                    cloudCovariance,
                    sensorCovariance
                )
                # 马氏距离阈值检验
                if mahalanobisDistance < 5.0:
                    # 创建或更新传感器跟踪文件
                    if sensorType not in self.trackFiles[targetID]:
                        self.trackFiles[targetID][sensorType] = TrackFile(
                            targetID,
                            sensorType,
                            self.sensors[sensorType].covariance
                        )
                    self.trackFiles[targetID][sensorType].update(
                        sensorData,
                        self.ownState,
                        self.ownAttitude
                    )

    def _associate_sensor_without_cloud(self, sensorType: str) -> None:
        """
        当没有Cloud数据时，处理Radar/UWB数据 
        策略：
        1. 先将所有现有跟踪文件预测到当前时刻
        2. 用预测位置与传感器数据进行匹配
        3. 如果没有匹配的跟踪文件，创建新的"未知目标"跟踪文件
        :param sensorType: 传感器类型（"Radar"或"UWB"）
        :return: None
        """
        if self.ownState is None:
            return
        sensorDataList = self.sensors[sensorType].data  # 获取传感器数据列表
        # 如果没有传感器数据，直接返回
        if len(sensorDataList) == 0:
            return
        # 获取传感器数据的时间戳（假设同一批数据时间戳相同）
        currentSensorTime = sensorDataList[0].timeStamp
        usedSensorData = set()  # 记录已使用的传感器数据索引
        """预测所有现有跟踪文件到当前传感器时刻"""
        predictedStates = {}  # {targetID: predicted_position}
        for targetID, trackDict in self.trackFiles.items():
            # 过滤不存在该传感器类型的跟踪文件
            if sensorType not in trackDict:
                continue
            trackFile = trackDict[sensorType]
            # 计算时间差
            dt = currentSensorTime - trackFile.lastUpdate
            # 如果时间差过大（超过10秒），认为跟踪丢失，跳过
            if dt > 10.0 or dt < 0:
                continue
            # 预测到当前时刻（但不更新lastUpdate）
            if dt > 0 and trackFile.kf is not None:
                # 临时保存原始状态
                original_x = trackFile.kf.x.copy()
                original_P = trackFile.kf.P.copy()
                original_F = trackFile.kf.F.copy()
                # 更新状态转移矩阵并预测
                trackFile.kf.F[0, 3] = dt
                trackFile.kf.F[1, 4] = dt
                trackFile.kf.F[2, 5] = dt
                trackFile.kf.predict()
                # 提取预测位置
                predictedPos = np.array([
                    trackFile.kf.x[0],  # east
                    trackFile.kf.x[1],  # north
                    trackFile.kf.x[2]   # up
                ])
                predictedStates[targetID] = {
                    'position': predictedPos,
                    'trackFile': trackFile
                }
                # 恢复原始状态（预测只是为了匹配，实际更新在后面）
                trackFile.kf.x = original_x
                trackFile.kf.P = original_P
                trackFile.kf.F = original_F
        """将传感器数据与预测状态进行匹配"""
        matchResults = {}  # {idx: targetID} 传感器数据索引 -> 目标ID
        for targetID, predInfo in predictedStates.items():
            predictedPos = predInfo['position']
            # 找到最接近预测位置的传感器数据
            bestMatchIdx = None
            bestDistance = float('inf')
            for idx, sensorData in enumerate(sensorDataList):
                # 跳过已使用的传感器数据
                if idx in usedSensorData:
                    continue
                # 转换传感器数据到ENU坐标系
                posInENU = TrackFile.body_to_enu(
                    sensorData.range,
                    sensorData.azimuth,
                    sensorData.elevation,
                    self.ownState.position,
                    self.ownAttitude.yaw,
                    self.ownAttitude.pitch,
                    self.ownAttitude.roll
                )
                # 计算欧式距离
                distance = np.linalg.norm(posInENU - predictedPos)
                # 距离阈值（可配置，根据传感器类型和预测时间调整）
                # 阈值 = 基础阈值 + 速度不确定性 * 时间差
                baseThreshold = 20.0  # 基础阈值20米
                dt = currentSensorTime - predInfo['trackFile'].lastUpdate
                velocityUncertainty = 5.0  # 假设速度不确定性为5m/s
                adaptiveThreshold = baseThreshold + velocityUncertainty * dt
                if distance < bestDistance and distance < adaptiveThreshold:
                    bestDistance = distance
                    bestMatchIdx = idx
            # 如果找到匹配
            if bestMatchIdx is not None:
                matchResults[bestMatchIdx] = targetID
                usedSensorData.add(bestMatchIdx)
        """用匹配的传感器数据更新跟踪文件"""
        for idx, targetID in matchResults.items():
            trackFile = predictedStates[targetID]['trackFile']
            sensorData = sensorDataList[idx]
            # 正式更新跟踪文件（包含预测+更新）
            trackFile.update(
                sensorData,
                self.ownState,
                self.ownAttitude
            )
        """处理剩余未匹配的传感器数据（创建新跟踪文件）"""
        for idx, sensorData in enumerate(sensorDataList):
            # 跳过已经使用过的传感器数据
            if idx in usedSensorData:
                continue
            # 生成临时目标ID（使用传感器类型+时间戳+索引）
            tempTargetID = f"Unknown_{sensorType}_{sensorData.timeStamp}_{idx}"
            # 创建新的跟踪文件
            if tempTargetID not in self.trackFiles:
                self.trackFiles[tempTargetID] = {}
            self.trackFiles[tempTargetID][sensorType] = TrackFile(
                tempTargetID,
                sensorType,
                self.sensors[sensorType].covariance
            )
            # 初始化新跟踪文件
            self.trackFiles[tempTargetID][sensorType].update(
                sensorData,
                self.ownState,
                self.ownAttitude
            )

    def _cleanup_stale_tracks(self, maxAge: float = 5.0) -> None:
        """
        清理过时的跟踪文件
        规则：
        1. 如果某个目标的所有跟踪文件都超过maxAge秒未更新，删除该目标
        2. 如果某个跟踪文件超过maxAge秒未更新，删除该跟踪文件
        :param maxAge: 最大允许未更新时间（秒）
        :return: None
        """
        if self.ownState is None:
            return
        currentTime = self.ownState.timeStamp  # 当前时间戳
        targetsToRemove = []  # 待删除的目标ID列表
        for targetID, trackDict in self.trackFiles.items():
            sensorsToRemove = []
            # 检查每个传感器的跟踪文件
            for sensorType, trackFile in trackDict.items():
                # 计算跟踪文件年龄
                age = currentTime - trackFile.lastUpdate  
                if age > maxAge:
                    sensorsToRemove.append(sensorType)
            # 删除过时的传感器跟踪文件
            for sensorType in sensorsToRemove:
                del trackDict[sensorType]
            # 如果所有传感器的跟踪文件都被删除，标记该目标待删除
            if len(trackDict) == 0:
                targetsToRemove.append(targetID)
        # 删除没有任何跟踪文件的目标
        for targetID in targetsToRemove:
            del self.trackFiles[targetID]
    
    @staticmethod
    def mahalanobis_distance(x1, x2, cov1, cov2):
        """
        静态方法，计算马氏距离
        x1, x2: 位置向量（如 [east, north, up]）
        cov1, cov2: 各自的3x3位置协方差
        """
        diff = x1 - x2
        cov = cov1 + cov2  # 误差叠加
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        d2 = diff.T @ inv_cov @ diff
        return np.sqrt(d2)
    
    def _associate_targets(self) -> None:
        """
        对同一目标的多个TrackFile进行协方差加权融合
        :return: None
        """
        for targetID, trackDict in self.trackFiles.items():
            # 收集所有有效的状态估计
            estimates = []
            for sensorType, tf in trackDict.items():
                stateEstimate = tf.get_state_estimate()
                if stateEstimate is not None:
                    estimates.append(stateEstimate)
            # 如果没有有效估计，跳过
            if len(estimates) == 0:
                continue
            # 如果只有一个估计，直接使用
            if len(estimates) == 1:
                fusedEstimate = estimates[0]
            else:
                # 多个估计进行融合
                # 确定参考时间戳（使用最新的）
                refTime = max(e.timeStamp for e in estimates)
                # 过滤掉太旧的估计（超过3秒）
                validEstimates = [e for e in estimates if abs(e.timeStamp - refTime) < 3.0]
                if len(validEstimates) == 0:
                    continue
                elif len(validEstimates) == 1:
                    fusedEstimate = validEstimates[0]
                else:
                    # 协方差加权融合
                    covarianceInverseSum = np.zeros((6, 6))
                    weightedStateSum = np.zeros(6)
                    for estimate in validEstimates:
                        covariance = estimate.covariance
                        stateVector = np.array([
                            estimate.position.east,
                            estimate.position.north,
                            estimate.position.up,
                            estimate.velocity.eastVelocity,
                            estimate.velocity.northVelocity,
                            estimate.velocity.upVelocity
                        ])
                        try:
                            covarianceInverse = np.linalg.inv(covariance)
                        except np.linalg.LinAlgError:
                            covarianceInverse = np.linalg.pinv(covariance)
                        covarianceInverseSum += covarianceInverse
                        weightedStateSum += covarianceInverse @ stateVector
                    try:
                        fusedCovariance = np.linalg.inv(covarianceInverseSum)
                    except np.linalg.LinAlgError:
                        fusedCovariance = np.linalg.pinv(covarianceInverseSum)
                    fusedState = fusedCovariance @ weightedStateSum
                    # 构造融合后的 StateEstimate
                    fusedEstimate = StateEstimate(
                        ID=targetID,
                        position=Position(east=fusedState[0], north=fusedState[1], up=fusedState[2]),
                        velocity=Velocity(
                            eastVelocity=fusedState[3],
                            northVelocity=fusedState[4],
                            upVelocity=fusedState[5]
                        ),
                        covariance=fusedCovariance,
                        timeStamp=refTime
                    )
            # 保存融合结果
            self.targets[targetID] = fusedEstimate


class TRM:
    """TRM类，威胁化解模块"""
    def __init__(self, cfg) -> None:
        """
        初始化TRM对象
        :param cfg: 配置参数对象
        :return: None
        """
        self.horizontalPolicyTable: Dict = self._init_horizontal_policy_table()
        self.verticalPolicyTable: Dict = self._init_vertical_policy_table()
        self.policy: AlgorithmBaseClass = ExampleAlgorithm(cfg)

    def _init_horizontal_policy_table(self) -> dict:
        """
        初始化水平威胁化解策略表
        :return: 水平威胁化解策略表字典
        """
        # 实际应用中从文件加载预先计算的水平威胁化解策略表
        # 这里占位使用
        return {}
    
    def _init_vertical_policy_table(self) -> dict:
        """
        初始化垂直威胁化解策略表
        :return: 垂直威胁化解策略表字典
        """
        # 实际应用中从文件加载预先计算的垂直威胁化解策略表
        # 这里占位使用
        return {}
    
    def resolve_threat(self, ownState: StateEstimate, targetStates: Dict[str, StateEstimate]) -> Tuple[BlendedAction, np.ndarray]:
        """
        生成避让动作和风险评估
        :param ownState: 无人机自身状态
        :param targetStates: 所有入侵者的状态
        :return: 融合动作和风险评估
        """
        actions = self.policy.select_action(ownState, targetStates)  # 生成避让动作
        riskProfile = self.policy.get_risk_profile()  # 风险评估
        return actions, riskProfile




if __name__ == "__main__":
    pass