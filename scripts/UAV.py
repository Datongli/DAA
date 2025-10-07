"""
定义UAV类
"""
from sensor import *
from STMandTRM import STM, TRM, TrackFile


class UAV:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.ID: float| int| str = self.cfg.id  # 无人机ID
        self.ownState: StateEstimate | None = None  # 无人机自身状态估计（位置和速度）
        self.targets: Dict[str: StateEstimate] = {}  # 目标状态估计字典
        self.sensors: Dict[str: Sensor] = {}  # 传感器字典
        self.trackFiles: Dict[str: TrackFile] = {}  # 跟踪文件字典
        self._init_sensors()
        self.stm: STM = STM(self.ID, self.sensors)  # 状态管理模块
        self.trm: TRM = TRM(self.cfg)  # 跟踪管理模块

    def _init_sensors(self) -> None:
        """初始化传感器"""
        for sensor in self.cfg.sensors:
            sensor_cls = SENSOR_REGISTRY.get(sensor)
            if sensor_cls:
                self.sensors[sensor] = sensor_cls(self.cfg)
            else:
                raise ValueError(f"未知传感器类型: {sensor}")
            
    def update(self, data: Dict) -> np.array:
        """
        UAV更新逻辑
        """
        """更新传感器数据"""
        for sensor, data_list in data.items():
            if sensor in self.sensors:
                self.sensors[sensor].get_observations(data_list)
            else:
                raise ValueError(f"未知传感器类型: {sensor}")
        """STM模块逻辑更新"""
        self.ownState, self.targets, self.trackFiles = self.stm.update()
        """TRM模块逻辑更新"""
        actions, riskProfile = self.trm.resolve_threat(self.ownState, self.targets)
        return actions, riskProfile, self.trackFiles


if __name__ == "__main__":
    pass