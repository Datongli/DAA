"""
定义传感器类
"""
from abc import ABC, abstractmethod
from typing import Dict, List
from dataClass import *


class Sensor(ABC):
    """传感器类"""
    def __init__(self, sensorType: str) -> None:
        self.sensorType: str = sensorType  # 传感器类型
        self.data: List[UWBData| RadarData| CloudData| TrackData] = []  # 传感器数据列表
        self.covariance: np.ndarray| None = self._calculate_covariance()  # 传感器协方差矩阵

    @abstractmethod
    def _calculate_covariance(self) -> np.ndarray | None:
        """根据传感器精度计算协方差矩阵"""
        pass

    @abstractmethod
    def get_observations(self, data: List[Dict]) -> None:
        """获取当前时刻的观测数据"""
        pass


class UWBSensor(Sensor):
    """UWB传感器实现"""
    def __init__(self, cfg) -> None:
        # 精度参数 (距离, 方位角, 俯仰角)
        self.accuracy = (cfg.UWB.rangeAcc, 
                         np.deg2rad(cfg.UWB.azimuthAcc), 
                         np.deg2rad(cfg.UWB.elevationAcc))
        super().__init__("UWB")
        
    def _calculate_covariance(self) -> np.ndarray:
        """根据传感器精度计算协方差矩阵"""
        # 简化实现 - 实际应根据传感器特性建模
        return np.diag([self.accuracy[0]**2, 
                       (self.accuracy[1]*10)**2,  # 角度误差放大距离不确定性
                       (self.accuracy[2]*10)**2])

    def get_observations(self, data: List[Dict]) -> None:
        """获取当前时刻的UWB观测数据"""
        uwbData: List[UWBData] = []
        for item in data:
            # 键名必须与UWBData类中的属性名一致
            uwbData.append(UWBData(**item))
        self.data = uwbData


class RadarSensor(Sensor):
    """雷达传感器实现"""
    def __init__(self, cfg) -> None:
        # 精度参数 (距离, 方位角, 俯仰角)
        self.accuracy = (cfg.Radar.rangeAcc, 
                         np.deg2rad(cfg.Radar.azimuthAcc), 
                         np.deg2rad(cfg.Radar.elevationAcc))
        super().__init__("Radar")

    def _calculate_covariance(self) -> np.ndarray:
        """根据传感器精度计算协方差矩阵"""
        # 简化实现 - 实际应根据传感器特性建模
        return np.diag([self.accuracy[0]**2, 
                       (self.accuracy[1]*10)**2,  # 角度误差放大距离不确定性
                       (self.accuracy[2]*10)**2])
    
    def get_observations(self, data: List[Dict]) -> None:
        """获取当前时刻的雷达观测数据"""
        radarData: List[RadarData] = []
        for item in data:
            # 键名必须与RadarData类中的属性名一致
            radarData.append(RadarData(**item))
        self.data = radarData


class TrackSensor(Sensor):
    """Track传感器实现"""
    def __init__(self, cfg) -> None:
        # 精度参数 (东坐标，北坐标，高度坐标，东向速度，北向速度，垂直速度)
        self.accuracy = (cfg.Track.eastAcc, cfg.Track.northAcc, cfg.Track.upAcc, 
                         cfg.Track.eastVelocityAcc, cfg.Track.northVelocityAcc, cfg.Track.upVelocityAcc)
        super().__init__("Track")

    def _calculate_covariance(self) -> np.ndarray:
        """根据传感器精度计算协方差矩阵"""
        return np.diag(np.array(self.accuracy) ** 2)
    
    def get_observations(self, data: List[Dict]) -> None:
        """获取当前时刻的雷达观测数据"""
        trackData: List[TrackData] = []
        for item in data:
            # 键名必须与TrackData类中的属性名一致
            item['coordinate'] = Position(**item['coordinate'])
            item['velocity'] = Velocity(**item['velocity'])
            trackData.append(TrackData(**item))
        self.data = trackData


class CloudSensor(Sensor):
    """云平台传感器实现"""
    def __init__(self, cfg) -> None:
        # 精度参数 (东坐标，北坐标，高度坐标，东向速度，北向速度，垂直速度)
        self.accuracy = (cfg.Cloud.eastAcc, cfg.Cloud.northAcc, cfg.Cloud.upAcc, 
                         cfg.Cloud.eastVelocityAcc, cfg.Cloud.northVelocityAcc, cfg.Cloud.upVelocityAcc)
        super().__init__("Cloud")

    def _calculate_covariance(self) -> np.ndarray:
        """根据传感器精度计算协方差矩阵"""
        return np.diag(np.array(self.accuracy) ** 2)
    
    def get_observations(self, data: List[Dict]) -> None:
        """获取当前时刻的云平台观测数据"""
        cloudData: List[CloudData] = []
        for item in data:
            # 键名必须与CloudData类中的属性名一致
            item['coordinate'] = Position(**item['coordinate'])
            item['velocity'] = Velocity(**item['velocity'])
            cloudData.append(CloudData(**item))
        self.data = cloudData



"""传感器注册表"""
SENSOR_REGISTRY = {
    "UWB": UWBSensor,
    "Radar": RadarSensor,
    "Track": TrackSensor,
    "Cloud": CloudSensor,
}


if __name__ == "__main__":
    pass