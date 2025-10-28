"""
数据处理脚本
从服务端进行数据读取
"""
import pandas as pd
from pyproj import Transformer
# import hydra
import os
from dataclasses import *
import numpy as np


# FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
# @hydra.main(config_path=FILE_PATH, config_name="main", version_base=None)
def merge_by_timeStamp(cfg):
    """
    读取所有json文件中的数据，根据时间戳合并
    :param cfg: 配置参数
    :return: 合并后的数据
    """
    # 获取项目根目录的绝对路径
    projectRoot = os.path.dirname(os.path.dirname(__file__))
    datadir = os.path.join(projectRoot, "data")
    flightControlData = pd.read_json(os.path.join(datadir, "FlightControl.json"))
    trackData = pd.read_json(os.path.join(datadir, "Track.json"))
    cloudData = pd.read_json(os.path.join(datadir, "Cloud.json"))
    radarData = pd.read_json(os.path.join(datadir, "Radar.json"))
    uwbData = pd.read_json(os.path.join(datadir, "UWB.json"))
    intrudersRealData = pd.read_json(os.path.join(datadir, "IntruderReal.json"))
    # 建立时间戳索引
    merged = {}
    # 读取Track文件
    for item in trackData.to_dict(orient="records"):
        process_track_data(merged, item, cfg.utmZone)
    # 读取FlightControl文件，与Track数据合并
    for item in flightControlData.to_dict(orient="records"):
        process_flight_data(merged, item)
    # 读取Cloud文件
    for item in cloudData.to_dict(orient="records"):
        process_cloud_data(merged, item, cfg.utmZone)
    # 读取入侵者真实数据
    for item in intrudersRealData.to_dict(orient="records"):
        process_intruder_real_data(merged, item, cfg.utmZone)
    """现在radar和uwb数据由模拟构成"""
    # 读取Radar文件
    for item in radarData.to_dict(orient="records"):
        process_radar_data(merged, item)
    # 读取UWB文件
    for item in uwbData.to_dict(orient="records"):
        process_uwb_data(merged, item)
    return merged
        

def coordinate_transformation(item: dict, utmZone: str) -> tuple[dict, dict]:
    """ 
    将经纬度转换为东北天坐标系下的坐标
    :param item: 包含经纬度、高度、方向、水平速度、垂直速度的字典
    :param utmZone: UTM区域号，例如 '50N' / '50S'
    :return: 包含东北天坐标和速度的元组
    """
    # 提取UTM区域号和半球
    zoneNumber = utmZone[:-1]  # 分离数字部分和半球标识
    hemisphere = 'north' if utmZone[-1].upper() == 'N' else 'south'
    # 坐标转换到UTM
    transformer = Transformer.from_crs("EPSG:4326",
                                    f"+proj=utm +zone={zoneNumber} +{hemisphere} +ellps=WGS84")
    east, north = transformer.transform(item["location"]["latitude"], item["location"]["longitude"])
    up = item["location"]["relativeHeight"]
    headingRad = np.radians(item["location"]["direction"])
    eastV = item["location"]["horizontalSpeed"] * np.sin(headingRad)
    northV = item["location"]["horizontalSpeed"] * np.cos(headingRad)
    upV = item["location"]["verticalSpeed"]
    coordinate = {"east": east, "north": north, "up": up}
    velocity = {"eastVelocity": eastV, "northVelocity": northV, "upVelocity": upV}
    return coordinate, velocity


def utm_to_wgs84(east: float, north: float, height: float, utmZone: str) -> dict:
    """
    UTM (east, north, height) -> WGS84经纬高
    utmZone: 例如 '50N' / '50S'
    返回: {"latitude": lat, "longitude": lon, "height": height}
    :param east: UTM东坐标
    :param north: UTM北坐标
    :param height: 高度
    :param utmZone: UTM区域号，例如 '50N' / '50S'
    :return: WGS84经纬高字典
    """
    zoneNumber = utmZone[:-1]
    hemisphereFlag = utmZone[-1].upper()
    hemisphere = 'north' if hemisphereFlag == 'N' else 'south'
    utmCrs = f"+proj=utm +zone={zoneNumber} +{hemisphere} +ellps=WGS84"
    transformer = Transformer.from_crs(utmCrs, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(east, north)  # 注意返回经度、纬度
    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "height": float(height)
    }


def process_track_data(merged: dict, item: dict, utmZone: str) -> None:
    """
    处理Track数据
    :param merged: 合并后的数据
    :param item: Track数据项
    :param utmZone: UTM区域号，例如 '50N' / '50S'
    :return: None
    """
    timeStamp = item["location"]["timestamp"]  # 时间戳
    if timeStamp not in merged:
        merged[timeStamp] = {"Track": [], "Cloud": [], "Radar": [], "UWB": []}
    UAVID = item["aircraft"]["uaIdType"]  # UAVID
    coordinate, velocity = coordinate_transformation(item, utmZone)
    # 创建Track数据
    merged[timeStamp]["Track"].append({
        "UAVID": UAVID,
        "timeStamp": timeStamp,
        "coordinate": coordinate,
        "velocity": velocity,
    })


def process_flight_data(merged: dict, item: dict) -> None:
    """
    处理FlightControl数据，与Track数据合并
    :param merged: 合并后的数据
    :param item: FlightControl数据项
    :return: None
    """
    timeStamp = item["timestamp"]  # 时间戳
    merged[timeStamp]["Track"][0]["yaw"] = np.deg2rad(item["yaw"])
    merged[timeStamp]["Track"][0]["pitch"] = np.deg2rad(item["pitch"])
    merged[timeStamp]["Track"][0]["roll"] = np.deg2rad(item["roll"])


def process_cloud_data(merged: dict, item: dict, utmZone: str) -> None:
    """
    处理Cloud数据
    :param merged: 合并后的数据
    :param item: Cloud数据项
    :param utmZone: UTM区域号，例如 '50N' / '50S'
    :return: None
    """
    timeStamp = item["location"]["timestamp"]  # 时间戳
    UAVID = item["aircraft"]["uaIdType"]  # UAVID
    coordinate, velocity = coordinate_transformation(item, utmZone)
    # 创建Cloud数据
    merged[timeStamp]["Cloud"].append({
        "UAVID": UAVID,
        "timeStamp": timeStamp,
        "coordinate": coordinate,
        "velocity": velocity,
    })


def process_intruder_real_data(merged: dict, item: dict, utmZone: str) -> None:
    """
    处理入侵者真实轨迹
    :param merged: 合并后的数据
    :param item: 入侵者真实轨迹数据项
    :param utmZone: UTM区域号，例如 '50N' / '50S'
    :return: None
    """
    timeStamp = item["location"]["timestamp"]  # 时间戳
    if timeStamp not in merged:
        merged[timeStamp] = {"Track": [], "Cloud": [], "IntruderReal": [], "Radar": [], "UWB": []}
    # 确保IntruderReal键存在
    if "IntruderReal" not in merged[timeStamp]:
        merged[timeStamp]["IntruderReal"] = []
    UAVID = item["aircraft"]["uaIdType"]  
    coordinate, velocity = coordinate_transformation(item, utmZone)
    # 创建IntruderReal数据
    merged[timeStamp]["IntruderReal"].append({
        "UAVID": UAVID,
        "timeStamp": timeStamp,
        "coordinate": coordinate,
        "velocity": velocity,
    })


def process_radar_data(merged: dict, item: dict) -> None:
    """
    处理Radar数据
    :param merged: 合并后的数据
    :param item: Radar数据项
    :return: None
    """
    timeStamp = item["timestamp"]  # 时间戳
    merged[timeStamp]["Radar"].append({
        "timeStamp": timeStamp,
        "range": item["range"],
        "azimuth": np.deg2rad(item["azimuth"]),
        "elevation": np.deg2rad(item["elevation"]),
    })


def process_uwb_data(merged: dict, item: dict) -> None:
    """
    处理UWB数据
    :param merged: 合并后的数据
    :param item: UWB数据项
    :return: None
    """
    timeStamp = item["timestamp"]  # 时间戳
    merged[timeStamp]["UWB"].append({
        "timeStamp": timeStamp,
        "range": item["range"],
        "azimuth": np.deg2rad(item["azimuth"]),
        "elevation": np.deg2rad(item["elevation"]),
    })



if __name__ == "__main__":
    merge_by_timeStamp()