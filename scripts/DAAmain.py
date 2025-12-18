from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
try:
    import hydra
    from omegaconf import OmegaConf, DictConfig
except ImportError:
    hydra = None
    from omegaconf import OmegaConf, DictConfig
import os
from UAV import UAV
from STMandTRM import TrackFile
from dataProcessLocal import merge_by_timeStamp, utm_to_wgs84


# 保证根目录在 sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
CFG_DIR = ROOT_DIR / "cfg"


def load_cfg_service() -> DictConfig:
    """
    服务模式：手动加载main.yaml和UAV.yaml
    """
    mainPath = CFG_DIR / "main.yaml"
    if not mainPath.exists():
        raise FileNotFoundError(f"缺少配置文件: {mainPath}")
    cfg = OmegaConf.load(mainPath)
    uavPath = CFG_DIR / "UAV.yaml"
    if not uavPath.exists():
        raise FileNotFoundError(f"缺少配置文件: {uavPath}")
    uavCfg = OmegaConf.load(uavPath)
    cfg = OmegaConf.merge(cfg, uavCfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def build_uav(cfg: DictConfig) -> list[UAV]:
    """
    根据配置文件构建无人机
    :param cfg: 配置文件
    :return: 无人机列表
    """
    if "UAVs" not in cfg or cfg.UAVs is None:
        return None
    return UAV(cfg.UAVs)


def DAAmain() -> dict:
    """
    DAA主函数
    返回一个json数据格式
    """
    # 获取配置
    cfg = load_cfg_service()
    # 读取数据
    data = merge_by_timeStamp(cfg)
    # 获取无人机对象
    uav: UAV = build_uav(cfg)
    if uav is None:
        return {"error": "No UAVs found in configuration"}
    timeLine = []
    for timeStamp, value in data.items():
        # 确保所有必要的键都存在
        if "IntruderReal" not in value:
            value["IntruderReal"] = []
        if "Cloud" not in value:
            value["Cloud"] = []
        if "Radar" not in value:
            value["Radar"] = []
        if "UWB" not in value:
            value["UWB"] = []
        actions, riskProfile, uavTrackFiles = uav.update(value)
        """检查框架有效性使用"""
        print({
            "timeStamp": timeStamp,
            "coordinate": {
                "east": uav.ownState.position.east,
                "north": uav.ownState.position.north,
                "up": uav.ownState.position.up,
            },
            "velocity": {
                "eastVelocity": uav.ownState.velocity.eastVelocity,
                "northVelocity": uav.ownState.velocity.northVelocity,
                "upVelocity": uav.ownState.velocity.upVelocity,
            },
            "actions":{
                "horizontal": actions.horizontal.name if actions and actions.horizontal else "NONE",
                "vertical": actions.vertical.name if actions and actions.vertical else "NONE",
            },
            "attitude": {
                "yaw": float(getattr(uav.stm.ownAttitude, "yaw", 0.0)),
                "pitch": float(getattr(uav.stm.ownAttitude, "pitch", 0.0)),
                "roll": float(getattr(uav.stm.ownAttitude, "roll", 0.0)),
            }
        })
        if not uavTrackFiles:
            # 计算按照原速度移动的下一个点位
            originNextPointEast = uav.ownState.position.east + uav.ownState.velocity.eastVelocity * uav.dt
            originNextPointNorth = uav.ownState.position.north + uav.ownState.velocity.northVelocity * uav.dt
            originNextPointUp = uav.ownState.position.up + uav.ownState.velocity.upVelocity * uav.dt
            originNextWGS84 = utm_to_wgs84(
                originNextPointEast,
                originNextPointNorth,
                originNextPointUp,
                cfg.utmZone
            )
            # 转换自身坐标到WGS84
            ownWGS84 = utm_to_wgs84(
                uav.ownState.position.east,
                uav.ownState.position.north,
                uav.ownState.position.up,
                cfg.utmZone
            )
            timeLine.append({
                # 时间戳
                "timeStamp": timeStamp,
                # 轨迹文件
                "riskProfile": [float(x) for x in (riskProfile if riskProfile is not None else [])],
                # 动作
                "actions": {
                    "H": actions.horizontal.name if actions and actions.horizontal else "NONE",
                    "V": actions.vertical.name if actions and actions.vertical else "NONE"
                },
                # 当前点位
                "currentPoint": {
                    "latitude": ownWGS84["latitude"],
                    "longitude": ownWGS84["longitude"],
                    "height": ownWGS84["height"]
                },
                # 按照原速度移动的下一个点位
                "originallyNextPoint": {
                    "latitude": originNextWGS84["latitude"],
                    "longitude": originNextWGS84["longitude"],
                    "height": originNextWGS84["height"]
                },
                # 状态提示
                "status": "OK"
            })
        else:
            # 有入侵者时，每个目标单独一条
            for trackID, trackFiles in uavTrackFiles.items():
                entry = {
                    "timeStamp": int(timeStamp),
                    "riskProfile": [float(x) for x in (riskProfile if riskProfile is not None else [])],
                    "actions": {
                        "H": actions.horizontal.name if actions and actions.horizontal else "NONE",
                        "V": actions.vertical.name if actions and actions.vertical else "NONE"
                    },
                    "trackID": trackID
                }
                # 各传感器观测
                for sensorType in ["Cloud", "Radar", "UWB"]:
                    if sensorType in trackFiles:
                        tf: TrackFile = trackFiles[sensorType]
                        # 获取状态估计对象
                        stateEstimate = tf.get_state_estimate()
                        if stateEstimate and stateEstimate.position:
                            # 转换为WGS84
                            wgs84 = utm_to_wgs84(float(stateEstimate.position.east), 
                                                 float(stateEstimate.position.north), 
                                                 float(stateEstimate.position.up), 
                                                 cfg.utmZone)
                            entry[sensorType] = {
                                "longitude": wgs84["longitude"],
                                "latitude": wgs84["latitude"],
                                "height": wgs84["height"]
                            }
                # 融合结果
                fused = uav.targets.get(trackID)
                if fused:
                    # 转换为WGS84
                    wgs84 = utm_to_wgs84(float(fused.position.east), float(fused.position.north), float(fused.position.up), cfg.utmZone)
                    entry["comprehensive"] = {
                        "longitude": wgs84["longitude"],
                        "latitude": wgs84["latitude"],
                        "height": wgs84["height"]
                    }
                # 修改风险提示与融合，若uwb不存在则全为0
                if "UWB" not in trackFiles:
                    entry["riskProfile"] = [0.0] * len(entry["riskProfile"])
                    entry["comprehensive"] = []
            # 转换自身坐标到WGS84
            ownWGS84 = utm_to_wgs84(
                uav.ownState.position.east,
                uav.ownState.position.north,
                uav.ownState.position.up,
                cfg.utmZone
            )
            entry["currentPoint"] = {
                "longitude": ownWGS84["longitude"],
                "latitude": ownWGS84["latitude"],
                "height": ownWGS84["height"]
            }
            # 按照原速度移动的下一个点位
            originNextPointEast = uav.ownState.position.east + uav.ownState.velocity.eastVelocity * uav.dt
            originNextPointNorth = uav.ownState.position.north + uav.ownState.velocity.northVelocity * uav.dt
            originNextPointUp = uav.ownState.position.up + uav.ownState.velocity.upVelocity * uav.dt
            originNextWGS84 = utm_to_wgs84(
                originNextPointEast,
                originNextPointNorth,
                originNextPointUp,
                cfg.utmZone
            )
            entry["originallyNextPoint"] = {
                "latitude": originNextWGS84["latitude"],
                "longitude": originNextWGS84["longitude"],
                "height": originNextWGS84["height"]
            }
            # 添加状态提示
            hasAction = entry["actions"]["H"] != "NONE" or entry["actions"]["V"] != "NONE"
            hasUWB = "UWB" in entry
            hasRadar = "Radar" in entry
            hasCloud = "Cloud" in entry
            if hasAction:
                # 只要有动作就是警告状态
                entry["status"] = "warning"
            elif hasUWB or hasRadar:
                # 有UWB或Radar数据但没有动作，观察状态
                entry["status"] = "watch"
            elif hasCloud:
                # 只有Cloud数据且没有动作，正常状态
                entry["status"] = "OK"
            else:
                # 其他情况
                entry["status"] = "unknown"
            timeLine.append(entry)
    return {"timeline": timeLine}


if __name__ == "__main__":
    """
    数据预处理后，数据格式为：
    key=时间戳，value=字典，字典的key为传感器类型，value为该传感器类型下的所有数据
    示例数据如下
    key=0, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 0, 'coordinate': {'east': 503757.215586069, 'north': 4370861.992682273, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(1.23), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [], 'Radar': [], 'UWB': []}
    key=1, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 1, 'coordinate': {'east': 503757.21498870925, 'north': 4370863.224564647, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(0.59), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [], 'Radar': [], 'UWB': []}
    key=2, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 2, 'coordinate': {'east': 503757.2147034834, 'north': 4370863.812760737, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(1.65), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [], 'Radar': [], 'UWB': []}
    key=3, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 3, 'coordinate': {'east': 503757.2139016218, 'north': 4370865.466368616, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(1.18), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [], 'Radar': [], 'UWB': []}
    key=4, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 4, 'coordinate': {'east': 503757.21333116986, 'north': 4370866.642760799, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(0.9), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [], 'Radar': [], 'UWB': []}
    key=5, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 5, 'coordinate': {'east': 503757.21289525833, 'north': 4370867.541701999, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(0.75), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [], 'Radar': [], 'UWB': []}
    key=6, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 6, 'coordinate': {'east': 503757.2125293079, 'north': 4370868.296368684, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(1.87), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [], 'Radar': [], 'UWB': []}
    key=7, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 7, 'coordinate': {'east': 503757.2116251947, 'north': 4370870.160839326, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(0.74), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [{'UAVID': 'ANSI/CTA-2063-B', 'timeStamp': 7, 'coordinate': {'east': 503748.6041152361, 'north': 4370886.814831649, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.8730079720117836), 'northVelocity': np.float64(-1.1173992486143316), 'upVelocity': 0}}], 'Radar': [{'timeStamp': 7, 'range': 18.747, 'azimuth': np.float64(-0.47699848457005023), 'elevation': np.float64(0.0)}], 'UWB': [{'timeStamp': 7, 'range': 18.8, 'azimuth': np.float64(-0.49445177708999355), 'elevation': np.float64(0.0)}]}
    key=8, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 8, 'coordinate': {'east': 503757.21126462566, 'north': 4370870.904407976, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(1.4), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [{'UAVID': 'ANSI/CTA-2063-B', 'timeStamp': 8, 'coordinate': {'east': 503749.47320030705, 'north': 4370885.694349883, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.8016176108385282), 'northVelocity': np.float64(-1.0637824993829943), 'upVelocity': 0}}], 'Radar': [{'timeStamp': 8, 'range': 16.692, 'azimuth': np.float64(-0.48258353817643207), 'elevation': np.float64(0.0)}], 'UWB': [{'timeStamp': 8, 'range': 16.4, 'azimuth': np.float64(-0.48432886742842646), 'elevation': np.float64(0.0)}]}
    key=9, value={'Track': [{'UAVID': 'ANSI/CTA-2063-A', 'timeStamp': 9, 'coordinate': {'east': 503757.21058654046, 'north': 4370872.302760963, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.0), 'northVelocity': np.float64(1.1), 'upVelocity': 0}, 'yaw': np.float64(0.0), 'pitch': np.float64(0.0), 'roll': np.float64(0.0)}], 'Cloud': [{'UAVID': 'ANSI/CTA-2063-B', 'timeStamp': 9, 'coordinate': {'east': 503750.2734632952, 'north': 4370884.62932512, 'up': 10}, 'velocity': {'eastVelocity': np.float64(0.894264275681819), 'northVelocity': np.float64(-1.1043244112303556), 'upVelocity': 0}}], 'Radar': [{'timeStamp': 9, 'range': 14.145, 'azimuth': np.float64(-0.5126032013107346), 'elevation': np.float64(0.0)}], 'UWB': [{'timeStamp': 9, 'range': 14.5, 'azimuth': np.float64(-0.5059709501531561), 'elevation': np.float64(0.0)}]}    
    """
    daaResult = DAAmain()
    print("=" * 200)
    for result in daaResult["timeline"]:
        print(result)