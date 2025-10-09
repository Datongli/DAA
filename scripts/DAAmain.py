from __future__ import annotations
import sys
from pathlib import Path
import json
try:
    import hydra
    from omegaconf import OmegaConf, DictConfig
except ImportError:
    hydra = None
    from omegaconf import OmegaConf, DictConfig
import os
from UAV import UAV
from dataProcessLocal import merge_by_timeStamp, utm_to_wgs84
from QLearningAlgorithm import QLearningAlgorithm, train_qlearning


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
    # 获取无人机列表
    uav = build_uav(cfg)
    if uav is None:
        return {"error": "No UAVs found in configuration"}
    timeLine = []
    for timeStamp, value in data.items():
        actions, riskProfile, uavTrackFiles = uav.update(value)
        if not uavTrackFiles:
            timeLine.append({
                "timeStamp": timeStamp,
                "riskProfile": [float(x) for x in (riskProfile if riskProfile is not None else [])],
                "actions": {
                    "H": actions.horizontal.name if actions and actions.horizontal else "NONE",
                    "V": actions.vertical.name if actions and actions.vertical else "NONE"
                }
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
                        tf = trackFiles[sensorType]
                        # 转换为WGS84
                        wgs84 = utm_to_wgs84(float(tf.kf.x[0]), float(tf.kf.x[1]), float(tf.kf.x[2]), cfg.utmZone)
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
                timeLine.append(entry)
    daaResult = {"timeline": timeLine}

    # === Q-learning集成部分 ===
    import json
    range_path = ROOT_DIR / "data" / "UWB.json"
    fc_path = ROOT_DIR / "data" / "FlightControl.json"
    with open(range_path, "r", encoding="utf-8") as f:
        range_data_raw = json.load(f)
    with open(fc_path, "r", encoding="utf-8") as f:
        fc_data_raw = json.load(f)
    range_data = [d for d in range_data_raw if d["timestamp"] in [7,8,9]]
    fc_data = [d for d in fc_data_raw if d["timestamp"] in [7,8,9]]

    # 按时间戳对齐，提取 sector_probs、yaw、pitch、roll、altitudes、longitudes、latitudes
    sector_probs = []
    yaws = []
    pitchs = []
    rolls = []
    altitudes = []
    longitudes = []
    latitudes = []
    range_data_aligned = []
    for range_item in range_data:
        ts = range_item["timestamp"]
        entry = next((e for e in daaResult["timeline"] if e.get("timeStamp", 0) == ts), None)
        fc_entry = next((f for f in fc_data if f["timestamp"] == ts), None)
        if entry and "riskProfile" in entry and "comprehensive" in entry and fc_entry:
            sector_probs.append(entry["riskProfile"][0] if entry["riskProfile"] else 0)
            yaws.append(fc_entry["yaw"])
            pitchs.append(fc_entry["pitch"])
            rolls.append(fc_entry["roll"])
            altitudes.append(entry["comprehensive"].get("height", 0))
            longitudes.append(entry["comprehensive"].get("longitude", 0))
            latitudes.append(entry["comprehensive"].get("latitude", 0))
            range_data_aligned.append(range_item)

    # 确保长度一致且至少有两个数据点
    n = min(len(sector_probs), len(yaws), len(pitchs), len(rolls), len(altitudes), len(range_data_aligned), len(longitudes), len(latitudes))
    if n < 2:
        return daaResult  # 数据不足不训练


    # Q-learning训练
    q_algo, actions = train_qlearning(
        daaResult,
        range_data_aligned,
        sector_probs,
        yaws,
        pitchs,
        rolls,
        altitudes,
        longitudes,
        latitudes
    )

    # 打印Q表
    print("Q表内容:")
    for key, value in q_algo.q_table.items():
        print(f"State-Action: {key}, Q-value: {value}")

    # 写入动作到 daaResult
    idx = 0
    for range_item in range_data_aligned:
        ts = range_item["timestamp"]
        for entry in daaResult["timeline"]:
            if entry.get("timeStamp", 0) == ts and idx < len(actions):
                entry["QAction"] = actions[idx]  # 二维动作元组
                idx += 1

    return daaResult

if __name__ == "__main__":
    result = DAAmain()
    import json
    for entry in result["timeline"]:
        print(json.dumps(entry, ensure_ascii=False))
