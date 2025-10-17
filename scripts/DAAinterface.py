"""
DAA服务接口
用于和DAA算法进行交互
"""
from flask import Flask, request, jsonify
import yaml, json
from DAAmain import DAAmain
from pathlib import Path
from typing import Any, Dict
app = Flask(__name__)


# 基础目录
BASE_DIR = Path(__file__).resolve().parent.parent
# 配置目录
CFG_DIR = BASE_DIR / "cfg"
# 数据目录
DATA_DIR = BASE_DIR / "data"
# 传感器数据文件
SENSOR_FILE_MAP = {
    "Cloud": "Cloud.json",
    "Radar": "Radar.json",
    "UWB": "UWB.json",
    "Track": "Track.json",
    "FlightControl": "FlightControl.json"
}


def safe_write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """
    安全地写入yaml文件
    :param path: 文件路径
    :param data: 要写入的数据
    :return: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def merge_yaml(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归浅合并：new 覆盖 old
    :param old: 旧数据
    :param new: 新数据
    :return: 合并后的数据
    """
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(old.get(k), dict):
            old[k] = merge_yaml(old[k], v)
        else:
            old[k] = v
    return old


def overwrite_configs(configPayload: Dict[str, Any]) -> None:
    """
    覆盖写入配置文件
    :param configPayload: 配置数据
    :return: None
    """
    if not configPayload:
        return
    if "main" in configPayload:
        main_path = CFG_DIR / "main.yaml"
        if main_path.exists():
            with main_path.open("r", encoding="utf-8") as f:
                old = yaml.safe_load(f) or {}
        else:
            old = {}
        merged = merge_yaml(old, configPayload["main"])
        safe_write_yaml(main_path, merged)
    if "UAVs" in configPayload:
        uav_path = CFG_DIR / "UAV.yaml"
        if uav_path.exists():
            with uav_path.open("r", encoding="utf-8") as f:
                old = yaml.safe_load(f) or {}
        else:
            old = {}
        merged = merge_yaml(old, configPayload["UAVs"])
        safe_write_yaml(uav_path, merged)


def overwrite_sensor_files(sensorsPayload: Dict[str, Any]) -> None:
    """
    覆盖写入传感器数据文件
    :param sensorsPayload: 传感器数据
    :return: None
    """
    if not sensorsPayload:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, fname in SENSOR_FILE_MAP.items():
        if key in sensorsPayload:
            path = DATA_DIR / fname
            with path.open("w", encoding="utf-8") as f:
                json.dump(sensorsPayload[key], f, ensure_ascii=False, indent=2)


@app.route("/health", methods=["GET"])
def health():
    """
    健康检查接口
    """
    return jsonify({"status": "healthy"})


@app.route("/daa/run", methods=["POST"])
def run_daa():
    """
    运行DAA算法
    """
    try:
        payLoad = request.get_json(force=True)
        if payLoad is None:
            return jsonify({"status": "error", "msg": "payload is None"}), 400
        # 覆盖写入配置文件
        overwrite_configs(payLoad.get("config", {}))
        # 覆盖写入传感器数据文件
        overwrite_sensor_files(payLoad.get("sensors", {}))
        # 运行DAA算法
        res = DAAmain()
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500
    

if __name__ == "__main__":
    app.run()