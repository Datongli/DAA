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
import time
# 导入我们刚刚写的服务类
from DAA_RL_Service import DAAReasoner

# 保证根目录在 sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
CFG_DIR = ROOT_DIR / "cfg"
DATA_DIR = ROOT_DIR / "data"  # <--- 确保定义了 DATA_DIR

# --- 关键修复：定义全局变量 ---
_REASONER = None
# ---------------------------

def load_cfg_service() -> DictConfig:
    """服务模式：手动加载main.yaml和UAV.yaml"""
    mainPath = CFG_DIR / "main.yaml"
    if not mainPath.exists():
        raise FileNotFoundError(f"缺少配置文件: {mainPath}")
    cfg = OmegaConf.load(mainPath)
    
    uavPath = CFG_DIR / "UAV.yaml"
    if not uavPath.exists():
        # 这里如果文件不存在，可能要给个空配置或跳过，视业务而定
        print(f"Warning: {uavPath} not found.")
    else:
        uavCfg = OmegaConf.load(uavPath)
        cfg = OmegaConf.merge(cfg, uavCfg)
        
    OmegaConf.set_struct(cfg, False)
    return cfg

def build_uav(cfg: DictConfig) -> list[UAV]:
    if "UAVs" not in cfg or cfg.UAVs is None:
        return None
    return UAV(cfg.UAVs)

def get_reasoner():
    global _REASONER
    if _REASONER is None:
        _REASONER = DAAReasoner()
    return _REASONER

def DAAmain() -> dict:
    """
    DAA主函数
    返回一个json数据格式
    """
    start_time = time.time()
    
    # 获取推理引擎
    engine = get_reasoner()
    
    # 指定输入文件
    own_file = DATA_DIR / "FlightControl.json"
    
    # 优先读取真实传感器 Track.json (如果有)，如果没有再读模拟的 IntruderReal.json
    intr_file = DATA_DIR / "IntruderReal.json"
    if not intr_file.exists() and (DATA_DIR / "Track.json").exists():
        intr_file = DATA_DIR / "Track.json"

    # 执行推理
    result = engine.make_decision(own_file, intr_file)
    
    # 构造标准返回格式
    output = {
        "timestamp": start_time,
        "status": "success" if result["valid"] else "warning",
        "guidance": {
            "yaw_rate": float(result["yaw_rate_cmd"]), 
            "maneuver_text": result["maneuver"]
        },
        "debug": {
            "msg": result["msg"],
            "rl_state": result["rl_state"],
            "action_code": result["action_code"]
        }
    }
    
    # 打印日志
    print(f"[DAA] {result['maneuver']} (YawRate: {result['yaw_rate_cmd']} deg/s) | Msg: {result['msg']}")
    
    return output

if __name__ == "__main__":
    print("-" * 30)
    print(">>> 开始 DAAmain 本地测试")
    print("-" * 30)
    
    # 1. 确保目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. 检查文件是否存在
    fc_path = DATA_DIR / "FlightControl.json"
    track_path = DATA_DIR / "Track.json"
    intr_path = DATA_DIR / "IntruderReal.json"
    
    print(f"检查 FlightControl.json: {fc_path.exists()} ({fc_path})")
    print(f"检查 Track.json: {track_path.exists()} ({intr_path})")
    print(f"检查 IntruderReal.json: {intr_path.exists()} ({intr_path})")
    
    # 3. 强制指定读取 Track.json (为了验证你的 Track.json)
    # 如果 IntruderReal 存在，脚本默认会读它。为了测试，我们先把它改名或删掉，或者在代码里强制指定。
    # 这里我们临时通过重命名来屏蔽 IntruderReal.json，确保测试 Track.json
    temp_backup = None
    if intr_path.exists():
        print("发现旧的 IntruderReal.json，暂时重命名以强制使用 Track.json...")
        temp_backup = intr_path.with_suffix(".json.bak")
        intr_path.rename(temp_backup)
        
    try:
        print("正在调用 DAAmain()...")
        res = DAAmain()
        print("\n>>> DAAmain 返回结果:")
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"运行出错: {e}")
    finally:
        # 恢复由于测试而重命名的文件
        if temp_backup and temp_backup.exists():
            print("恢复 IntruderReal.json...")
            temp_backup.rename(intr_path)

    print("-" * 30)
    print(">>> 测试结束")
    print("-" * 30)