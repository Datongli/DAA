import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# 从你的算法文件中导入必要的组件
# 确保 QLearningAlgorithm.py 和本文件在同一目录下
from QLearningAlgorithm import (
    QLearningAgent, 
    QTABLE_PATH, 
    ACTIONS,
    in_front_fov,
    bin_theta, 
    bin_h, 
    bin_dist, 
    bin_heading
)

class DAAReasoner:
    """RL 推理服务：加载模型并根据状态作出决策"""
    
    def __init__(self):
        self.agent = QLearningAgent()
        self.load_model()

    def load_model(self):
        """加载训练好的 Q 表"""
        if QTABLE_PATH.exists():
            try:
                self.agent.Q = np.load(QTABLE_PATH)
                print(f"[DAA_RL] Successfully loaded Q-table from: {QTABLE_PATH}")
                print(f"[DAA_RL] Q-table shape: {self.agent.Q.shape}")
            except Exception as e:
                print(f"[DAA_RL] Error loading Q-table: {e}")
        else:
            print(f"[DAA_RL] WARNING: Q-table not found at {QTABLE_PATH}. Predictions will be random!")

    def _parse_ownship(self, data: Dict) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        解析本无人机数据 (FlightControl.json)
        返回: (pos_vec3, yaw_deg, vel_vec2)
        """
        if not data:
            return np.zeros(3), 0.0, np.zeros(2)
            
        try:
            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            z = float(data.get('z', 0.0))
            yaw = float(data.get('psi', data.get('yaw', 90.0))) 
            
            # 解析速度
            vx = float(data.get('vx', data.get('ve', 0.0)))
            vy = float(data.get('vy', data.get('vn', 0.0)))
        except (ValueError, TypeError):
            return np.zeros(3), 0.0, np.zeros(2)

        return np.array([x, y, z], dtype=float), yaw, np.array([vx, vy], dtype=float)

    def _parse_intruder(self, data: Any) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        解析入侵者数据 (IntruderReal.json 或 Track.json)
        返回: (pos_vec3, yaw_deg, vel_vec2)
        """
        target = None
        if isinstance(data, list):
            if len(data) > 0:
                target = data[0]
        elif isinstance(data, dict):
            target = data
            
        if not target:
            return None

        try:
            x = float(target.get('x', 0.0))
            y = float(target.get('y', 0.0))
            z = float(target.get('z', 0.0))
            
            vx = float(target.get('vx', target.get('ve', 0.0)))
            vy = float(target.get('vy', target.get('vn', 0.0)))
            
            if 'yaw' in target:
                yaw = float(target['yaw'])
            else:
                if abs(vx) < 1e-3 and abs(vy) < 1e-3:
                    yaw = 0.0
                else:
                    yaw = np.degrees(np.arctan2(vy, vx))
                
            return np.array([x, y, z], dtype=float), yaw, np.array([vx, vy], dtype=float)
        except (ValueError, TypeError):
            return None

    def make_decision(self, own_file: Path, intr_file: Path) -> Dict[str, Any]:
        """
        核心流程：读取文件 -> 解析状态 -> RL查表 -> 返回结果
        """
        response = {
            "valid": False,
            "action_code": 0,
            "yaw_rate_cmd": 0.0,
            "maneuver": "Hold", # Display text
            "rl_state": None,
            "msg": ""
        }

        # 1. 检查文件
        if not own_file.exists():
            response["msg"] = f"Missing ownship file: {own_file}"
            return response
        if not intr_file.exists():
            response["msg"] = f"Missing intruder file: {intr_file}"
            # 没有入侵者时默认保持
            response["valid"] = True 
            return response

        # 2. 读取 JSON
        try:
            with open(own_file, 'r', encoding='utf-8') as f:
                own_json = json.load(f)
            with open(intr_file, 'r', encoding='utf-8') as f:
                intr_json = json.load(f)
        except Exception as e:
            response["msg"] = f"JSON Error: {e}"
            return response

        # 3. 解析物理量
        own_pos, own_yaw, own_vel = self._parse_ownship(own_json)
        intr_res = self._parse_intruder(intr_json)
        
        if intr_res is None:
            response["valid"] = True
            response["msg"] = "No intruder data found."
            return response
            
        intr_pos, intr_yaw, intr_vel = intr_res

        # 4. 计算 RL 状态 (与 QLearningAlgorithm.evaluate 逻辑一致)
        detected, theta_deg, dist3 = in_front_fov(own_yaw, own_pos, intr_pos)
        
        if not detected:
            response["valid"] = True
            response["msg"] = f"Intruder out of FOV (Dist={dist3:.1f}m)"
            return response

        # 计算分桶索引
        h_abs = abs(intr_pos[2] - own_pos[2])
        
        # --- 核心修改：矢量查表（相对速度角） ---
        rel_vel = intr_vel - own_vel
        # 如果相对速度极小，默认认为其航向角差为0（或按原逻辑备用）
        if np.linalg.norm(rel_vel) < 1e-3:
            rel_heading = 0.0
        else:
            rel_vel_ang = np.degrees(np.arctan2(rel_vel[1], rel_vel[0]))
            # 导入 numpy 或 math 实现归一化 
            rel_heading = (rel_vel_ang - own_yaw + 180.0) % 360.0 - 180.0
        # --------------------------------------
        
        th_idx = bin_theta(theta_deg)
        dist_idx = bin_dist(dist3)
        h_idx = bin_h(h_abs)
        head_idx = bin_heading(rel_heading) # 调用已有的 bin_heading 分桶

        # 5. 查表
        if th_idx is not None and dist_idx is not None:
            # 记录计算出的状态以便调试
            state_tuple = (th_idx, h_idx, dist_idx, head_idx)
            response["rl_state"] = state_tuple
            
            try:
                # 查表获取动作价值
                q_values = self.agent.Q[th_idx, h_idx, dist_idx, head_idx]
                
                # 贪婪策略选动作
                action_idx = int(np.argmax(q_values))
                
                # 6. 解析动作
                yaw_rate = ACTIONS[action_idx]
                
                response["valid"] = True
                response["action_code"] = action_idx
                response["yaw_rate_cmd"] = yaw_rate
                
                # --- 核心修改：适配极坐标系 (正角左转，负角右转) ---
                if yaw_rate < -20.0:
                    response["maneuver"] = "Hard Right Turn" # -30 角度变小，往右转
                elif yaw_rate < -1e-3:
                    response["maneuver"] = "Turn Right"      # -10
                elif yaw_rate > 20.0:
                    response["maneuver"] = "Hard Left Turn"  # +30 角度变大，往左转
                elif yaw_rate > 1e-3:
                    response["maneuver"] = "Turn Left"       # +10
                else:
                    response["maneuver"] = "Maintain"
                # --------------------------------------

                response["msg"] = "RL Decision made."
                
            except IndexError:
                response["msg"] = f"State index out of bounds: {state_tuple}"
        else:
            response["msg"] = "State mapping failed (likely out of range)"
            response["valid"] = True # 虽然失败但这仅仅是没触发RL，并不是系统错误

        return response