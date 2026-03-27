"""
scripts/simulate_radar_stream.py
模拟雷达源源不断地发送 Track.json 数据，验证主程序挂起时的连续处理能力。
"""
import requests
import time
import json
import math

URL = "http://127.0.0.1:5000/daa/run"

def main():
    print(f"开始向 {URL} 发送连续数据流...")
    print("-" * 60)
    print(f"{'Own(X, Y)':<15} | {'Intr(X, Y)':<15} | {'Maneuver':>15} {'(Rate)'} | {'Msg'}")
    print("-" * 60)
    
    # --- 场景定义：交叉汇聚 (Crossing Conflict) ---
    # 本机：从 (0, 0) 开始，向北飞 (90度)，速度 10 m/s
    own_x, own_y = 0.0, 0.0
    own_vx, own_vy = 0, 10
    own_yaw = 135.0

    # 敌人：从右侧 (50, 50) 开始，向西飞 (180度)，横切本机航线，速度 10 m/s
    # 预计 5秒后，本机到达 (0, 50)，敌人到达 (0, 50) -> 发生碰撞！
    intr_x, intr_y = -50.0, 100.0
    intr_vx, intr_vy = 10, -10
    intr_yaw = 315
    
    # 仿真步长
    dt = 1.0 
    
    # 连续发送 10 帧 (模拟 10秒过程)
    for i in range(10):
        # 1. 构造本机数据
        flight_control = {
            "x": own_x, "y": own_y, "z": 25.0, "yaw": own_yaw,
            "vx": own_vx, "vy": own_vy
        }
        
        # 2. 构造入侵者数据
        intruder_data = {
            "x": intr_x, "y": intr_y, "z": 25.0,
            "vx": intr_vx, "vy": intr_vy, "yaw": intr_yaw
        }
        
        # 3. 构造请求包
        payload = {
            "sensors": {
                "FlightControl": flight_control,
                "IntruderReal": intruder_data,
                "Track": None
            }
        }
        
        # 4. 发送并接收指令
        try:
            resp = requests.post(URL, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                maneuver = data.get('guidance', {}).get('maneuver_text', 'N/A')
                yaw_rate = data.get('guidance', {}).get('yaw_rate', 0.0)
                msg = data.get('debug', {}).get('msg', '')
                
                # 打印友好的单行日志
                own_str = f"({own_x:.0f}, {own_y:.0f})"
                intr_str = f"({intr_x:.0f}, {intr_y:.0f})"
                print(f"{own_str:<15} | {intr_str:<15} | {maneuver:>15} ({yaw_rate:>4.0f}) | {msg}")
            else:
                print(f"Server Error: {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")
            
        # 5. 物理引擎推演下一帧 (更新双方位置)
        own_x += own_vx * dt
        own_y += own_vy * dt
        
        intr_x += intr_vx * dt
        intr_y += intr_vy * dt
        
        time.sleep(1.0) # 模拟 1Hz 刷新

if __name__ == "__main__":
    main()