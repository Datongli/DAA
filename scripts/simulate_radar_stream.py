"""
scripts/simulate_radar_stream.py
模拟雷达源源不断地发送 Track.json 数据，验证主程序挂起时的连续处理能力，并执行动作闭环，
同时在二维平面实时可视化无人机和入侵飞机的轨迹，并输出当前控制策略对应的Q表状态。
"""
import requests
import time
import math
import matplotlib.pyplot as plt

URL = "http://127.0.0.1:5000/daa/run"

def main():
    print(f"开始向 {URL} 发送连续数据流并执行动作闭环...")
    print("-" * 110)
    # 修改1: 表头增加了 Q-State，并稍微排齐了一下宽度
    print(f"{'Own(X, Y)':<15} | {'Intr(X, Y)':<15} | {'Maneuver':>14} {'(Rate)'} | {'Msg':<10} | {'Q-State'}")
    print("-" * 110)
    
    # --- 场景定义：交叉汇聚 (Crossing Conflict) ---
    own_x, own_y = 0.0, 0.0
    own_vx, own_vy = 0.0, 10.0
    own_yaw = 90.0  
    own_speed = math.hypot(own_vx, own_vy)  

    intr_x, intr_y = -50.0, 100.0
    intr_vx, intr_vy = 10.0, -10.0
    intr_yaw = 315.0
    
    dt = 1.0 
    
    # --- 绘图初始化 ---
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 用于记录历史轨迹
    own_history_x, own_history_y = [], []
    intr_history_x, intr_history_y = [], []

    # 连续发送并推演 (模拟 10 秒过程，您可以自行扩展循环次数)
    for i in range(10):
        # 记录每帧位置
        own_history_x.append(own_x)
        own_history_y.append(own_y)
        intr_history_x.append(intr_x)
        intr_history_y.append(intr_y)

        # 1. 构造本机和入侵者数据
        flight_control = {
            "x": own_x, "y": own_y, "z": 25.0, "yaw": own_yaw,
            "vx": own_vx, "vy": own_vy
        }
        intruder_data = {
            "x": intr_x, "y": intr_y, "z": 25.0,
            "vx": intr_vx, "vy": intr_vy, "yaw": intr_yaw
        }
        
        payload = {
            "sensors": {
                "FlightControl": flight_control,
                "IntruderReal": intruder_data,
                "Track": None
            }
        }
        
        # 4. 发送并接收指令
        yaw_rate = 0.0  
        try:
            resp = requests.post(URL, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                maneuver = data.get('guidance', {}).get('maneuver_text', 'N/A')
                yaw_rate = float(data.get('guidance', {}).get('yaw_rate', 0.0))
                
                # 修改2: 在这里提取 rl_state 以及 msg
                msg = data.get('debug', {}).get('msg', '')
                rl_state = data.get('debug', {}).get('rl_state', 'None')
                
                own_str = f"({own_x:.0f}, {own_y:.0f})"
                intr_str = f"({intr_x:.0f}, {intr_y:.0f})"
                
                # 修改3: 将提取的 Q表状态追加到打印输出项中
                print(f"{own_str:<15} | {intr_str:<15} | {maneuver:>14} ({yaw_rate:>4.0f}) | {msg:<10} | {rl_state}")
            else:
                print(f"Server Error: {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")
            
        # 实时更新绘图
        ax.clear()
        
        # 绘制历史轨迹
        ax.plot(own_history_x, own_history_y, 'b--', label='Ownship Path')
        ax.plot(intr_history_x, intr_history_y, 'r--', label='Intruder Path')
        
        # 绘制当前点
        ax.plot(own_x, own_y, 'bo', markersize=8, label='Ownship')
        ax.plot(intr_x, intr_y, 'ro', markersize=8, label='Intruder')
        
        # 图表设置
        ax.set_title(f"DAA Avoidance Simulation - Time step {i}s")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        
        # 固定坐标系范围，防止画面忽大忽小
        ax.set_xlim(-100, 100)
        ax.set_ylim(-20, 150)
        ax.grid(True)
        ax.legend()
        
        # 暂停以刷新图表（代替 time.sleep）
        plt.pause(dt)
        
        # 5. 物理引擎推演下一帧
        own_yaw = (own_yaw + yaw_rate * dt) % 360.0
        rad = math.radians(own_yaw)
        own_vx = own_speed * math.cos(rad)
        own_vy = own_speed * math.sin(rad)
        
        own_x += own_vx * dt
        own_y += own_vy * dt
        
        intr_x += intr_vx * dt
        intr_y += intr_vy * dt

    # 循环结束后保持窗口不马上关闭
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()