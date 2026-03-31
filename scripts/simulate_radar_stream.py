"""
scripts/simulate_radar_stream.py
模拟雷达数据流并与服务端交互闭环控制，
在二维平面展示轨迹，并在右侧极坐标图展示无人机前向FOV状态空间(Theta)分桶并高亮当前状态。
"""
import requests
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

URL = "http://127.0.0.1:5000/daa/run"

# Theta 分桶边界 (角度)，对应代码中的 8 个桶，范围从 -180 到 180
THETA_BINS = [-180, -150, -90, -30, 0, 30, 90, 150, 180]

# Q-learning 里 heading 分为 [-180, -150, -90, -30, 0, 30, 90, 150, 180] 共 8 个区间
HEADING_BINS = [-180, -150, -90, -30, 0, 30, 90, 150, 180]

def main():
    print(f"开始向 {URL} 发送连续数据流并执行动作闭环...")
    print("-" * 115)
    print(f"{'Own(X, Y)':<15} | {'Intr(X, Y)':<15} | {'Maneuver':>15} {'(Rate)'} | {'Msg':<8} | {'Q-State'}")
    print("-" * 115)
    
    # --- 场景定义：交叉汇聚 (Crossing Conflict) ---
    own_x, own_y = 0.0, 0.0
    own_vx, own_vy = 0.0, 10.0
    own_yaw = 90.0  
    own_speed = math.hypot(own_vx, own_vy)  

    intr_x, intr_y = -50.0, 100.0
    intr_vx, intr_vy = 10.0, -10.0
    intr_yaw = 315.0
    
    dt = 1.0 
    
    # --- 绘图初始化: 1行2列 ---
    plt.ion()  
    fig = plt.figure(figsize=(14, 6))
    
    # 左侧主图和右侧主图
    ax_map = fig.add_subplot(121)
    ax_state = fig.add_subplot(122, polar=True)
    
    # 创建罗盘镶嵌子图 (相对速度向量指示：处于右图偏右上区域)
    ax_compass = fig.add_axes([0.80, 0.70, 0.15, 0.15], polar=True)

    # --- 记录历史 ---
    own_history_x, own_history_y = [], []
    intr_history_x, intr_history_y = [], []

    # 进行仿真推演 15s
    for i in range(10):
        own_history_x.append(own_x)
        own_history_y.append(own_y)
        intr_history_x.append(intr_x)
        intr_history_y.append(intr_y)

        # 构造包
        flight_control = {"x": own_x, "y": own_y, "z": 25.0, "yaw": own_yaw, "vx": own_vx, "vy": own_vy}
        intruder_data = {"x": intr_x, "y": intr_y, "z": 25.0, "vx": intr_vx, "vy": intr_vy, "yaw": intr_yaw}
        payload = {"sensors": {"FlightControl": flight_control, "IntruderReal": intruder_data, "Track": None}}
        
        yaw_rate = 0.0  
        current_theta_idx = -1 
        current_dist_idx = -1  
        current_heading_idx = -1 

        try:
            resp = requests.post(URL, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                maneuver = data.get('guidance', {}).get('maneuver_text', 'N/A')
                yaw_rate = float(data.get('guidance', {}).get('yaw_rate', 0.0))
                msg = data.get('debug', {}).get('msg', '')
                
                # 获取状态以更新热力图
                rl_state = data.get('debug', {}).get('rl_state', None)
                if rl_state and isinstance(rl_state, list) and len(rl_state) >= 4: 
                    current_theta_idx = rl_state[0] 
                    current_dist_idx = rl_state[2]  
                    current_heading_idx = rl_state[3]
                
                own_str = f"({own_x:.0f}, {own_y:.0f})"
                intr_str = f"({intr_x:.0f}, {intr_y:.0f})"
                print(f"{own_str:<15} | {intr_str:<15} | {maneuver:>15} ({yaw_rate:>4.0f}) | {msg:<8} | {str(rl_state)}")
            else:
                print(f"Server Error: {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")
            
        # ==================================
        # 左侧子图：物理空间更新
        # ==================================
        ax_map.clear()
        ax_map.plot(own_history_x, own_history_y, 'b--', label='Ownship Path')
        ax_map.plot(intr_history_x, intr_history_y, 'r--', label='Intruder Path')
        
        own_dx = math.cos(math.radians(own_yaw)) * 15
        own_dy = math.sin(math.radians(own_yaw)) * 15
        ax_map.arrow(own_x, own_y, own_dx, own_dy, head_width=4, head_length=4, fc='blue', ec='blue')
        
        ax_map.plot(own_x, own_y, 'bo', markersize=8, label='Ownship')
        ax_map.plot(intr_x, intr_y, 'ro', markersize=8, label='Intruder')
        
        ax_map.set_title(f"DAA Avoidance Map (t={i}s)")
        ax_map.set_xlabel("X Position (m)")
        ax_map.set_ylabel("Y Position (m)")
        ax_map.set_xlim(-100, 100)
        ax_map.set_ylim(-20, 150)
        ax_map.grid(True)
        ax_map.legend(loc="upper right")
        
        # ==================================
        # 右侧子图：状态空间(Theta方位角 + 距离)
        # ==================================
        ax_state.clear()
        ax_state.set_theta_zero_location("N")
        
        # 必须设为 1 (逆时针)，这与数学坐标系(右负左正)对齐，保证现实你在右边图上也在右边
        ax_state.set_theta_direction(1)       
        
        ax_state.set_thetagrids([0, 30, 90, 150, 180, 210, 270, 330])
        ax_state.set_yticks([25, 50])
        ax_state.set_yticklabels(["25m", "50m"], color="#666666", fontsize=8)
        ax_state.set_ylim(0, 50) 
        ax_state.set_title("Forward FOV State Space", pad=20)
        ax_state.plot(0, 0, 'bo', markersize=10)
        
        DIST_BINS = [0, 25, 50]
        for t_idx in range(8):
            start_rad = math.radians(THETA_BINS[t_idx])
            end_rad = math.radians(THETA_BINS[t_idx+1])
            theta_vals = np.linspace(start_rad, end_rad, 50)
            
            for d_idx in range(len(DIST_BINS) - 1):
                r_inner = DIST_BINS[d_idx]
                r_outer = DIST_BINS[d_idx+1]
                
                is_active = (t_idx == current_theta_idx and d_idx == current_dist_idx)
                color = 'red' if is_active else '#efefef'
                alpha = 0.6 if is_active else 0.3
                
                r_inner_vals = np.full_like(theta_vals, r_inner)
                r_outer_vals = np.full_like(theta_vals, r_outer)
                
                ax_state.fill_between(theta_vals, r_inner_vals, r_outer_vals, color=color, alpha=alpha, edgecolor='gray')
                
                if is_active:
                    mid_rad = (start_rad + end_rad) / 2
                    mid_r = (r_inner + r_outer) / 2
                    ax_state.text(mid_rad, mid_r, f"S({t_idx},{d_idx})", ha='center', va='center', fontsize=8, color='white', fontweight='bold')

        # ==================================
        # 罗盘嵌图：相对速度向量指示
        # ==================================
        ax_compass.clear()
        ax_compass.set_theta_zero_location("N") 
        
        # 罗盘同样必须设为 1 (逆时针)，否则也会左右颠倒
        ax_compass.set_theta_direction(1)
        ax_compass.set_yticks([]) 
        ax_compass.set_xticks([])
        ax_compass.spines['polar'].set_visible(False) 
        ax_compass.set_title("Rel Velocity\n(Heading Idx)", fontsize=9, pad=10)
        
        ax_compass.set_ylim(0, 1.5)
        
        for h_idx in range(8):
            start_ang = HEADING_BINS[h_idx]
            end_ang = HEADING_BINS[h_idx+1]
            mid_ang = (start_ang + end_ang) / 2.0
            mid_rad = math.radians(mid_ang)
            
            is_active_heading = (h_idx == current_heading_idx)
            color = 'red' if is_active_heading else 'lightgray'
            linewidth = 3 if is_active_heading else 1
            length = 1.0 if is_active_heading else 0.7
            
            ax_compass.plot([mid_rad, mid_rad], [0, length], color=color, linewidth=linewidth)
            ax_compass.plot([mid_rad], [length], marker='^', color=color, markersize=linewidth*3)
            
            if is_active_heading:
                ax_compass.text(mid_rad, 1.3, f"H{h_idx}", color=color, fontweight='bold', ha='center', va='center', fontsize=9)
            else:
                ax_compass.text(mid_rad, 0.9, f"{h_idx}", color='gray', ha='center', va='center', fontsize=7)

        # UI 刷新
        plt.pause(dt)
        
        # --- 物理引擎推演下一帧 ---
        own_yaw = (own_yaw - yaw_rate * dt) % 360.0
        rad = math.radians(own_yaw)
        own_vx = own_speed * math.cos(rad)
        own_vy = own_speed * math.sin(rad)
        own_x += own_vx * dt
        own_y += own_vy * dt
        
        intr_x += intr_vx * dt
        intr_y += intr_vy * dt

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()