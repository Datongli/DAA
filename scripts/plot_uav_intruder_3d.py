import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_uav_intruder_3d(timeline):
    """
    传入DAAmain的timeline,直接绘制UAV和fused的ENU三维轨迹动图
    时间戳≥7后,强制UAV向右上方飞行以避让入侵者
    """
    uav_east, uav_north, uav_up = [], [], []
    fused_east, fused_north, fused_up = [], [], []
    # 先按 timeStamp 对齐:提取所有时间戳并排序,然后为每个时间戳填充 UAV/fused 的 ENU(缺失用 NaN)
    # timeline 的每个 entry 可能对应同一个 timeStamp(多个 track),这里按第一个匹配的 entry 取 UAV/fused
    timestamps = sorted({int(e.get('timeStamp')) for e in timeline if e.get('timeStamp') is not None})
    for ts in timestamps:
        # 找到第一个 timeStamp == ts 的 entry
        entry = next((e for e in timeline if int(e.get('timeStamp', -999)) == ts), None)
        # UAV ownState.position(east/north/up)
        uav_pos = None
        if entry is not None:
            if "UAV" in entry and "position" in entry["UAV"]:
                uav_pos = entry["UAV"]["position"]
            elif "uav" in entry and "position" in entry["uav"]:
                uav_pos = entry["uav"]["position"]
        if uav_pos and all(k in uav_pos for k in ("east", "north", "up")):
            uav_east.append(uav_pos["east"])
            uav_north.append(uav_pos["north"])
            uav_up.append(uav_pos["up"])
        else:
            uav_east.append(np.nan)
            uav_north.append(np.nan)
            uav_up.append(np.nan)
        # 入侵者fused.position(east/north/up)
        fused_pos = None
        if entry is not None and "fused" in entry and "position" in entry["fused"]:
            fused_pos = entry["fused"]["position"]
        if fused_pos and all(k in fused_pos for k in ("east", "north", "up")):
            fused_east.append(fused_pos["east"])
            fused_north.append(fused_pos["north"])
            fused_up.append(fused_pos["up"])
        else:
            fused_east.append(np.nan)
            fused_north.append(np.nan)
            fused_up.append(np.nan)
    # 转为numpy数组
    uav_east = np.array(uav_east)
    uav_north = np.array(uav_north)
    uav_up = np.array(uav_up)
    fused_east = np.array(fused_east)
    fused_north = np.array(fused_north)
    fused_up = np.array(fused_up)
    
    # 时间戳≥7后强制UAV向右上方飞行(增加east和up)
    avoidance_start_idx = None
    for i, ts in enumerate(timestamps):
        if ts >= 7:
            avoidance_start_idx = i
            break
    
    if avoidance_start_idx is not None and not np.isnan(uav_east[avoidance_start_idx]):
        # 从avoidance_start_idx开始,每帧增加east和up
        for i in range(avoidance_start_idx, len(uav_east)):
            if not np.isnan(uav_east[i]):
                # 向右(east增加)和向上(up增加)
                offset = (i - avoidance_start_idx) * 5  # 每帧east增加5米
                uav_east[i] += offset
                uav_up[i] += offset * 0.05  # 每帧up增加1米的一半
    
    # === 轨迹插值部分 ===
    interp_steps = 10  # 每两个轨迹点之间插值帧数（越大越平滑）
    def interp_traj(arr):
        arr = np.array(arr)
        arr_interp = []
        for i in range(len(arr)-1):
            arr_interp.extend(np.linspace(arr[i], arr[i+1], interp_steps, endpoint=False))
        arr_interp.append(arr[-1])
        return np.array(arr_interp)
    # 对UAV和fused轨迹插值
    uav_east_interp = interp_traj(uav_east)
    uav_north_interp = interp_traj(uav_north)
    uav_up_interp = interp_traj(uav_up)
    fused_east_interp = interp_traj(fused_east)
    fused_north_interp = interp_traj(fused_north)
    fused_up_interp = interp_traj(fused_up)
    total_frames = len(uav_east_interp)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    uav_line, = ax.plot([], [], [], 'b-', lw=1, label='UAV')         # 线宽1，更细
    fused_line, = ax.plot([], [], [], 'r-', lw=1, label='Intruder')  # 线宽1，更细
    uav_point, = ax.plot([], [], [], 'bo', markersize=10)
    fused_point, = ax.plot([], [], [], 'ro', markersize=10)
    timestamp_marker, = ax.plot([], [], [], marker='^', color='k', markersize=5, linestyle='', label='TimeStamp Point')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('HEIGHT')
    ax.legend()
    ax.set_xlim(503700, 503800)
    ax.set_ylim(4370860, 4370890)
    ax.set_zlim(5, 15)
    ax.dist = 5
    timestamp_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
    action_text = ax.text2D(0.75, 0.90, '', transform=ax.transAxes, color='red', fontsize=14)

    def init():
        uav_line.set_data([], [])
        uav_line.set_3d_properties([])
        fused_line.set_data([], [])
        fused_line.set_3d_properties([])
        uav_point.set_data([], [])
        uav_point.set_3d_properties([])
        fused_point.set_data([], [])
        fused_point.set_3d_properties([])
        timestamp_marker.set_data([], [])
        timestamp_marker.set_3d_properties([])
        timestamp_text.set_text('')
        action_text.set_text('')
        return uav_line, fused_line, uav_point, fused_point, timestamp_text, timestamp_marker, action_text

    def update(frame):
        # 计算当前属于哪两个原始点之间
        idx = frame // interp_steps
        uav_line.set_data(uav_east_interp[:frame+1], uav_north_interp[:frame+1])
        uav_line.set_3d_properties(uav_up_interp[:frame+1])
        fused_line.set_data(fused_east_interp[:frame+1], fused_north_interp[:frame+1])
        fused_line.set_3d_properties(fused_up_interp[:frame+1])
        uav_point.set_data(uav_east_interp[frame:frame+1], uav_north_interp[frame:frame+1])
        uav_point.set_3d_properties(uav_up_interp[frame:frame+1])
        fused_point.set_data(fused_east_interp[frame:frame+1], fused_north_interp[frame:frame+1])
        fused_point.set_3d_properties(fused_up_interp[frame:frame+1])
        # 找出所有原始时间戳对应的插值帧索引
        ts_indices = [i * interp_steps for i in range(len(timestamps))]
        # 只在这些帧上画点
        marker_x = [uav_east_interp[i] for i in ts_indices if i <= frame]
        marker_y = [uav_north_interp[i] for i in ts_indices if i <= frame]
        marker_z = [uav_up_interp[i] for i in ts_indices if i <= frame]
        timestamp_marker.set_data(marker_x, marker_y)
        timestamp_marker.set_3d_properties(marker_z)
        # 显示当前时间戳标签
        try:
            ts = timestamps[idx]
            timestamp_text.set_text(f'timeStamp: {ts}')
            # 在时间戳为7的第一个插值帧动图上打印提示并停顿1秒
            if ts == 7 and frame == ts_indices[idx]:
                action_text.set_text("发现入侵者, 采取动作:\nTURN_RIGHT_FAST, CLIMB")
                plt.draw()      # 先刷新画布，确保文字显示
                plt.pause(2)    # 再停顿2秒
            else:
                action_text.set_text('')
        except Exception:
            timestamp_text.set_text('')
            action_text.set_text('')
        return uav_line, fused_line, uav_point, fused_point, timestamp_text, timestamp_marker, action_text

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False, interval=50, repeat=False)
    plt.show()

if __name__ == "__main__":
    from DAAmain import DAAmain
    result = DAAmain()
    plot_uav_intruder_3d(result["timeline"])

