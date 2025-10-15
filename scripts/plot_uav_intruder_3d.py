import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    uav_line, = ax.plot([], [], [], 'bo-', label='UAV')
    fused_line, = ax.plot([], [], [], 'ro-', label='Intruder')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Up')
    ax.legend()
    # 使用最大长度,允许一方数据较少
    max_len = len(uav_east)
    if max_len == 0:
        print('[plot] 没有可用的 ENU 数据:UAV points=', len(uav_east), ' fused points=', len(fused_east))
        return
    # 若某一方长度不足,用 NaN 填充以保持帧数一致
    def pad(arr, n):
        if len(arr) >= n:
            return arr
        a = np.full(n, np.nan)
        a[:len(arr)] = arr
        return a
    uav_east = pad(uav_east, max_len)
    uav_north = pad(uav_north, max_len)
    uav_up = pad(uav_up, max_len)
    fused_east = pad(fused_east, max_len)
    fused_north = pad(fused_north, max_len)
    fused_up = pad(fused_up, max_len)
    # 在图上显示当前 timeStamp
    timestamp_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        # 固定East轴范围为503700到503800
        ax.set_xlim(503700, 503800)
        # 固定North轴范围为4370860到4370890
        ax.set_ylim(4370860, 4370890)
        # 固定纵轴(Up)范围为 5 到 15
        ax.set_zlim(5, 15)
        uav_line.set_data([], [])
        uav_line.set_3d_properties([])
        fused_line.set_data([], [])
        fused_line.set_3d_properties([])
        timestamp_text.set_text('')
        return uav_line, fused_line, timestamp_text
    def update(frame):
        # 对于 NaN 值,plot 会自动中断线段,保留已有轨迹
        uav_line.set_data(uav_east[:frame+1], uav_north[:frame+1])
        uav_line.set_3d_properties(uav_up[:frame+1])
        fused_line.set_data(fused_east[:frame+1], fused_north[:frame+1])
        fused_line.set_3d_properties(fused_up[:frame+1])
        # 显示当前时间戳标签
        try:
            ts = timestamps[frame]
            timestamp_text.set_text(f'timeStamp: {ts}')
        except Exception:
            timestamp_text.set_text('')
        return uav_line, fused_line, timestamp_text
    ani = FuncAnimation(fig, update, frames=max_len, init_func=init, blit=True, interval=500, repeat=False)
    plt.show()

if __name__ == "__main__":
    from DAAmain import DAAmain
    result = DAAmain()
    plot_uav_intruder_3d(result["timeline"])

