import csv
from pathlib import Path

# Define mappings
theta_map = {
    0: "左后方 (-180~-150)",
    1: "左侧 (-150~-90)",
    2: "左前方 (-90~-30)",
    3: "正左前方 (-30~0)",
    4: "正右前方 (0~30)",
    5: "右前方 (30~90)",
    6: "右侧 (90~150)",
    7: "右后方 (150~180)"
}

h_map = {
    0: "同高度 (<10m)",
    1: "略有高差 (10-20m)",
    2: "中等高差 (20-30m)",
    3: "较大高差 (30-40m)",
    4: "很大高差 (>40m)"
}

dist_map = {
    0: "近距离 (<25m)",
    1: "远距离 (25-50m)"
}

input_path = Path("checkpoints/qtable_uav.csv")
output_path = Path("checkpoints/qtable_readable.md")

if not input_path.exists():
    print(f"Error: {input_path} not found.")
    exit(1)

markdown_table = []
markdown_table.append("| 方位 (Theta) | 高度差 (H) | 距离 (Dist) | 左急转 (-30°) | 左缓转 (-10°) | 直飞 (0°) | 右缓转 (+10°) | 右急转 (+30°) |")
markdown_table.append("|---|---|---|---|---|---|---|---|")

with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader) # Skip header
    
    for row in reader:
        theta_idx = int(row[0])
        h_idx = int(row[1])
        dist_idx = int(row[2])
        
        # Check if row is all zeros (optional, but user asked for the table based on the file, usually implies all data)
        # But to make it readable in chat, maybe I should filter? 
        # The user said "generate a table", usually implies the full structure. 
        # However, 80 rows is a lot. I will generate the full file and print non-zero rows for the user.
        
        q_values = [float(x) for x in row[3:]]
        
        # Format Q-values to 2 decimal places
        q_strs = [f"{q:.2f}" for q in q_values]
        
        line = f"| {theta_map[theta_idx]} | {h_map[h_idx]} | {dist_map[dist_idx]} | {q_strs[0]} | {q_strs[1]} | {q_strs[2]} | {q_strs[3]} | {q_strs[4]} |"
        markdown_table.append(line)

# Write to file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(markdown_table))

print(f"Readable table generated at: {output_path}")

# Print non-zero rows for display
print("\n--- Preview of Active States (Non-Zero Q-Values) ---")
print("| 方位 | 高度差 | 距离 | 左急转 | 左缓转 | 直飞 | 右缓转 | 右急转 |")
print("|---|---|---|---|---|---|---|---|")
with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        q_values = [float(x) for x in row[3:]]
        if any(abs(q) > 0.001 for q in q_values):
             theta_idx = int(row[0])
             h_idx = int(row[1])
             dist_idx = int(row[2])
             q_strs = [f"{q:.2f}" for q in q_values]
             print(f"| {theta_map[theta_idx].split(' ')[0]} | {h_map[h_idx].split(' ')[0]} | {dist_map[dist_idx].split(' ')[0]} | {q_strs[0]} | {q_strs[1]} | {q_strs[2]} | {q_strs[3]} | {q_strs[4]} |")
