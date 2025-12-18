import pandas as pd
from pyproj import Transformer
import numpy as np
import json
import copy


def preprocess_data(filePath, utmZone="50N"):
    df = pd.read_json(filePath)
    zoneNumber = utmZone[:-1]
    hemisphere = "north" if utmZone[-1].upper() == "N" else "south"
    # EPSG:4326 是 (lon, lat)
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=utm +zone={zoneNumber} +{hemisphere} +ellps=WGS84",
        always_xy=True,   # 强制使用 (lon, lat) 顺序
    )
    df["location_latitude"] = df["location"].apply(lambda x: x["latitude"])
    df["location_longitude"] = df["location"].apply(lambda x: x["longitude"])

    # 注意顺序：先经度(longitude)，再纬度(latitude)
    df["easting"], df["northing"] = transformer.transform(
        df["location_longitude"].values,  # lon
        df["location_latitude"].values,   # lat
    )
    return df, transformer


def enu_to_latlon(easting: float, northing: float, transformer: Transformer) -> tuple[float, float]:
    """
    将 UTM (easting, northing) 转回 (latitude, longitude)
    """
    # inverse 时返回顺序为 (lon, lat)
    lon, lat = transformer.transform(easting, northing, direction="inverse")
    return lat, lon


def generate_intruder_json_from_datas(datas, transformer: Transformer, template_path: str, out_path: str,
                                      horizontalV: float, yaw: float):
    """
    使用 datas 中的 (east, north)，通过 transformer 转为 (lat, lon)，
    以 template_path 提供的 JSON 为模板，修改 location.latitude / location.longitude / location.timestamp，
    生成一个包含多条记录的 JSON 数组写入 out_path。
    """
    # 读取模板 JSON（单条）
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    results = []
    for idx, (easting, northing) in enumerate(datas):
        lat, lon = enu_to_latlon(easting, northing, transformer)

        # 深拷贝模板，避免修改原始模板对象
        item = copy.deepcopy(template)
        # 修改 location 中的字段
        item["location"]["latitude"] = float(lat)
        item["location"]["longitude"] = float(lon)
        item["location"]["timestamp"] = int(idx)  # 或者用别的时间规则
        item["location"]["horizontalSpeed"] = float(horizontalV)
        item["location"]["direction"] = float(np.degrees(yaw))  # 转为度
        # 顶层 time / connection.timestamp 如需要也可以一起改
        item["time"] = int(idx)
        item["connection"]["timestamp"] = int(idx)

        results.append(item)

    # 写出为 JSON 数组
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dataPath = "../data/IntruderReal.json"
    # dataPath = "../data/Cloud.json"
    # dataPath = "../data/Track.json"
    df, utm_transformer = preprocess_data(dataPath, utmZone="50N")

    # 示例：打印 Cloud.json 里的往返转换
    for idx, row in df.iterrows():
        print("ENU:", row["easting"], row["northing"])
        lat, lon = enu_to_latlon(row["easting"], row["northing"], utm_transformer)
        print("Back to WGS84:", row["easting"], row["northing"], "->", lon, lat)

    # print("=====" * 10)


    # dataNum = 20  # 要生成的数据数量
    # datas = [[] for _ in range(dataNum)]
    # eastBase = 503445  # 基准东坐标
    # northBase = 4371251.9  # 基准北坐标
    # horizontalV = 10  # 水平速度，米/秒
    # yaw = -60  # 偏航角，度为单位，0度为正东
    # yaw = np.radians(yaw)
    # for index, data in enumerate(datas):
    #     data.append(eastBase - horizontalV * np.cos(yaw) * (dataNum - index - 1))
    #     data.append(northBase - horizontalV * np.sin(yaw) * (dataNum - index - 1))

    # # 模板 JSON 路径（请把该模板内容保存为一个单独的文件，如 ../data/intruder_template.json）
    # template_json_path = "../data/intruder_template.json"
    # # 输出 JSON 路径
    # output_json_path = "../data/IntruderReal.json"

    # generate_intruder_json_from_datas(datas, utm_transformer, template_json_path, output_json_path, horizontalV, yaw)
    # print(f"生成完成: {output_json_path}")

