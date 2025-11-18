import math
from math import cos, sin, radians
from dataclasses import dataclass
from typing import List, NamedTuple
from collections import namedtuple
import pandas as pd
from pyproj import Transformer
from typing import NamedTuple
import random
import matplotlib.pyplot as plt
import csv
from lxml import etree
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False    # 负号支持
# 定义 Point 类型
@dataclass
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))  # 基于 x 和 y 的元组计算哈希值
# 定义 Point 类型
# Point = namedtuple("Point", ["x", "y"])
class GeoPoint(NamedTuple):
    latitude: float
    longitude: float

class GADistance(NamedTuple):
    totalDist: float
    Start_index: int
    horizontal: bool

class GAResult(NamedTuple):
    bestAngle: float
    bestDistance: float
    horizontal: bool
    start_index: int

# 经纬度 → UTM（Zone 50N）转换
def latlon_to_xy(latlon: GeoPoint) -> Point:
    # 创建转换器：WGS84 → UTM Zone 50N
    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84
        "EPSG:32650", # UTM Zone 50N (northern hemisphere)
        always_xy=True  # 确保 (lon, lat) 顺序
    )
    x, y = transformer.transform(latlon.longitude, latlon.latitude)
    return Point(x, y)
def latlon_to_xy1(latlon):
    transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
    # latlon 是一个字典，如 {'latitude': xxx, 'longitude': xxx}
    x, y = transformer.transform(latlon['longitude'], latlon['latitude'])
    return x, y
# UTM → 经纬度转换
def xy_to_latlon(point: Point) -> GeoPoint:
    # 创建反向转换器：UTM Zone 50N → WGS84
    transformer = Transformer.from_crs(
        "EPSG:32650",  # UTM Zone 50N
        "EPSG:4326",   # WGS84
        always_xy=True
    )
    lon, lat = transformer.transform(point.x, point.y)
    return GeoPoint(lat, lon)
# 凸包计算：Andrew's Monotone Chain
def convex_hull(points: List[Point]) -> List[Point]:
    points = sorted(set(points), key=lambda p: (p.x, p.y))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last point of each half (repeats start point)
    return lower[:-1] + upper[:-1]
def rotate_point(p: Point, angle_deg: float, center: Point) -> Point:
    """绕 center 点旋转 angle_deg 角度"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx, dy = p.x - center.x, p.y - center.y
    x_new = dx * cos_a - dy * sin_a + center.x
    y_new = dx * sin_a + dy * cos_a + center.y
    return Point(x_new, y_new)

def rotate_points(points: List[Point], angle_deg: float, center: Point) -> List[Point]:
    return [rotate_point(p, angle_deg, center) for p in points]


# 生成航线航点
from typing import List


def generate_waypoints_with_direction(
        convex_hull_points: List[Point],
        center: Point,
        along_step: float,
        across_step: float,
        best_angle: float,
        horizontal: bool,
        start_index: int,  # 0: 左下, 1: 左上, 2: 右上, 3: 右下
        csv_path1: str  # 保存CSV文件的路径
) -> List[Point]:
    # 方向步长
    step_x = along_step if horizontal else across_step
    step_y = across_step if horizontal else along_step

    # Step 1: 旋转对齐
    aligned_hull = rotate_points(convex_hull_points, best_angle, center)

    # Step 2: 包围盒
    x_min = min(p.x for p in aligned_hull)
    x_max = max(p.x for p in aligned_hull)
    y_min = min(p.y for p in aligned_hull)
    y_max = max(p.y for p in aligned_hull)
     # 0: 左下, 1: 左上, 2: 右上, 3: 右下
    # 输出旋转后矩形范围的四个角的经纬度坐标
    rotated_coords = [
        ('0', xy_to_latlon(Point(x_min, y_min))),
        ('1', xy_to_latlon(Point(x_min, y_max))),
        ('2', xy_to_latlon(Point(x_max, y_max))),
        ('3', xy_to_latlon(Point(x_max, y_min)))
    ]

    # 输出旋转前矩形范围的四个角的经纬度坐标
    x_min_original = min(p.x for p in convex_hull_points)
    x_max_original = max(p.x for p in convex_hull_points)
    y_min_original = min(p.y for p in convex_hull_points)
    y_max_original = max(p.y for p in convex_hull_points)
    original_coords = [
        ('0', xy_to_latlon(Point(x_min_original, y_min_original))),
        ('1', xy_to_latlon(Point(x_min_original, y_max_original))),
        ('2', xy_to_latlon(Point(x_max_original, y_max_original))),
        ('3', xy_to_latlon(Point(x_max_original, y_min_original)))
    ]
    # 下面这部分是还原旋转后的矩形角点到原始坐标系下
    # 生成旋转后的矩形角点进行还原
    rotated_corners_xy = [
        Point(x_min, y_min),
        Point(x_min, y_max),
        Point(x_max, y_max),
        Point(x_max, y_min)
    ]
    # 将旋转后的矩形角点还原回原始坐标系
    original_corners1 = rotate_points(rotated_corners_xy, -best_angle, center)
    # 输出旋转前矩形范围的四个角的经纬度坐标
    rotated_original_coords = [(f'{i}', xy_to_latlon(p)) for i, p in enumerate(original_corners1)]
    # 将旋转前后矩形坐标保存到CSV文件
    with open(csv_path1, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "类型", "纬度", "经度"])

        # 写入旋转后矩形坐标
        for i, (label, latlon) in enumerate(rotated_coords):
            writer.writerow([i + 1, f"旋转后{label}", latlon.latitude, latlon.longitude])

        # 写入旋转前矩形坐标
        for i, (label, latlon) in enumerate(original_coords):
            writer.writerow([i + 5, f"旋转前{label}", latlon.latitude, latlon.longitude])
        # 写入旋转前矩形坐标
        for i, (label, latlon) in enumerate(rotated_original_coords):
            writer.writerow([i + 5, f"旋转后还原{label}", latlon.latitude, latlon.longitude])
    # Step 3: 生成拐点航线
    corner_points = []

    if horizontal:
        # 横向扫描：Y轴推进，X方向蛇形
        row_num = 0
        y_start = y_min if start_index in (0, 3) else y_max
        y_step = step_y if start_index in (0, 3) else -step_y

        y = y_start
        while (y_step > 0 and y <= y_max + 1e-6) or (y_step < 0 and y >= y_min - 1e-6):
            left_to_right = (start_index in (0, 1)) ^ (row_num % 2 != 0)
            x_range = [x_min + i * step_x for i in range(int((x_max - x_min) / step_x) + 1)] if left_to_right else [
                x_max - i * step_x for i in range(int((x_max - x_min) / step_x) + 1)]

            # 添加每行的第一个和最后一个点（这些是拐点）
            if left_to_right:
                corner_points.append(Point(x_min, y))  # 行的最左端
                corner_points.append(Point(x_max, y))  # 行的最右端
            else:
                corner_points.append(Point(x_max, y))  # 行的最右端
                corner_points.append(Point(x_min, y))  # 行的最左端

            y += y_step
            row_num += 1

    else:
        # 纵向扫描：X轴推进，Y方向蛇形
        col_num = 0
        x_start = x_min if start_index in (0, 1) else x_max
        x_step = step_x if start_index in (0, 1) else -step_x

        x = x_start
        while (x_step > 0 and x <= x_max + 1e-6) or (x_step < 0 and x >= x_min - 1e-6):
            bottom_to_top = (start_index in (0, 3)) ^ (col_num % 2 != 0)
            y_range = [y_min + i * step_y for i in range(int((y_max - y_min) / step_y) + 1)] if bottom_to_top else [
                y_max - i * step_y for i in range(int((y_max - y_min) / step_y) + 1)]

            # 添加每列的第一个和最后一个点（这些是拐点）
            if bottom_to_top:
                corner_points.append(Point(x, y_min))  # 列的最底端
                corner_points.append(Point(x, y_max))  # 列的最顶端
            else:
                corner_points.append(Point(x, y_max))  # 列的最顶端
                corner_points.append(Point(x, y_min))  # 列的最底端

            x += x_step
            col_num += 1

    # Step 4: 旋转还原
    rotated_corner_points = rotate_points(corner_points, -best_angle, center)
    return rotated_corner_points


# GE遗传算法求范围

# 可视化航点和航线
def visualize_waypoints(waypoints: List[Point]):
    x_vals = [p.x for p in waypoints]
    y_vals = [p.y for p in waypoints]

    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', markersize=5, label="航点")
    plt.scatter(x_vals, y_vals, color='red', label='航点')  # 标注航点

    # 标题与标签
    plt.title('航点及航线可视化')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()

def calculateFlightDistance(angleDeg: float,
                             points: List[Point],
                             center: Point,
                             alongStep: float,
                             acrossStep: float,
                             horizontal: bool,
                             Takeoff_point: Point,
                             now_point: Point) -> GADistance:

    # 1. Rotate points to specified angle
    rotated = rotate_points(points, angleDeg, center)

    # 2. Compute bounding box
    minX = min(p.x for p in rotated)
    maxX = max(p.x for p in rotated)
    minY = min(p.y for p in rotated)
    maxY = max(p.y for p in rotated)

    # 3. Rectangle corners after rotation
    cornersRotated = [
        Point(minX, minY),
        Point(minX, maxY),
        Point(maxX, maxY),
        Point(maxX, minY)
    ]

    # 4. Rotate takeoff and nowpoint point
    takeoffRotated = rotate_point(Takeoff_point, angleDeg, center)
    nowpointRotated = rotate_point(now_point, angleDeg, center)
    # 5. Find closest corner
    minDist = float('inf')
    startIdx = 0
    for i, corner in enumerate(cornersRotated):
        d = math.hypot(corner.x - nowpointRotated.x, corner.y - nowpointRotated.y)
        if d < minDist:
            minDist = d
            startIdx = i
    start = cornersRotated[startIdx]

    # 6. Step sizes
    stepSizeX = alongStep if horizontal else acrossStep
    stepSizeY = acrossStep if horizontal else alongStep

    lines = math.ceil((maxY - minY) / stepSizeY) + 1 if horizontal else \
            math.ceil((maxX - minX) / stepSizeX) + 1

    # 7. Determine Z-path end point
    even = (lines % 2 == 0)
    if horizontal:
        if startIdx == 0:
            end = Point(minX, minY + (lines - 1) * stepSizeY) if even else Point(maxX, minY + (lines - 1) * stepSizeY)
        elif startIdx == 1:
            end = Point(minX, maxY - (lines - 1) * stepSizeY) if even else Point(maxX, maxY - (lines - 1) * stepSizeY)
        elif startIdx == 2:
            end = Point(maxX, maxY - (lines - 1) * stepSizeY) if even else Point(minX, maxY - (lines - 1) * stepSizeY)
        else:
            end = Point(maxX, minY + (lines - 1) * stepSizeY) if even else Point(minX, minY + (lines - 1) * stepSizeY)
    else:
        if startIdx == 0:
            end = Point(minX + (lines - 1) * stepSizeX, minY) if even else Point(minX + (lines - 1) * stepSizeX, maxY)
        elif startIdx == 1:
            end = Point(minX + (lines - 1) * stepSizeX, maxY) if even else Point(minX + (lines - 1) * stepSizeX, minY)
        elif startIdx == 2:
            end = Point(maxX - (lines - 1) * stepSizeX, maxY) if even else Point(maxX - (lines - 1) * stepSizeX, minY)
        else:
            end = Point(maxX - (lines - 1) * stepSizeX, minY) if even else Point(maxX - (lines - 1) * stepSizeX, maxY)

    # 8. Distance calculation
    d_start = math.hypot(start.x - nowpointRotated.x, start.y - nowpointRotated.y)
    d_end = math.hypot(end.x - takeoffRotated.x, end.y - takeoffRotated.y)

    scanLength = (maxX - minX) if horizontal else (maxY - minY)
    if horizontal:
        coverageDist = lines * scanLength + (lines - 1) * stepSizeY
    else:
        coverageDist = lines * scanLength + (lines - 1) * stepSizeX

    # 9. Turn penalty
    numTurns = lines - 1
    turnPenalty = 0.7
    smoothPenalty = numTurns * turnPenalty

    totalDist = coverageDist + d_start + d_end + smoothPenalty

    return GADistance(totalDist=totalDist, Start_index=startIdx, horizontal=horizontal)

def evaluate_angle(angle_deg: float, points: List[Point],
                   center: Point, along_step: float,
                   across_step: float, takeoff_point: Point,now_point: Point) -> GADistance:
    dist_h = calculateFlightDistance(angle_deg, points, center, along_step, across_step, True, takeoff_point,now_point)
    dist_v = calculateFlightDistance(angle_deg, points, center, along_step, across_step, False, takeoff_point,now_point)
    return dist_h if dist_h.totalDist < dist_v.totalDist else dist_v

def genetic_algorithm(points: List[Point], center: Point,
                      along_step: float, across_step: float,
                      takeoff_point: Point, now_point: Point,
                      pop_size: int = 50,
                      generations: int = 100, mutation_rate: float = 0.1,
                      crossover_rate: float = 0.8) -> GAResult:

    population = [random.uniform(0.0, 180.0) for _ in range(pop_size)]
    best_result = GAResult(0.0, float('inf'), True, 0)

    for gen_idx in range(generations):
        fitness = []
        directions = []
        starts = []

        for angle in population:
            dist = evaluate_angle(angle, points, center, along_step, across_step, takeoff_point,now_point)
            fitness.append(dist.totalDist)
            directions.append(dist.horizontal)
            starts.append(dist.Start_index)

            if dist.totalDist < best_result.bestDistance:
                best_result = GAResult(angle, dist.totalDist, dist.horizontal, dist.Start_index)

        # 锦标赛选择
        selected = []
        for _ in range(pop_size):
            a, b = random.randint(0, pop_size - 1), random.randint(0, pop_size - 1)
            selected.append(population[a] if fitness[a] < fitness[b] else population[b])

        # 交叉 + 变异
        new_population = []
        for i in range(0, pop_size, 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % pop_size]

            c1, c2 = p1, p2
            if random.random() < crossover_rate:
                alpha = random.uniform(-0.3, 1.3)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1

            if random.random() < mutation_rate:
                c1 = (c1 + random.gauss(0, 10)) % 180.0
            if random.random() < mutation_rate:
                c2 = (c2 + random.gauss(0, 10)) % 180.0

            new_population.extend([c1, c2])

        population = new_population

    print(f"[Best] Angle = {best_result.bestAngle:.2f}, "
          f"Horizontal = {best_result.horizontal}, "
          f"Start Corner = {best_result.start_index}, "
          f"Distance = {best_result.bestDistance:.2f}")
    return best_result


# def read_csv_to_points(csv_path: str) -> List[Point]:
#     points = []
#     with open(csv_path, newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             lon = float(row['经度'])
#             lat = float(row['纬度'])
#
#             points.append(Point(lon, lat))  # x=经度，y=纬度
#     return points


# 生成KML文件
def generate_kml(waypoints_latlon: List[GeoPoint], kml_path: str):
    kml_ns = {
        None: 'http://www.opengis.net/kml/2.2',
        'gx': 'http://www.google.com/kml/ext/2.2'
    }

    # 创建根元素 <kml>
    kml = etree.Element('kml', nsmap=kml_ns)
    doc = etree.SubElement(kml, 'Document')

    # 创建 <Placemark> 元素，包含 <LineString>
    placemark = etree.SubElement(doc, 'Placemark')
    name = etree.SubElement(placemark, 'name')
    name.text = 'Flight Path'
    line = etree.SubElement(placemark, 'LineString')
    coordinates = etree.SubElement(line, 'coordinates')

    # 将所有航点坐标按顺序添加到 <coordinates> 中
    coords_text = ' '.join(f'{p.longitude},{p.latitude},30' for p in waypoints_latlon)
    coordinates.text = coords_text

    # 保存文档到 kml_path
    tree = etree.ElementTree(kml)
    tree.write(kml_path, pretty_print=True, encoding='utf-8', xml_declaration=True)
    print(f"KML 文件已生成并保存到: {kml_path}")


import csv
from typing import List


def read_csv_to_points(csv_path: str) -> List:
    points = []
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        # 1️⃣ 标准化列名（去空格、小写）
        fieldnames = [f.strip().lower() for f in reader.fieldnames]
        print("📋 检测到的列名:", fieldnames)

        # 2️⃣ 自动匹配经纬度列
        lat_key = next((f for f in fieldnames if 'lat' in f), None)
        lon_key = next((f for f in fieldnames if 'lon' in f), None)

        if not lat_key or not lon_key:
            raise ValueError(f"❌ CSV 文件中未找到经纬度列！检测到的列名: {fieldnames}")

        # 3️⃣ 逐行读取并转为点
        for row in reader:
            # 注意：DictReader 返回原始列名，需要用原始 key 查找
            # 因此我们构造一个“标准化后的行字典”
            row_std = {k.strip().lower(): v for k, v in row.items()}

            try:
                lat = float(row_std[lat_key])
                lon = float(row_std[lon_key])
                point = latlon_to_xy(GeoPoint(latitude=lat, longitude=lon))
                points.append(point)
            except (ValueError, KeyError, TypeError):
                continue  # 跳过坏行

    print(f"✅ 成功读取 {len(points)} 个点")
    return points


def generate_waypoints_from_csv(csv_path: str, overlap_h: float, overlap_w: float, Z_c: float):
    # 固定起飞点和当前位置
    takeoff_lat = 40.89050666
    takeoff_lon = 121.79796901
    takeoff_point = latlon_to_xy(GeoPoint(latitude=takeoff_lat, longitude=takeoff_lon))

    now_lat = 40.88859076
    now_lon = 121.80002605
    now_point = latlon_to_xy(GeoPoint(latitude=now_lat, longitude=now_lon))

    # 相机参数
    sensor_width = 34
    sensor_height = 19
    focal_length = 35.0

    # 读取点集并求凸包
    plane_points = read_csv_to_points(csv_path)
    convex_hull_points = convex_hull(plane_points)

    print("凸包点集的经纬度:")
    for p in convex_hull_points:
        latlon = xy_to_latlon(p)
        print(f"纬度: {latlon.latitude}, 经度: {latlon.longitude}")

    # 计算中心点
    center = Point(sum(p.x for p in plane_points) / len(plane_points),
                   sum(p.y for p in plane_points) / len(plane_points))
    center_latlon = xy_to_latlon(center)
    print(f"中心点: 纬度={center_latlon.latitude:.8f}, 经度={center_latlon.longitude:.8f}")

    # 保存中心点
    center_csv_path = r"D:\Desktop\论文\实验\航线规划\中心点.csv"
    with open(center_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerow([center_latlon.latitude, center_latlon.longitude])

    # 计算航带间距
    image_width = sensor_width * (Z_c / focal_length)
    image_height = sensor_height * (Z_c / focal_length)
    across_step = image_width * (1 - overlap_w)
    along_step = image_height * (1 - overlap_h)

    # 遗传算法优化
    result = genetic_algorithm(convex_hull_points, center, along_step, across_step,
                               takeoff_point, now_point, pop_size=100, generations=50,
                               mutation_rate=0.1, crossover_rate=0.8)

    # 生成航点
    waypoints = generate_waypoints_with_direction(convex_hull_points, center,
                                                  along_step, across_step,
                                                  result.bestAngle,
                                                  result.horizontal,
                                                  result.start_index,
                                                  r"G:\Darklabel\seal_project\location\15m\矩形范围（检测跟踪聚类）_60.csv")

    # 转换为经纬度
    waypoints_latlon = [xy_to_latlon(p) for p in waypoints]

    # ✅ 计算航线总长度
    total_distance = 0.0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1].x - waypoints[i].x
        dy = waypoints[i+1].y - waypoints[i].y
        total_distance += math.hypot(dx, dy)

    # 输出航点数量与总距离
    print("\n================ 航线统计信息 ================")
    print(f"航点数量: {len(waypoints)} 个")
    print(f"航线总距离: {total_distance:.2f} 米")
    print(f"平均航段距离: {total_distance / (len(waypoints)-1):.2f} 米")
    print("==============================================\n")

    # 可视化与输出文件
    visualize_waypoints(waypoints)

    # 保存航点经纬度 CSV
    waypoints_latlon_csv_path = r"G:\Darklabel\seal_project\location\15m\航点(检测跟踪聚类)_60.csv"
    with open(waypoints_latlon_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Target_Latitude", "Target_Latitude"])
        for i, p in enumerate(waypoints_latlon):
            writer.writerow([i + 1, p.latitude, p.longitude])

    # 输出经纬度航点
    print("生成的拐点坐标（经纬度）：")
    for i, p in enumerate(waypoints_latlon):
        print(f"Point {i}: lat = {p.latitude:.8f}, lon = {p.longitude:.8f}")

    # 生成 KML 文件
    generate_kml(waypoints_latlon, r'G:\Darklabel\seal_project\location\15m\飞行航线（检测跟踪聚类）_60.kml')

    # 返回结果
    return waypoints, waypoints_latlon, total_distance

# ===== 主程序入口 =====
if __name__ == '__main__':
    # generate_waypoints_from_csv( r'G:\Darklabel\seal_project\location\15m\video_tracking（track）.csv',  0.8,  0.8, 60)
    generate_waypoints_from_csv( r'G:\Darklabel\seal_project\location\15m\position_clusing.csv',  0.8,  0.8, 60)