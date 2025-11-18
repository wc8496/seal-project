import pandas as pd
import numpy as np
from pyproj import Proj

# ========================
# 相机参数与无人机姿态类
# ========================
class CameraParams:
    def __init__(self, img_width, img_height, sensor_width, sensor_height, focal_length):
        self.img_width = img_width
        self.img_height = img_height
        self.sensor_width = sensor_width      # mm
        self.sensor_height = sensor_height    # mm
        self.focal_length = focal_length      # mm

class UAVPose:
    def __init__(self, lat, lon, alt, yaw, pitch, roll):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

# ========================
# 角度转弧度
# ========================
def deg2rad(deg):
    return deg * np.pi / 180.0

# ========================
# 旋转矩阵（ZYX）
# ========================
def build_rotation_matrix(pose):
    cy = np.cos(deg2rad(pose.yaw))
    sy = np.sin(deg2rad(pose.yaw))
    cp = np.cos(deg2rad(pose.pitch))
    sp = np.sin(deg2rad(pose.pitch))
    cr = np.cos(deg2rad(pose.roll))
    sr = np.sin(deg2rad(pose.roll))

    R = np.zeros((3, 3))
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    return R

# ========================
# 像素 → 经纬度
# ========================
def pixel_to_latlon(cam, uav, pixel_coords, relative_height):
    x_pix, y_pix = pixel_coords

    # 图像物理坐标（mm）
    x_img = (x_pix - cam.img_width / 2.0) * cam.sensor_width / cam.img_width
    y_img = (y_pix - cam.img_height / 2.0) * cam.sensor_height / cam.img_height

    # 相机坐标（m）
    X_c = x_img * relative_height / cam.focal_length
    Y_c = y_img * relative_height / cam.focal_length
    Z_c = relative_height

    # 相机 → 机体
    X_b = -Y_c
    Y_b = X_c
    Z_b = Z_c

    # 机体 → NED
    R = build_rotation_matrix(uav)
    N = R[0, 0] * X_b + R[0, 1] * Y_b + R[0, 2] * Z_b
    E = R[1, 0] * X_b + R[1, 1] * Y_b + R[1, 2] * Z_b

    # UTM 投影
    utm_proj = Proj(proj='utm', zone=51, ellps='WGS84')
    e_uav, n_uav = utm_proj(uav.lon, uav.lat)
    e_target = e_uav + E
    n_target = n_uav + N
    lon_t, lat_t = utm_proj(e_target, n_target, inverse=True)

    return lat_t, lon_t

# ========================
# 主函数：只定位 新目标
# ========================
# === 主函数：对 所有检测框 进行定位 ===
def localize_all_targets(track_txt, log_csv, output_csv):
    # === 相机参数 ===
    cam = CameraParams(
        img_width=3840,      # M3T 视频分辨率
        img_height=2160,
        sensor_width=6.4,    # 1/2-inch CMOS
        sensor_height=4.8,
        focal_length=6.72    # 实际焦距
    )
    # === 缩放比例（标注 1440×1080 → 原图 3840×2160） ===
    SCALE_X = 3840 / 1440  # 2.66667
    SCALE_Y = 2160 / 1080  # 2.0

    # === 读取飞行日志（修复版）===
    df_log = pd.read_csv(log_csv)
    df_log['frame_num'] = df_log['real_frame'].astype(int)
    df_log = df_log.drop_duplicates(subset='frame_num', keep='first')
    print(f"飞行日志帧号范围: {df_log['frame_num'].min()} ~ {df_log['frame_num'].max()}")

    # === 读取跟踪结果 ===
    detections = []
    with open(track_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('frame'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 10:
                continue
            try:
                frame = int(parts[0])
                obj_id = int(parts[1]) if parts[1] != '-1' else -1
                x1 = float(parts[2])
                y1 = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                # 缩放像素坐标
                center_x_raw = x1 + w / 2
                center_y_raw = y1 + h / 2
                center_x = center_x_raw * SCALE_X
                center_y = center_y_raw * SCALE_Y
                center_x = np.clip(center_x, 0, 8192)
                center_y = np.clip(center_y, 0, 5460)

                detections.append({
                    'frame': frame,
                    'id': obj_id,
                    'pixel_x': center_x,
                    'pixel_y': center_y
                })
            except Exception as e:
                print(f"跳过无效行: {line} -> {e}")

    # === 结果列表 ===
    results = []

    for det in detections:
        frame = det['frame']
        obj_id = det['id']

        # === 查找飞行数据 ===
        match = df_log[df_log['frame_num'] == frame]
        if match.empty:
            print(f"警告：帧 {frame} 无飞行数据")
            continue

        row = match.iloc[0]
        relative_height = 70.0  # 相机到目标距离
        uav_alt_true = 80.0     # 真实海拔（假设起飞点海拔 10m）
        # relative_height = row['altitude']                    # 相机到目标距离
        # uav_alt_true = row['altitude'] + 10.0                # 真实海拔

        uav = UAVPose(
            lat=row['latitude'],
            lon=row['longitude'],
            alt=uav_alt_true,
            yaw=row['yaw'],
            pitch=row['pitch'],
            roll=row['roll']
        )

        try:
            lat_t, lon_t = pixel_to_latlon(
                cam=cam,
                uav=uav,
                pixel_coords=(det['pixel_x'], det['pixel_y']),
                relative_height=relative_height
            )

            results.append({
                'frame': frame,
                'track_id': obj_id,
                'pixel_x': round(det['pixel_x'], 2),
                'pixel_y': round(det['pixel_y'], 2),
                'uav_lat': round(uav.lat, 8),
                'uav_lon': round(uav.lon, 8),
                'uav_alt': round(uav_alt_true, 3),
                'relative_height': round(relative_height, 3),
                'target_lat': round(lat_t, 8),
                'target_lon': round(lon_t, 8),
                'target_alt': 0.0
            })

            print(f"定位成功 → 帧 {frame}, ID {obj_id} → {lat_t:.8f}, {lon_t:.8f}")

        except Exception as e:
            print(f"定位失败（帧 {frame}, ID {obj_id}）：{e}")

    # === 保存 CSV ===
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n所有目标定位完成！共 {len(results)} 个检测框")
        print(f"结果已保存：{output_csv}")
    else:
        print("没有检测到任何目标")
# ========================
# 运行入口
# ========================
if __name__ == "__main__":
    track_txt = r'E:\Data\ultralytics-yolo11-main\runs\track\bytetrack_319（0.7）\tracking_results.txt'
    log_csv   = r'G:\Darklabel\seal_project\location\sim_flight_log_corrected.csv'
    output_csv = r'G:\Darklabel\seal_project\location\实验用点\detect_targets.csv'

    localize_all_targets(track_txt, log_csv, output_csv)