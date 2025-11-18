# -------------------------------------------------
#  SNF 升级版：100% 保真 + 精准杀误检聚团
#  自动推导 5% 阈值 + 生态依据（1.5km / 0.8km / size<3）
# -------------------------------------------------
# -------------------------------------------------
#  SNF 升级版：100% 保真 + 精准杀误检聚团
# -------------------------------------------------
import math
import numpy as np
import pandas as pd
from collections import deque, namedtuple
from matplotlib.patches import Circle   # ← 正确位置！
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 数据结构
GeoPoint = namedtuple('GeoPoint', ['latitude', 'longitude'])


# Haversine 距离 (km)
def haversine(p1, p2):
    R = 6371.0
    lat1, lon1 = math.radians(p1.latitude), math.radians(p1.longitude)
    lat2, lon2 = math.radians(p2.latitude), math.radians(p2.longitude)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# 1. LOO-CV 优化带宽
def cv_bandwidth(points, candidates=np.arange(0.008, 0.025, 0.001)):
    n = len(points)
    best, best_ll = None, -np.inf
    for h in candidates:
        ll = 0.0
        for i in range(n):
            train = points[:i] + points[i + 1:]
            if not train: continue
            d = np.mean([math.exp(-haversine(points[i], p) ** 2 / (2 * h * h)) for p in train])
            ll += math.log(d + 1e-12)
        if ll > best_ll:
            best, best_ll = h, ll
    print(f"最佳带宽 h = {best:.4f} km")
    return best


# 2. 密度中心
def density_center(points, h):
    dens = [sum(math.exp(-haversine(p, q) ** 2 / (2 * h * h)) for q in points) for p in points]
    return points[np.argmax(dens)]


# 3. 聚类 (8m)
def cluster_points(points, eps_km=0.008):
    visited = [False] * len(points)
    clusters = []
    for i in range(len(points)):
        if visited[i]: continue
        cluster = []
        q = deque([i])
        visited[i] = True
        while q:
            cur = q.popleft()
            cluster.append(points[cur])
            for j in range(len(points)):
                if not visited[j] and haversine(points[cur], points[j]) <= eps_km:
                    visited[j] = True
                    q.append(j)
        if cluster:
            clusters.append(cluster)
    return clusters


# 4. SNF 增强版（自动推导阈值）
# -------------------------------------------------
#  SNF 升级版：100% 保真 + 精准杀误检聚团
#  自动推导 5% 阈值 + 生态依据（1.5km / 0.8km / size<3）
# -------------------------------------------------
# -------------------------------------------------
#  SNF 升级版：100% 保真 + 精准杀误检聚团
# -------------------------------------------------
import math
import numpy as np
import pandas as pd
from collections import deque, namedtuple
from matplotlib.patches import Circle   # ← 正确位置！
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 数据结构
GeoPoint = namedtuple('GeoPoint', ['latitude', 'longitude'])


# Haversine 距离 (km)
def haversine(p1, p2):
    R = 6371.0
    lat1, lon1 = math.radians(p1.latitude), math.radians(p1.longitude)
    lat2, lon2 = math.radians(p2.latitude), math.radians(p2.longitude)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# 1. LOO-CV 优化带宽
def cv_bandwidth(points, candidates=np.arange(0.008, 0.025, 0.001)):
    n = len(points)
    best, best_ll = None, -np.inf
    for h in candidates:
        ll = 0.0
        for i in range(n):
            train = points[:i] + points[i + 1:]
            if not train: continue
            d = np.mean([math.exp(-haversine(points[i], p) ** 2 / (2 * h * h)) for p in train])
            ll += math.log(d + 1e-12)
        if ll > best_ll:
            best, best_ll = h, ll
    print(f"最佳带宽 h = {best:.4f} km")
    return best


# 2. 密度中心
def density_center(points, h):
    dens = [sum(math.exp(-haversine(p, q) ** 2 / (2 * h * h)) for q in points) for p in points]
    return points[np.argmax(dens)]


# 3. 聚类 (8m)
def cluster_points(points, eps_km=0.008):
    visited = [False] * len(points)
    clusters = []
    for i in range(len(points)):
        if visited[i]: continue
        cluster = []
        q = deque([i])
        visited[i] = True
        while q:
            cur = q.popleft()
            cluster.append(points[cur])
            for j in range(len(points)):
                if not visited[j] and haversine(points[cur], points[j]) <= eps_km:
                    visited[j] = True
                    q.append(j)
        if cluster:
            clusters.append(cluster)
    return clusters


# 4. SNF 增强版（自动推导阈值）
def SNF_enhanced(points,
                 merge_eps=0.008,        # unit: km (e.g. 0.008 km = 8 m)
                 max_dist=1.0,           # R1: 最大允许距生态中心距离 (km)
                 core_radius_ratio=0.75, # 缩放自动推导的 core_radius
                 min_size=3,             # R3 中的小团阈值
                 density_safety_factor=0.65):
    """
    修正版 SNF（保持 R1/R2/R3 思路并增加核心区内低密度剔除）
    接口与原来一致：返回 kept, center, clusters, discarded, params
    """
    print(f"开始 SNF 处理... 原始点数: {len(points)}")

    # Step1: 带宽 + 密度中心 + 聚类
    h = cv_bandwidth(points)
    center = density_center(points, h)
    clusters = cluster_points(points, merge_eps)
    print(f"[SNF] 聚类完成: {len(clusters)} 个团 (merge_eps={merge_eps} km)")

    # Step2: 计算所有团特征（先全部计算）
    feats = []
    for c in clusters:
        centroid = GeoPoint(np.mean([p.latitude for p in c]), np.mean([p.longitude for p in c]))
        dist = haversine(centroid, center)
        size = len(c)
        # KDE 按你原始定义：每簇内点对全部点的核密度平均
        kde_vals = [sum(math.exp(-haversine(p, q) ** 2 / (2 * h * h)) for q in points) for p in c]
        density = np.mean(kde_vals) if kde_vals else 0.0
        feats.append({'centroid': centroid, 'size': size, 'dist': dist, 'density': density, 'points': c})

    if not feats:
        return [], center, clusters, [], {'h': h, 'core_radius': 0.0, 'density_threshold_ratio': 0.0}

    # Step2.5: 全局统计用于阈值推导
    densities = np.array([f['density'] for f in feats])
    sizes = np.array([f['size'] for f in feats])
    d25 = np.percentile(densities, 25)
    d30 = np.percentile(densities, 30)
    peak_density = densities.max() if len(densities) > 0 else 1.0

    # 收集“可靠团”：size >= 5 且 density > d30
    candidate_dists = [f['dist'] for f in feats if f['size'] >= 5 and f['density'] > d30]

    # 自动推导 core_radius（优先 candidate_dists 的 80% 分位，否则基于簇间中位距回退）
    if candidate_dists:
        core_radius_auto = np.percentile(candidate_dists, 80) * core_radius_ratio
    else:
        centroids = [f['centroid'] for f in feats]
        if len(centroids) >= 2:
            pair_dists = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    pair_dists.append(haversine(centroids[i], centroids[j]))
            med_pair = np.median(pair_dists) if pair_dists else 0.4
        else:
            med_pair = 0.4
        core_radius_auto = max(0.25, med_pair * 0.6) * core_radius_ratio

    core_radius = min(max(core_radius_auto, 0.05), 0.8)  # 下限 0.05km，上限 0.8km

    # 误检密度阈值推导（更严格）——基于低密度且远离的簇来估计
    false_densities = [f['density'] for f in feats if f['size'] < min_size and f['dist'] > 1.2 and f['density'] < d25]
    if false_densities:
        max_false_ratio = max(false_densities) / peak_density
    else:
        max_false_ratio = 0.08  # 备选默认

    density_threshold_ratio = max_false_ratio * density_safety_factor
    # 强制至少 5%（自动推导 5% 阈值）
    density_threshold_ratio = max(density_threshold_ratio, 0.05)
    inner_density_threshold = density_threshold_ratio * peak_density  # 绝对密度阈值用于核心区内部筛除

    print(f"[SNF] peak_density={peak_density:.4f}, core_radius={core_radius:.4f} km")
    print(f"[SNF] density_threshold_ratio={density_threshold_ratio:.4f} => inner_threshold={inner_density_threshold:.6f}")

    # Step3: 语义过滤（R1 / R2 / R2b(core内低密度) / R3）
    kept = []
    discarded = []

    for f in feats:
        reason = None
        # R1: 绝对远距离过滤
        if f['dist'] > max_dist:
            reason = f"R1: 太远 (> {max_dist}km)"
        # R2: 单点且超出核心区
        elif f['size'] == 1 and f['dist'] > core_radius:
            reason = f"R2: 孤立远点 (> {core_radius:.3f}km)"
        # R2b: **核心区内部的低密度小团** -> 增加严格性（消除核心内的误检小团）
        elif f['dist'] <= core_radius and f['size'] < max(3, min_size) and f['density'] < inner_density_threshold:
            # size<3（你提到的生态依据）或小于 min_size，且密度低于自动 5% 下限 -> 删除
            reason = f"R2b: 核心内小团低密度 (< {density_threshold_ratio*100:.1f}% 峰值)"
        # R3: 小团且密度低（只有当上面 R2/R2b 未删除时才判）
        elif f['size'] < min_size and f['density'] < (density_threshold_ratio * peak_density):
            reason = f"R3: 小团 + 低密度 (< {density_threshold_ratio*100:.1f}% 峰值)"
        # else: 保留
        if reason:
            discarded.append((f['centroid'], reason, f['size'], f['dist'], f['density']))
        else:
            kept.append(f['centroid'])

    params = {'h': h, 'core_radius': core_radius, 'density_threshold_ratio': density_threshold_ratio,
              'peak_density': peak_density, 'candidate_dists': candidate_dists}


    # === 附加输出：每个保留团的实际点数 ===
    print("\n📊 每个保留目标包含的原始点数量：")
    kept_summary = []
    for i, k in enumerate(kept):
        # 找到与 kept 对应的 feats 项（按质心坐标匹配）
        matched = next((f for f in feats if abs(f['centroid'].latitude - k.latitude) < 1e-7
                        and abs(f['centroid'].longitude - k.longitude) < 1e-7), None)
        if matched:
            kept_summary.append({
                "Target_ID": f"T{i+1}",
                "Latitude": k.latitude,
                "Longitude": k.longitude,
                "Cluster_Size": matched['size']
            })
            print(f"  T{i+1}: {matched['size']} 个点 (dist={matched['dist']:.3f}km, dens={matched['density']:.4f})")
        else:
            print(f"  T{i+1}: ⚠️ 未匹配到聚类！")

    # 也可以导出为 DataFrame
    kept_summary_df = pd.DataFrame(kept_summary)
    kept_summary_df.to_csv(r"G:\Darklabel\seal_project\location\15m\kept_summary.csv", index=False, encoding='utf-8-sig')
    print("📄 已导出保留目标聚类统计：kept_summary.csv")

    return kept, center, clusters, discarded, params

# -------------------------------------------------
# 可视化（进一步优化版：缩小图像、聚焦视图、避免标注遮挡）
# -------------------------------------------------
def visualize_snf_ultimate(points, center, kept, discarded, core_radius, params):
    # 专业颜色方案：使用柔和、对比强的调色板
    colors = {
        'bg': '#F8F9FA',       # 背景：浅灰白
        'points': '#6C757D',   # 原始点：中灰
        'center': '#DC3545',   # 中心：红色
        'kept': '#28A745',     # 保留：绿色
        'discarded': '#FD7E14',# 删除：橙色
        'core_fill': '#007BFF',# 核心填充：蓝色
        'grid': '#DEE2E6'      # 网格：浅灰
    }

    plt.figure(figsize=(12, 9), dpi=200, facecolor=colors['bg'])  # 缩小图像尺寸，避免过大

    # 计算数据范围并聚焦视图（添加小边距，避免点太小或视图过宽）
    if points:
        min_lon = min(p.longitude for p in points)
        max_lon = max(p.longitude for p in points)
        min_lat = min(p.latitude for p in points)
        max_lat = max(p.latitude for p in points)
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        margin = max(lon_range, lat_range) * 0.1  # 动态边距：数据范围的10%
        plt.xlim(min_lon - margin, max_lon + margin)
        plt.ylim(min_lat - margin, max_lat + margin)

    # ================== 1. 原始点：稍大圆点 + 柔和灰 + 微透明 ==================
    plt.scatter([p.longitude for p in points], [p.latitude for p in points],
                c=colors['points'], s=12, alpha=0.65, marker='o', edgecolors='none',
                label=f'原始定位点 ({len(points)}个)', zorder=1)

    # ================== 2. 密度中心：精致图标 + 光晕 + 优雅标注 ==================
    # 主点：红色 + 白边 + 适中大小
    plt.scatter(center.longitude, center.latitude,
                c=colors['center'], s=80, marker='D', edgecolors='white', linewidth=1.5,
                label='密度中心 (KDE)', zorder=10)

    # 多层光晕：渐变红环，减小半径避免遮挡
    for r, alpha in zip([0.0003, 0.0006, 0.0009], [0.35, 0.2, 0.1]):  # 缩小光晕范围
        circle_halo = Circle((center.longitude, center.latitude), r,
                             color=colors['center'], fill=True, alpha=alpha, zorder=9)
        plt.gca().add_patch(circle_halo)

    # 标注：缩小偏移 + 小字体 + 细箭头，避免遮挡点
    plt.annotate('生态核心区\n(全局最密集点)',
                 (center.longitude, center.latitude),
                 xytext=(15, 15), textcoords='offset points',
                 fontsize=10, color=colors['center'], weight='semibold',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=colors['center'], alpha=0.85),
                 arrowprops=dict(arrowstyle='->', color=colors['center'], lw=1, connectionstyle="arc3,rad=0.15"),
                 zorder=11)

    # ================== 3. 保留团中心：星形 + 阴影 + 简洁编号 ==================
    if kept:
        klats = [p.latitude for p in kept]
        klons = [p.longitude for p in kept]
        # 星形标记 + 黑色边 + 轻微阴影（缩小大小）
        plt.scatter(klons, klats, c='none', s=150, marker='*', edgecolors='black', linewidth=0.4, alpha=0.15, zorder=7)  # 阴影层
        plt.scatter(klons, klats, c=colors['kept'], marker='*', s=130, edgecolors='black', linewidth=0.8,
                    label=f'保留目标 ({len(kept)}个)', zorder=8)

        # 编号：小字体 + 微偏移，避免重叠
        for i, (lat, lon) in enumerate(zip(klats, klons)):
            plt.annotate(f'T{i+1}', (lon, lat), xytext=(5, 5), textcoords='offset points',
                         fontsize=9, color='white', weight='bold',
                         bbox=dict(boxstyle="circle,pad=0.15", facecolor=colors['kept'], alpha=0.75), zorder=9)

    # ================== 4. 误检团：叉形 + 气泡说明 + 避免 clutter ==================
    if discarded:
        dlats = [p[0].latitude for p in discarded]
        dlons = [p[0].longitude for p in discarded]
        plt.scatter(dlons, dlats, c=colors['discarded'], marker='X', s=60, linewidth=1.8,
                    label=f'删除误检团 ({len(discarded)}个)', zorder=6)

        # 气泡：精简文本 + 小字体 + 半透明 + 只前5个，避免 overcrowd和遮挡
        for i, (cent, reason, size, dist, dens) in enumerate(discarded[:5]):
            plt.annotate(f'×{size} | {dist:.2f}km\n{reason}',
                         (cent.longitude, cent.latitude),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=8, color='white', ha='center',
                         bbox=dict(boxstyle="round4,pad=0.3", facecolor=colors['discarded'], edgecolor='none', alpha=0.8),
                         arrowprops=dict(arrowstyle='-', color=colors['discarded'], lw=0.8, alpha=0.6),
                         zorder=7)

    # ================== 5. 核心区：渐变填充 + 虚线边 + 标签 ==================
    # 渐变填充：使用径向渐变模拟（多层圆），减小alpha避免主导视图
    for r_factor, alpha in zip([1.0, 0.8, 0.6, 0.4], [0.07, 0.05, 0.03, 0.01]):
        core_fill = Circle((center.longitude, center.latitude), (core_radius * r_factor) / 111.32,
                           color=colors['core_fill'], fill=True, alpha=alpha, zorder=0)
        plt.gca().add_patch(core_fill)

    # 边框：虚线 + 蓝色 + 细线
    core_border = Circle((center.longitude, center.latitude), core_radius / 111.32,
                         color=colors['core_fill'], fill=False, linestyle='--', linewidth=1.5,
                         label=f'核心区 ({core_radius:.3f} km)', zorder=5)
    plt.gca().add_patch(core_border)

    # ================== 6. 美化设置：专业布局 ==================
    # 轴标签：适中字体 + 粗体
    plt.xlabel('经度 (°E)', fontsize=12, weight='bold', labelpad=8)
    plt.ylabel('纬度 (°N)', fontsize=12, weight='bold', labelpad=8)

    # 标题：适中字体 + 居中 + 间距
    plt.title('SNF 终极版：生态核心区 + 团中心 + 误检剔除', fontsize=16, weight='bold', pad=20)

    # 图例：右侧外部 + 阴影 + 圆角 + 小字体
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10,
               frameon=True, fancybox=True, shadow=True, borderpad=0.6)

    # 网格：细线 + 浅色 + 后置
    plt.grid(True, color=colors['grid'], alpha=0.35, linestyle='-', linewidth=0.5, zorder=-1)

    # 轴比例 + 紧凑布局
    plt.axis('equal')
    plt.tight_layout(pad=1.5)

    # 保存高清图（PNG + 白色背景）
    output_png = r"G:\Darklabel\seal_project\location\SNF_Result_Ultimate_Optimized_Focused.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=colors['bg'])
    print(f"高清可视化已保存：{output_png}")

    plt.show()
# -------------------------------------------------
# 加载 CSV
# -------------------------------------------------
def load_points(csv_path):
    df = pd.read_csv(csv_path)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), 'latitude')
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), 'longitude')
    points = [GeoPoint(float(lat), float(lon)) for lat, lon in zip(df[lat_col], df[lon_col])
              if pd.notna(lat) and pd.notna(lon)]
    print(f"加载点数: {len(points)}")
    return points


# -------------------------------------------------
# 可视化（进一步优化版：缩小图像、聚焦视图、避免标注遮挡）
# -------------------------------------------------
def visualize_snf_ultimate(points, center, kept, discarded, core_radius, params):
    # 专业颜色方案：使用柔和、对比强的调色板
    colors = {
        'bg': '#F8F9FA',       # 背景：浅灰白
        'points': '#6C757D',   # 原始点：中灰
        'center': '#DC3545',   # 中心：红色
        'kept': '#28A745',     # 保留：绿色
        'discarded': '#FD7E14',# 删除：橙色
        'core_fill': '#007BFF',# 核心填充：蓝色
        'grid': '#DEE2E6'      # 网格：浅灰
    }

    plt.figure(figsize=(12, 9), dpi=200, facecolor=colors['bg'])  # 缩小图像尺寸，避免过大

    # 计算数据范围并聚焦视图（添加小边距，避免点太小或视图过宽）
    if points:
        min_lon = min(p.longitude for p in points)
        max_lon = max(p.longitude for p in points)
        min_lat = min(p.latitude for p in points)
        max_lat = max(p.latitude for p in points)
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        margin = max(lon_range, lat_range) * 0.1  # 动态边距：数据范围的10%
        plt.xlim(min_lon - margin, max_lon + margin)
        plt.ylim(min_lat - margin, max_lat + margin)

    # ================== 1. 原始点：稍大圆点 + 柔和灰 + 微透明 ==================
    plt.scatter([p.longitude for p in points], [p.latitude for p in points],
                c=colors['points'], s=12, alpha=0.65, marker='o', edgecolors='none',
                label=f'原始定位点 ({len(points)}个)', zorder=1)

    # ================== 2. 密度中心：精致图标 + 光晕 + 优雅标注 ==================
    # 主点：红色 + 白边 + 适中大小
    plt.scatter(center.longitude, center.latitude,
                c=colors['center'], s=80, marker='D', edgecolors='white', linewidth=1.5,
                label='密度中心 (KDE)', zorder=10)

    # 多层光晕：渐变红环，减小半径避免遮挡
    for r, alpha in zip([0.0003, 0.0006, 0.0009], [0.35, 0.2, 0.1]):  # 缩小光晕范围
        circle_halo = Circle((center.longitude, center.latitude), r,
                             color=colors['center'], fill=True, alpha=alpha, zorder=9)
        plt.gca().add_patch(circle_halo)

    # 标注：缩小偏移 + 小字体 + 细箭头，避免遮挡点
    plt.annotate('生态核心区\n(全局最密集点)',
                 (center.longitude, center.latitude),
                 xytext=(15, 15), textcoords='offset points',
                 fontsize=10, color=colors['center'], weight='semibold',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=colors['center'], alpha=0.85),
                 arrowprops=dict(arrowstyle='->', color=colors['center'], lw=1, connectionstyle="arc3,rad=0.15"),
                 zorder=11)

    # ================== 3. 保留团中心：星形 + 阴影 + 简洁编号 ==================
    if kept:
        klats = [p.latitude for p in kept]
        klons = [p.longitude for p in kept]
        # 星形标记 + 黑色边 + 轻微阴影（缩小大小）
        plt.scatter(klons, klats, c='none', s=150, marker='*', edgecolors='black', linewidth=0.4, alpha=0.15, zorder=7)  # 阴影层
        plt.scatter(klons, klats, c=colors['kept'], marker='*', s=130, edgecolors='black', linewidth=0.8,
                    label=f'保留目标 ({len(kept)}个)', zorder=8)

        # 编号：小字体 + 微偏移，避免重叠
        for i, (lat, lon) in enumerate(zip(klats, klons)):
            plt.annotate(f'T{i+1}', (lon, lat), xytext=(5, 5), textcoords='offset points',
                         fontsize=9, color='white', weight='bold',
                         bbox=dict(boxstyle="circle,pad=0.15", facecolor=colors['kept'], alpha=0.75), zorder=9)

    # ================== 4. 误检团：叉形 + 气泡说明 + 避免 clutter ==================
    if discarded:
        dlats = [p[0].latitude for p in discarded]
        dlons = [p[0].longitude for p in discarded]
        plt.scatter(dlons, dlats, c=colors['discarded'], marker='X', s=60, linewidth=1.8,
                    label=f'删除误检团 ({len(discarded)}个)', zorder=6)

        # 气泡：精简文本 + 小字体 + 半透明 + 只前5个，避免 overcrowd和遮挡
        for i, (cent, reason, size, dist, dens) in enumerate(discarded[:5]):
            plt.annotate(f'×{size} | {dist:.2f}km\n{reason}',
                         (cent.longitude, cent.latitude),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=8, color='white', ha='center',
                         bbox=dict(boxstyle="round4,pad=0.3", facecolor=colors['discarded'], edgecolor='none', alpha=0.8),
                         arrowprops=dict(arrowstyle='-', color=colors['discarded'], lw=0.8, alpha=0.6),
                         zorder=7)

    # ================== 5. 核心区：渐变填充 + 虚线边 + 标签 ==================
    # 渐变填充：使用径向渐变模拟（多层圆），减小alpha避免主导视图
    for r_factor, alpha in zip([1.0, 0.8, 0.6, 0.4], [0.07, 0.05, 0.03, 0.01]):
        core_fill = Circle((center.longitude, center.latitude), (core_radius * r_factor) / 111.32,
                           color=colors['core_fill'], fill=True, alpha=alpha, zorder=0)
        plt.gca().add_patch(core_fill)

    # 边框：虚线 + 蓝色 + 细线
    core_border = Circle((center.longitude, center.latitude), core_radius / 111.32,
                         color=colors['core_fill'], fill=False, linestyle='--', linewidth=1.5,
                         label=f'核心区 ({core_radius:.3f} km)', zorder=5)
    plt.gca().add_patch(core_border)

    # ================== 6. 美化设置：专业布局 ==================
    # 轴标签：适中字体 + 粗体
    plt.xlabel('经度 (°E)', fontsize=12, weight='bold', labelpad=8)
    plt.ylabel('纬度 (°N)', fontsize=12, weight='bold', labelpad=8)

    # 标题：适中字体 + 居中 + 间距
    plt.title('SNF 终极版：生态核心区 + 团中心 + 误检剔除', fontsize=16, weight='bold', pad=20)

    # 图例：右侧外部 + 阴影 + 圆角 + 小字体
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10,
               frameon=True, fancybox=True, shadow=True, borderpad=0.6)

    # 网格：细线 + 浅色 + 后置
    plt.grid(True, color=colors['grid'], alpha=0.35, linestyle='-', linewidth=0.5, zorder=-1)

    # 轴比例 + 紧凑布局
    plt.axis('equal')
    plt.tight_layout(pad=1.5)

    # 保存高清图（PNG + 白色背景）
    output_png = r"G:\Darklabel\seal_project\location\SNF_Result_Ultimate_Optimized_Focused.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=colors['bg'])
    print(f"高清可视化已保存：{output_png}")

    plt.show()
# -------------------------------------------------
# 加载 CSV
# -------------------------------------------------
def load_points(csv_path):
    df = pd.read_csv(csv_path)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), 'latitude')
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), 'longitude')
    points = [GeoPoint(float(lat), float(lon)) for lat, lon in zip(df[lat_col], df[lon_col])
              if pd.notna(lat) and pd.notna(lon)]
    print(f"加载点数: {len(points)}")
    return points


# -------------------------------------------------
# 主函数
# -------------------------------------------------
if __name__ == "__main__":
    csv_path = r"G:\Darklabel\seal_project\location\实验用点\detect_targets（keyong）.csv"  # 改你的路径
    # csv_path = r"G:\Darklabel\seal_project\location\实验用点\0.7置信度\track_targets(up1).csv"  # 改你的路径
    # csv_path = r"G:\Darklabel\seal_project\location\实验用点\track_targets(up1).csv"  # 改你的路径
    # csv_path = r"D:\Desktop\论文\实验\定位\seal.csv"  # 改你的路径
    points = load_points(csv_path)

    # 运行 SNF（自动推导阈值）
    kept, center, clusters, discarded, params = SNF_enhanced(
        points,
        merge_eps=0.008,
        max_dist=1.5,
        core_radius_ratio=0.7,
        min_size=3,
        density_safety_factor=0.5
    )

    # 可视化
    visualize_snf_ultimate(points, center, kept, discarded, params['core_radius'], params)

    # -------------------------------------------------
    # 导出结果 CSV
    # -------------------------------------------------
    # output_path = r"G:\Darklabel\seal_project\location\15m\position_clusing（track测试）.csv"
    output_path = r"G:\Darklabel\seal_project\location\15m\position_clusing（detect_keyong）2.csv"
    kept_df = pd.DataFrame({
        "Target_Latitude": [p.latitude for p in kept],
        "Target_Longitude": [p.longitude for p in kept]
    })
    kept_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ 已导出过滤后点位：{len(kept)} 个")
    print(f"📂 输出文件：{output_path}")
    print(f"\n最终输出 {len(kept)} 个规划点 → 路径规划超高效！")