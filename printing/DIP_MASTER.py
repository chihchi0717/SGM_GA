# -*- coding: utf-8 -*-
"""
RB edge analyzer (integrated)
- Keep ALL right-side sloped segments (no merge)
- Robustly fit the left wall
- For EVERY sloped segment, compute angle vs left wall
- Save visualization and CSV
"""

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import degrees
from skimage.measure import approximate_polygon
from sklearn.linear_model import LinearRegression
from scipy.ndimage import binary_fill_holes
import sys

sys.stdout.reconfigure(encoding="utf-8")
# ======================= 全域參數 =======================
# 幾何擬合 / 顯示
RDP_TOLERANCE = 11.0  # 小→角點較多→斜邊切得更細
EXTEND_SCALE = 110.0

# 形態學（kernel 固定，不自動放大）
KERNEL_SIZE = 7
CLOSE_ITER = 2
OPEN_ITER = 1

# 連通域自動調參（只增 close，不增 kernel）
MAX_COMPONENTS = 1
MIN_COMPONENT_AREA_RATIO = 0.001
MAX_RETRIES = 50
CLOSE_STEP = 6

# 主體有效性門檻（避免 components==1 但其實是雜訊）
MIN_MAIN_AREA_RATIO = 0.02
MIN_MAIN_SHORT_SIDE = 80

# 是否保留左側細長基板（預設 False）
KEEP_LEFT_SUBSTRATE = False

# I/O
input_folder = (
    r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\DOE_RB\0.6_0.9"
)
output_folder = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\DOE_RB\0.6_0.9\0810ResultFigures"
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, "batch_edge_summary.csv")

# Debug 二值圖
DEBUG_BINARY = True
debug_dir = os.path.join(output_folder, "_debug_binary")
if DEBUG_BINARY:
    os.makedirs(debug_dir, exist_ok=True)


# ======================= 小工具 =======================
def prelim_binary(gray):
    """穩定簡單版：Otsu + 小型態學"""
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=2)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)
    return b


def count_components(binary, min_area_ratio=0.001):
    h, w = binary.shape
    area = h * w
    min_area = max(1, int(min_area_ratio * area))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(int(stats[i, cv2.CC_STAT_AREA] >= min_area) for i in range(1, num))


def is_valid_main_object(binary):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return False
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    h, w = binary.shape
    img_area = h * w
    main_area = stats[idx, cv2.CC_STAT_AREA]
    bw = stats[idx, cv2.CC_STAT_WIDTH]
    bh = stats[idx, cv2.CC_STAT_HEIGHT]
    return (main_area / img_area) >= MIN_MAIN_AREA_RATIO and min(
        int(bw), int(bh)
    ) >= int(MIN_MAIN_SHORT_SIDE)


def keep_main_and_left(binary):
    """（可選）保留最大主體 + 左側細長基板"""
    if not KEEP_LEFT_SUBSTRATE:
        return binary
    h, w = binary.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_id = 1 + np.argmax(areas)
    keep = {main_id}
    for i in range(1, num):
        if i == main_id:
            continue
        x, y, ww, hh, area = (
            stats[i, 0],
            stats[i, 1],
            stats[i, 2],
            stats[i, 3],
            stats[i, cv2.CC_STAT_AREA],
        )
        touches_left = x == 0
        tall_thin = (hh >= 0.40 * h) and (ww <= 0.15 * w)
        in_left = (x + ww) <= int(0.30 * w)
        big_enough = area >= 0.005 * (h * w)
        if touches_left and tall_thin and in_left and big_enough:
            keep.add(i)
    mask = np.isin(labels, list(keep))
    mask = binary_fill_holes(mask)
    return (mask.astype(np.uint8)) * 255


def find_main_contour(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def nearest_idx(contour, pt):
    d = np.linalg.norm(contour - pt, axis=1)
    return int(np.argmin(d))


def slice_closed_contour(contour, i0, i1):
    return (
        contour[i0 : i1 + 1]
        if i0 <= i1
        else np.vstack([contour[i0:], contour[: i1 + 1]])
    )


def fit_segment(points_xy):
    """線性回歸擬合一段，回傳端點、角度（右=0°, 上=正）、視覺長度"""
    if len(points_xy) < 2:
        return None
    X = points_xy[:, 0].reshape(-1, 1)
    y = points_xy[:, 1]
    reg = LinearRegression().fit(X, y)
    x1, x2 = float(X[0, 0]), float(X[-1, 0])
    y1 = float(reg.predict([[x1]])[0])
    y2 = float(reg.predict([[x2]])[0])
    ang = degrees(np.arctan2(-(y2 - y1), (x2 - x1)))  # 右=0°, 順時針為負（y 向下）
    L = float(np.hypot(x2 - x1, y2 - y1))
    return (x1, y1), (x2, y2), ang, L


def rdp_fit_segments(contour_xy, rdp_tolerance=15.0):
    """RDP 取角點 → 相鄰角點之間各自線性擬合成段"""
    corners = approximate_polygon(contour_xy, tolerance=rdp_tolerance)
    if not np.allclose(corners[0], corners[-1]):
        corners = np.vstack([corners, corners[0]])
    segs = []
    for i in range(len(corners) - 1):
        i0 = nearest_idx(contour_xy, corners[i])
        i1 = nearest_idx(contour_xy, corners[i + 1])
        pts = slice_closed_contour(contour_xy, i0, i1)
        res = fit_segment(pts)
        if res is not None:
            p1, p2, ang, L = res
            segs.append({"p1": p1, "p2": p2, "angle": ang, "length": L})
    return segs, corners


# ---- 幾何工具 ----
def extend_line(p1, p2, scale):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    n = np.linalg.norm(v)
    if n == 0:
        return p1, p2
    v = v / n
    return p1 - v * scale, p2 + v * scale


def norm_tilt(angle_deg):
    """相對『水平』的傾斜角度，0~90（避免 ±180 的模糊）"""
    a = abs(angle_deg) % 180.0
    return a if a <= 90.0 else 180.0 - a


def angle_diff(a, b):
    """兩線夾角（0~90）"""
    d = abs((a - b) % 180.0)
    return d if d <= 90.0 else 180.0 - d


# ---- 左牆：逐列找最左白點 → 線性回歸（一次修剪） ----
def fit_left_wall_from_mask(binary, y_step=5, trim_ratio=0.08):
    h, w = binary.shape
    ys = np.arange(0, h, y_step, dtype=int)
    xs, ysel = [], []
    for y in ys:
        row = binary[y] > 0
        if row.any():
            x = int(np.argmax(row))  # 這列最左的白點
            xs.append(x)
            ysel.append(y)
    if len(xs) < max(30, h // 200):
        return None
    X = np.array(ysel, dtype=float).reshape(-1, 1)  # y → x
    Y = np.array(xs, dtype=float)
    reg = LinearRegression().fit(X, Y)
    resid = np.abs(Y - reg.predict(X))
    thr = np.quantile(resid, 1 - trim_ratio)
    keep = resid <= thr
    if keep.sum() >= 20:
        reg = LinearRegression().fit(X[keep], Y[keep])
        X = X[keep]
    y1, y2 = float(X.min()), float(X.max())
    x1 = float(reg.predict([[y1]])[0])
    x2 = float(reg.predict([[y2]])[0])
    ang = degrees(np.arctan2(-(y2 - y1), (x2 - x1)))  # 跟主角度定義一致
    return (x1, y1), (x2, y2), ang


# ---- 右側：保留所有斜邊（不合併） ----
def pick_right_slopes(
    contour_xy, segments, corners, h, angle_range=(12.0, 88.0), right_quantile=0.55
):
    """
    在右側保留所有鋸齒斜邊（不合併）：
    - 位置：x 超過 right_quantile 分位數
    - 角度：angle_range（相對水平的傾角，0~90）
    - 長度：自適應最小長度（由右側角點的垂直 pitch 推得）
    - 保底：雖短但垂直跨幅夠，也保留
    """
    # 右側位置門檻（較寬鬆）
    x_gate = float(np.quantile(contour_xy[:, 0], right_quantile))

    # 估計右側齒距的垂直 pitch -> 自適應最小長度
    rc = corners[corners[:, 0] >= x_gate]
    if len(rc) >= 4:
        dy = np.abs(np.diff(rc[:, 1]))
        dy = dy[dy > 0]
        pitch = float(np.median(dy)) if len(dy) else 0.0
    else:
        pitch = 0.0
    # 退回值：圖高的 6%
    fallback = 0.06 * h
    min_len_px = int(max(8.0, 0.3 * pitch, fallback))

    kept = []
    for s in segments:
        # 位置在右側
        cx = 0.5 * (s["p1"][0] + s["p2"][0])
        if cx < x_gate:
            continue
        # 角度（對水平的傾角）
        tilt = norm_tilt(s["angle"])
        if not (angle_range[0] <= tilt <= angle_range[1]):
            continue

        # 長度 or 垂直跨幅達標（避免因 RDP 切太細而漏掉）
        vspan = abs(s["p2"][1] - s["p1"][1])
        if (s["length"] >= min_len_px) or (vspan >= 0.6 * min_len_px):
            kept.append(s)

    return kept


# ======================= 前處理（含自動 close 疊加） =======================
def preprocess_final(gray):
    b0 = prelim_binary(gray)

    def run_once(basis_binary, k, c_iter, o_iter):
        k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        filled = cv2.morphologyEx(
            basis_binary, cv2.MORPH_CLOSE, k_big, iterations=c_iter
        )
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k_big, iterations=o_iter)
        return filled

    k_used = KERNEL_SIZE
    close_used = CLOSE_ITER
    open_used = OPEN_ITER

    binary = run_once(b0, k_used, close_used, open_used)
    comp = count_components(binary, MIN_COMPONENT_AREA_RATIO)

    tries = 0
    while comp > MAX_COMPONENTS and tries < MAX_RETRIES:
        tries += 1
        close_used += CLOSE_STEP
        binary = run_once(b0, k_used, close_used, open_used)
        comp = count_components(binary, MIN_COMPONENT_AREA_RATIO)

    if KEEP_LEFT_SUBSTRATE:
        binary = keep_main_and_left(binary)

    meta = {
        "components": comp,
        "close_iter_used": close_used,
        "kernel_used": k_used,
    }
    return binary, meta


# ======================= 單張流程 =======================
def process_one_image(img_path, save_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"file": name, "status": "read_fail"}

    binary, meta = preprocess_final(gray)

    if meta.get("components", 0) <= 1 and not is_valid_main_object(binary):
        if DEBUG_BINARY:
            cv2.imwrite(os.path.join(debug_dir, f"{name}_binary.png"), binary)
        return {
            "file": name,
            "status": "noise_or_too_small",
            "n_slopes": 0,
            "angles_vs_left_deg": "",
            "left_wall_angle_deg": "",
            "components": meta.get("components", ""),
            "close_iter_used": meta.get("close_iter_used", ""),
            "kernel_used": meta.get("kernel_used", ""),
            "result_image": "",
        }

    if DEBUG_BINARY:
        cv2.imwrite(os.path.join(debug_dir, f"{name}_binary.png"), binary)

    cnt = find_main_contour(binary)
    if cnt is None:
        return {"file": name, "status": "no_contour"}

    contour = cnt.reshape(-1, 2).astype(np.float32)
    segments, corners = rdp_fit_segments(contour, RDP_TOLERANCE)
    h, w = binary.shape

    # 擬合左牆（維持你目前的方法）
    left_line = fit_left_wall_from_mask(binary, y_step=5, trim_ratio=0.08)
    left_ang = None
    if left_line is not None:
        (lx1, ly1), (lx2, ly2), left_ang = left_line

    # ➜ 右側所有斜邊（使用新的自適應規則）
    all_slopes = pick_right_slopes(
        contour, segments, corners, h, angle_range=(12, 88), right_quantile=0.52
    )

    angles_vs_left = []
    if left_ang is not None:
        for s in all_slopes:
            angles_vs_left.append(round(float(angle_diff(s["angle"], left_ang)), 3))

    # 視覺化
    fig, ax = plt.subplots(figsize=(7.5, 10))
    ax.imshow(binary, cmap="gray", origin="upper")
    ax.plot(
        contour[:, 0],
        contour[:, 1],
        color=(0.3, 0.8, 0.6, 0.4),
        linewidth=1.0,
        label="Raw Contour",
    )
    ax.scatter(
        corners[:, 0],
        corners[:, 1],
        s=10,
        c="red",
        edgecolors="white",
        linewidths=0.5,
        label="Corner Points",
    )

    # 左牆
    if left_ang is not None:
        (lx1e, ly1e), (lx2e, ly2e) = extend_line(
            (lx1, ly1), (lx2, ly2), EXTEND_SCALE * 2
        )
        ax.plot(
            [lx1e, lx2e], [ly1e, ly2e], color="gold", linewidth=3, label="Left Wall"
        )
        ax.text(
            (lx1 + lx2) / 2,
            (ly1 + ly2) / 2,
            f"Left {left_ang:.1f}°",
            color="gold",
            fontsize=11,
        )

    # 每一節右側斜邊 + 與左牆夾角
    for s in all_slopes:
        e1, e2 = extend_line(s["p1"], s["p2"], EXTEND_SCALE)
        ax.plot(
            [e1[0], e2[0]], [e1[1], e2[1]], linestyle="--", linewidth=1.8, color="blue"
        )
        if left_ang is not None:
            mid = (np.array(s["p1"]) + np.array(s["p2"])) / 2.0
            d = angle_diff(s["angle"], left_ang)
            ax.text(mid[0], mid[1] - 18, f"∠{d:.1f}°", color="yellow", fontsize=10)

    # 左上角小標籤：comp/close/kernel
    ax.text(
        0.02,
        0.98,
        f"comp={meta.get('components','?')}, close={meta.get('close_iter_used','?')}, k={meta.get('kernel_used','?')}",
        color="cyan",
        fontsize=10,
        bbox=dict(facecolor="gray", alpha=0.3),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.set_title(name)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    plt.tight_layout()

    out_png = os.path.join(save_dir, f"{name}_fitted_extended.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "file": name,
        "status": "ok",
        "n_slopes": len(all_slopes),
        "angles_vs_left_deg": (
            ";".join(map(str, angles_vs_left)) if angles_vs_left else ""
        ),
        "left_wall_angle_deg": (
            round(float(left_ang), 3) if left_ang is not None else ""
        ),
        "components": meta.get("components", ""),
        "close_iter_used": meta.get("close_iter_used", ""),
        "kernel_used": meta.get("kernel_used", ""),
        "result_image": out_png,
    }


# ======================= 批次流程 =======================
def batch_process(input_dir, save_dir, csv_out):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in exts]
    files.sort()
    if not files:
        print("⚠️ 找不到圖片檔")
        return

    rows = []
    for i, fname in enumerate(files, 1):
        path = os.path.join(input_dir, fname)
        print(f"[{i}/{len(files)}] {fname} …")
        try:
            res = process_one_image(path, save_dir)
        except Exception as e:
            res = {"file": os.path.splitext(fname)[0], "status": f"error: {e}"}
        rows.append(res)

    fieldnames = [
        "file",
        "status",
        "n_slopes",
        "angles_vs_left_deg",
        "left_wall_angle_deg",
        "components",
        "close_iter_used",
        "kernel_used",
        "result_image",
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"✅ 完成！共處理 {len(files)} 張圖。")
    print(f"📄 CSV：{csv_out}")
    print(f"🖼 圖片輸出資料夾：{save_dir}")
    if DEBUG_BINARY:
        print(f"🧪 Debug 二值化輸出：{debug_dir}")


# ======================= 執行 =======================
if __name__ == "__main__":
    batch_process(input_folder, output_folder, csv_path)
