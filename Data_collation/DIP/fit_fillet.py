FOLDER_PATH = (
    r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\DOE_RB\0.6_0.9"
)
OUTPUT_FOLDER = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\DOE_RB\0.6_0.9\results_27"

PX_TO_UM = 1.12  # 1 px = 幾 um；若不換算寫 0
SMOOTH_WIN = 31  # 51  # 右邊界平滑視窗(odd)
MIN_GAP = 100  # 相鄰尖/谷的最小 y 間距(去噪)
HALF_WINDOW = 27 # 擬合圓弧的半窗大小(像素)

# 過大圓防呆參數
R_MAX_FACTOR = 2.2  # R_max = R_MAX_FACTOR * HALF_WINDOW
R_MIN_PX = 5.0  # 太小的圓也忽略
RMSE_MAX = 50  # 擬合殘差上限
MIN_ARC_DEG = 10.0  # 擬合區段中央角下限
# ===================================

import os, csv
import numpy as np
import cv2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = max(3, int(w))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xpad = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    ker = np.ones(w, dtype=np.float64) / w
    return np.convolve(xpad, ker, mode="same")[pad:-pad]


def signed_zero_crossings(dx: np.ndarray):
    sign = np.sign(dx)
    sign[sign == 0] = 1
    prev = np.roll(sign, 1)
    peaks = np.where((prev > 0) & (sign < 0))[0]
    valleys = np.where((prev < 0) & (sign > 0))[0]
    return peaks[peaks > 0].tolist(), valleys[valleys > 0].tolist()


def dedup_by_gap(indices, min_gap):
    indices = sorted(int(i) for i in indices)
    res = []
    for i in indices:
        if not res or (i - res[-1]) >= min_gap:
            res.append(i)
    return res


def choose_struct_mask(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask1 = 255 - th
    mask2 = th
    return mask1 if mask1.sum() >= mask2.sum() else mask2


def right_edge_from_mask(mask):
    H, W = mask.shape
    xr = np.zeros(H, dtype=np.int32)
    for y in range(H):
        xs = np.flatnonzero(mask[y] > 0)
        xr[y] = xs.max() if xs.size else 0
    return xr


def fit_circle_least_squares(points: np.ndarray):
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c3 = c
    r = float(np.sqrt(max(1e-12, c3 + cx**2 + cy**2)))
    return float(cx), float(cy), r


def arc_angle_deg(cx, cy, p1, p2):
    v1 = np.array([p1[0] - cx, p1[1] - cy], dtype=np.float64)
    v2 = np.array([p2[0] - cx, p2[1] - cy], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def annotate_and_save(img, xr, peaks, valleys, half_window, px_to_um, out_stem):
    H, W = img.shape
    edge_points = np.column_stack([xr, np.arange(H)])
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    R_MAX = R_MAX_FACTOR * half_window

    def local_fit(y0):
        y_min = max(0, y0 - half_window)
        y_max = min(H - 1, y0 + half_window)
        local = edge_points[y_min : y_max + 1]
        if len(local) < 10:
            return None
        cx, cy, r = fit_circle_least_squares(local)
        d = np.sqrt((local[:, 0] - cx) ** 2 + (local[:, 1] - cy) ** 2)
        rmse = float(np.sqrt(np.mean((d - r) ** 2)))
        theta = arc_angle_deg(cx, cy, local[0], local[-1])
        if (r < R_MIN_PX) or (r > R_MAX) or (rmse > RMSE_MAX) or (theta < MIN_ARC_DEG):
            return None
        return cx, cy, r, rmse, theta

    rows = []
    # Peaks
    for y0 in peaks:
        fit = local_fit(y0)
        if not fit:
            continue
        cx, cy, r, rmse, theta = fit
        cv2.circle(rgb, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
        cv2.circle(rgb, (int(xr[y0]), int(y0)), 4, (0, 0, 255), -1)
        label = f"Peak R={r:.1f}px" + (f" / {r*px_to_um:.1f}um" if px_to_um else "")
        cv2.putText(
            rgb,
            label,
            (max(0, int(xr[y0] - 220)), int(y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        rows.append(
            ["peak", y0, cx, cy, r, rmse, theta, (r * px_to_um if px_to_um else "")]
        )

    # Valleys
    for y0 in valleys:
        fit = local_fit(y0)
        if not fit:
            continue
        cx, cy, r, rmse, theta = fit
        cv2.circle(rgb, (int(cx), int(cy)), int(r), (0, 165, 255), 2)
        cv2.circle(rgb, (int(xr[y0]), int(y0)), 4, (0, 165, 255), -1)
        label = f"Valley R={r:.1f}px" + (f" / {r*px_to_um:.1f}um" if px_to_um else "")
        cv2.putText(
            rgb,
            label,
            (max(0, int(xr[y0] - 260)), int(y0 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
        rows.append(
            ["valley", y0, cx, cy, r, rmse, theta, (r * px_to_um if px_to_um else "")]
        )

    out_img = os.path.join(OUTPUT_FOLDER, os.path.basename(out_stem) + "_radii.png")
    cv2.imwrite(out_img, rgb)
    out_csv = os.path.join(OUTPUT_FOLDER, os.path.basename(out_stem) + "_radii.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "type",
                "y_row",
                "center_x",
                "center_y",
                "radius_px",
                "rmse_px",
                "arc_deg",
                "radius_um",
            ]
        )
        for row in rows:
            w.writerow(row)

    print("Saved:", out_img, "and", out_csv)
    return rows


def process_one(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("讀不到影像:", img_path)
        return
    mask = choose_struct_mask(img)
    xr = right_edge_from_mask(mask)
    xr_s = moving_average(xr, SMOOTH_WIN)
    dx = np.gradient(xr_s)
    peaks_raw, valleys_raw = signed_zero_crossings(dx)
    peaks = dedup_by_gap(peaks_raw, MIN_GAP)
    valleys = dedup_by_gap(valleys_raw, MIN_GAP)
    stem = os.path.splitext(img_path)[0] + "_annotated"
    annotate_and_save(img, xr, peaks, valleys, HALF_WINDOW, PX_TO_UM, stem)


def main():
    for fn in os.listdir(FOLDER_PATH):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            process_one(os.path.join(FOLDER_PATH, fn))


if __name__ == "__main__":
    main()
