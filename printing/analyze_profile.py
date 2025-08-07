import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from math import atan2, degrees, acos
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

def analyze_profile(filepath, save_dir=None, show_plot=True):
    # === 1. 讀檔並解析數據 ===
    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    data_rows = []

    with open(filepath, encoding="ISO-8859-1") as f:
        lines = f.readlines()

    for line in lines[18:]:  # 從第19行開始
        numbers = re.findall(number_pattern, line)
        if len(numbers) >= 2:
            try:
                lateral = float(numbers[0])
                angstrom = float(numbers[1])
                data_rows.append((lateral, angstrom))
            except ValueError:
                continue

    if len(data_rows) < 10:
        print(f"⚠ 資料不足：{filepath}")
        return

    df = pd.DataFrame(data_rows, columns=["Lateral um", "Raw Angstrom"])
    x = df["Lateral um"].to_numpy()
    z = df["Raw Angstrom"].to_numpy() * 0.0001  # Å → μm

    # === 2. 基板傾斜校正 ===
    N = len(x)
    left_range = slice(0, int(N * 0.1))
    right_range = slice(int(N * 0.9), N)
    x_base = np.concatenate([x[left_range], x[right_range]])
    z_base = np.concatenate([z[left_range], z[right_range]])

    reg_base = LinearRegression().fit(x_base.reshape(-1, 1), z_base)
    base_slope = reg_base.coef_[0]
    base_angle = degrees(atan2(base_slope, 1))
    print(f"⚙ {os.path.basename(filepath)} 基板傾角補償：{base_angle:.2f}°")

    theta = atan2(base_slope, 1)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    x = x * cos_t - z * sin_t
    z = x * sin_t + z * cos_t
    z_inv = -z

    # === 3. 偵測波峰波谷 ===
    peaks, _ = find_peaks(z_inv, distance=400, prominence=100)
    valleys, _ = find_peaks(-z_inv, distance=400, prominence=100)

    pairs = []
    for p in peaks:
        v_after = valleys[valleys > p]
        if len(v_after) > 0:
            pairs.append((p, v_after[0]))
    for v in valleys:
        p_after = peaks[peaks > v]
        if len(p_after) > 0:
            pairs.append((v, p_after[0]))

    # === 4. 擬合斜邊、計算角度 ===
    M = 80
    results = []
    for idx1, idx2 in pairs:
        if idx2 - idx1 < 2 * M:
            continue
        idx1_adj = idx1 + M
        idx2_adj = idx2 - M
        x_seg = x[idx1_adj : idx2_adj + 1].reshape(-1, 1)
        z_seg = z[idx1_adj : idx2_adj + 1]
        if len(x_seg) == 0:
            continue
        reg = LinearRegression().fit(x_seg, z_seg)
        slope = reg.coef_[0]
        angle = degrees(atan2(slope, 1))
        slope_type = "Upward" if slope > 0 else "Downward"

        # === 與底邊角度 ===
        peak_before = peaks[peaks < idx1_adj]
        peak_after = peaks[peaks > idx2_adj]
        if len(peak_before) == 0 or len(peak_after) == 0:
            angle_w_base = np.nan
        else:
            xb1, zb1 = x[peak_before[-1]], z[peak_before[-1]]
            xb2, zb2 = x[peak_after[0]], z[peak_after[0]]
            v1 = np.array([x[idx2_adj] - x[idx1_adj], z[idx2_adj] - z[idx1_adj]])
            v2 = np.array([xb2 - xb1, zb2 - zb1])
            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            cos_theta = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
            angle_w_base = degrees(acos(cos_theta))

        results.append(
            {
                "From": idx1_adj,
                "To": idx2_adj,
                "Slope Type": slope_type,
                "Angle (°)": abs(angle),
                "Angle w.r.t Base (°)": angle_w_base,
            }
        )

    angle_df = pd.DataFrame(results)

    # === 5. 畫圖 ===
    plt.figure(figsize=(8, 8))
    plt.plot(x, z, color="lightgray", linewidth=1, label="Original Profile")

    # 畫底邊虛線
    for i in range(len(peaks) - 1):
        plt.plot(
            [x[peaks[i]], x[peaks[i + 1]]],
            [z[peaks[i]], z[peaks[i + 1]]],
            color="purple",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )

    # 畫擬合線
    for row in results:
        x_fit = x[row["From"] : row["To"] + 1]
        z_fit = z[row["From"] : row["To"] + 1]
        model = LinearRegression().fit(x_fit.reshape(-1, 1), z_fit)
        z_pred = model.predict(x_fit.reshape(-1, 1))
        plt.plot(
            x_fit,
            z_pred,
            label=f'{row["Slope Type"]} {row["Angle (°)"]:.1f}°',
            linewidth=2,
        )

    plt.xlabel("X (μm)")
    plt.ylabel("Z (μm)")
    plt.title(f"Slope Angle by Linear Regression\n{os.path.basename(filepath)}")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    # 儲存圖檔
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(
            save_dir, os.path.splitext(os.path.basename(filepath))[0] + ".png"
        )
        plt.savefig(fig_path)
        print(f"📷 儲存圖檔：{fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return angle_df
