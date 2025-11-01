import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression
from math import atan2, degrees, acos
import os, re, sys

sys.stdout.reconfigure(encoding="utf-8")


def analyze_profile(
    filepath,
    save_dir=None,
    show_plot=True,
    smooth_window=31,
    smooth_poly=3,
):
    # ---- 1) 讀檔 ----
    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    data_rows = []
    with open(filepath, encoding="ISO-8859-1") as f:
        lines = f.readlines()
    for line in lines[18:]:
        nums = re.findall(number_pattern, line)
        if len(nums) >= 2:
            try:
                data_rows.append((float(nums[0]), float(nums[1])))
            except ValueError:
                pass
    if len(data_rows) < 10:
        print(f"⚠ 資料不足：{filepath}")
        return
    df = pd.DataFrame(data_rows, columns=["Lateral um", "Raw Angstrom"])
    x0 = df["Lateral um"].to_numpy()
    z0 = df["Raw Angstrom"].to_numpy() * 0.0001  # Å→μm

    # ---- 2) 基板傾斜校正 ----
    N = len(x0)
    L = slice(0, int(N * 0.1))
    R = slice(int(N * 0.9), N)
    reg_base = LinearRegression().fit(
        np.concatenate([x0[L], x0[R]]).reshape(-1, 1), np.concatenate([z0[L], z0[R]])
    )
    base_slope = reg_base.coef_[0]
    theta = atan2(base_slope, 1)
    print(f"⚙ {os.path.basename(filepath)} 基板傾角補償：{degrees(theta):.2f}°")
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    x = x0 * cos_t - z0 * sin_t
    z = x0 * sin_t + z0 * cos_t

    # ---- 3) 平滑 ----
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window >= 5 and smooth_window < len(z):
        z_s = savgol_filter(z, smooth_window, smooth_poly)
    else:
        z_s = z.copy()

    # ---- 4) 峰谷偵測 ----
    amp_scale = np.std(z_s)
    prom = max(amp_scale * 0.2, 2.0)
    dist = max(int(0.03 * N), 20)
    peaks, _ = find_peaks(z_s, distance=dist, prominence=prom)
    valleys, _ = find_peaks(-z_s, distance=dist, prominence=prom)
    print(f"🔎 peaks={len(peaks)}, valleys={len(valleys)}")
    
    if len(peaks) < 1 or len(valleys) < 1:
        # ... (錯誤處理部分不變)
        return None

    # ---- 5) 對齊與斜率擬合 ----
    ext_idx = np.sort(np.concatenate([peaks, valleys]))
    is_peak = np.isin(ext_idx, peaks)
    if is_peak[0]:
        ext_idx = ext_idx[1:]
        is_peak = is_peak[1:]
    if len(ext_idx) < 2: return None
    if not is_peak[-1]:
        ext_idx = ext_idx[:-1]
        is_peak = is_peak[:-1]
    if len(ext_idx) < 2: return None

    trim_ratio = 0.15
    min_hspan_um = 50.0
    min_amp_um = 5.0
    min_r2 = 0.85

    results = []
    for k in range(0, len(ext_idx) - 1):
        i1, i2 = ext_idx[k], ext_idx[k + 1]
        label = ("Upward" if (not is_peak[k] and is_peak[k + 1]) else "Downward" if (is_peak[k] and not is_peak[k + 1]) else None)
        if label is None: continue
        span = i2 - i1
        if span < 10: continue
        a = int(i1 + span * trim_ratio)
        b = int(i2 - span * trim_ratio)
        if b - a < 10: continue
        xs = x[a : b + 1]
        zs = z_s[a : b + 1]
        hspan = abs(xs[-1] - xs[0])
        amp = abs(z_s[i2] - z_s[i1])
        if hspan < min_hspan_um or amp < min_amp_um: continue
        model = LinearRegression().fit(xs.reshape(-1, 1), zs)
        yhat = model.predict(xs.reshape(-1, 1))
        ss_res = np.sum((zs - yhat) ** 2)
        ss_tot = np.sum((zs - np.mean(zs)) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        if r2 < min_r2: continue
        slope = model.coef_[0]
        ang_line = abs(degrees(np.arctan2(slope, 1)))
        slope_type = "Upward" if slope > 0 else "Downward"
        if slope_type != label: continue
        results.append({"From": a, "To": b, "Slope Type": slope_type, "Angle (°)": ang_line, "R2": r2})

    print(f"✅ segments={len(results)}")
    angle_df = pd.DataFrame(results)

    # ---- 6) 視覺化 ----
    plt.figure(figsize=(10, 8))
    plt.plot(x, z, linewidth=1, alpha=0.3, label="Original Profile", color='gray')
    plt.plot(x, z_s, linewidth=1.5, alpha=0.8, label="Smoothed")

    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            plt.plot(
                [x[peaks[i]], x[peaks[i + 1]]],
                [z_s[peaks[i]], z_s[peaks[i + 1]]],
                color="lightgray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="_nolegend_",
            )

    if not angle_df.empty:
        for _, row in angle_df.iterrows():
            xs = x[row["From"] : row["To"] + 1]
            zs = z_s[row["From"] : row["To"] + 1]
            model = LinearRegression().fit(xs.reshape(-1, 1), zs)
            zpred = model.predict(xs.reshape(-1, 1))
            
            # <<< 修改：將詳細資訊直接放在圖例中
            label = f'{row["Slope Type"]} {row["Angle (°)"]:.1f}°'
            plt.plot(xs, zpred, linewidth=2, label=label)
            
            # <<< 修改：將圖上的文字標註功能完全移除

    plt.xlabel("X (μm)")
    plt.ylabel("Z (μm)")
    plt.title(f"Slope Angle by Linear Regression\n{os.path.basename(filepath)}")
    plt.legend() # <<< 修改：恢復顯示完整的圖例
    plt.axis("equal")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, os.path.splitext(os.path.basename(filepath))[0] + ".png")
        plt.savefig(fig_path, dpi=150)
        print(f"📷 儲存圖檔：{fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return angle_df if not angle_df.empty else None