import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression
from math import atan2, degrees, acos
import os, re, sys

sys.stdout.reconfigure(encoding="utf-8")


def _linear_fit_r2(x, y):
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    yhat = model.predict(x.reshape(-1, 1))
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return model, r2


def analyze_profile(
    filepath,
    save_dir=None,
    show_plot=True,
    smooth_window=31,
    smooth_poly=3,
    min_hspan_um=100,
    min_amp_um=15,
    trim_ratio=0.15,
    min_r2=0.7,
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

    # ---- 2) 基板傾斜校正（修掉覆寫 bug）----
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

    # 4) 峰谷偵測（自適應且較鬆）
    amp_scale = np.std(z_s)
    prom = max(amp_scale * 0.4, 5.0)  # 原本 0.6 → 0.4
    dist = max(int(0.06 * N), 50)  # 原本 0.08N → 0.06N
    # 這邊直接對平滑後的 z_s 做峰/谷（不取反號，語意直觀：峰=最高、谷=最低）
    peaks, _ = find_peaks(z_s, distance=dist, prominence=prom)
    valleys, _ = find_peaks(-z_s, distance=dist, prominence=prom)
    print(f"🔎 peaks={len(peaks)}, valleys={len(valleys)}")

    # 5) 對齊成 V-P-V-P... 序列，逐「週期」擬合
    # 若起點不是谷，丟掉第一個峰；若終點不是峰，丟掉最後一個谷
    ext_idx = np.sort(np.concatenate([peaks, valleys]))
    is_peak = np.isin(ext_idx, peaks)
    # 讓第一個是谷
    if is_peak[0]:
        ext_idx = ext_idx[1:]
        is_peak = is_peak[1:]
    # 讓最後一個是峰
    if not is_peak[-1]:
        ext_idx = ext_idx[:-1]
        is_peak = is_peak[:-1]

    # 參數（比全域門檻更寬鬆）
    trim_ratio = 0.20
    min_hspan_um = 150.0  # 原 300 → 150
    min_amp_um = 8.0  # 原 15  → 8
    min_r2 = 0.92  # 原 0.96 → 0.92

    results = []
    for k in range(0, len(ext_idx) - 1):
        i1, i2 = ext_idx[k], ext_idx[k + 1]
        label = (
            "Upward"
            if (not is_peak[k] and is_peak[k + 1])
            else "Downward" if (is_peak[k] and not is_peak[k + 1]) else None
        )
        if label is None:
            continue

        # 區段內修邊（避免靠近尖端的曲率）
        span = i2 - i1
        if span < 10:
            continue
        a = int(i1 + span * trim_ratio)
        b = int(i2 - span * trim_ratio)
        if b - a < 10:
            continue

        xs = x[a : b + 1]
        zs = z_s[a : b + 1]
        hspan = abs(xs[-1] - xs[0])
        amp = abs(z_s[i2] - z_s[i1])
        if hspan < min_hspan_um or amp < min_amp_um:
            continue

        # 線性擬合 + R²
        model = LinearRegression().fit(xs.reshape(-1, 1), zs)
        yhat = model.predict(xs.reshape(-1, 1))
        ss_res = np.sum((zs - yhat) ** 2)
        ss_tot = np.sum((zs - np.mean(zs)) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        if r2 < min_r2:
            continue

        slope = model.coef_[0]
        ang_line = abs(degrees(np.arctan2(slope, 1)))
        slope_type = "Upward" if slope > 0 else "Downward"
        if slope_type != label:
            # 若方向和理論(V→P=上斜、P→V=下斜)不一致則略過
            continue

        # 與底邊夾角（用鄰近兩個「峰」定義）
        angle_base = np.nan
        if len(peaks) >= 2:
            if slope_type == "Upward":
                rp = peaks[peaks >= i2]
                lp = peaks[peaks < i2]
                if len(rp) > 0 and len(lp) > 0:
                    v2 = np.array([x[rp[0]] - x[lp[-1]], z_s[rp[0]] - z_s[lp[-1]]])
                    v1 = np.array([xs[-1] - xs[0], yhat[-1] - yhat[0]])
                    cosv = np.clip(
                        np.dot(v1, v2)
                        / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12),
                        -1,
                        1,
                    )
                    angle_base = degrees(np.arccos(cosv))
            else:
                lp = peaks[peaks <= i1]
                rp = peaks[peaks > i1]
                if len(lp) > 0 and len(rp) > 0:
                    v2 = np.array([x[rp[0]] - x[lp[-1]], z_s[rp[0]] - z_s[lp[-1]]])
                    v1 = np.array([xs[-1] - xs[0], yhat[-1] - yhat[0]])
                    cosv = np.clip(
                        np.dot(v1, v2)
                        / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12),
                        -1,
                        1,
                    )
                    angle_base = degrees(np.arccos(cosv))

        results.append(
            {
                "From": a,
                "To": b,
                "Slope Type": slope_type,
                "Angle (°)": ang_line,
                "R2": r2,
                "Horiz. Span (um)": hspan,
                "Amplitude (um)": amp,
                "Angle w.r.t Base (°)": angle_base,
            }
        )

    print(f"✅ segments={len(results)}")

    angle_df = pd.DataFrame(results)

    # 6) 視覺化：底邊虛線 + 擬合線 + 角度標註
    plt.figure(figsize=(8, 8))
    plt.plot(x, z, linewidth=1, alpha=0.25, label="Original Profile")
    plt.plot(x, z_s, linewidth=1.2, alpha=0.9, label="Smoothed")

    # 相鄰峰作為底邊
    for i in range(len(peaks) - 1):
        plt.plot(
            [x[peaks[i]], x[peaks[i + 1]]],
            [z_s[peaks[i]], z_s[peaks[i + 1]]],
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label="_nolegend_",
        )

    # 擬合+標註
    for row in results:
        xs = x[row["From"] : row["To"] + 1]
        zs = z_s[row["From"] : row["To"] + 1]
        mdl = LinearRegression().fit(xs.reshape(-1, 1), zs)
        zpred = mdl.predict(xs.reshape(-1, 1))
        plt.plot(
            xs, zpred, linewidth=2, label=f'{row["Slope Type"]} {row["Angle (°)"]:.1f}°'
        )

        xm = 0.5 * (xs[0] + xs[-1])
        zm = 0.5 * (zpred[0] + zpred[-1])
        txt = (
            (
                f'∠base {row["Angle w.r.t Base (°)"]:.1f}°\n'
                f'({row["Slope Type"]} {row["Angle (°)"]:.1f}°)'
            )
            if not np.isnan(row["Angle w.r.t Base (°)"])
            else f'{row["Slope Type"]} {row["Angle (°)"]:.1f}°'
        )
        dx = (xs[-1] - xs[0]) * 0.02
        dy = (zpred[-1] - zpred[0]) * 0.02
        plt.text(xm + dx, zm + dy, txt, fontsize=9)

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


# def analyze_profile(filepath, save_dir=None, show_plot=True):
#     # === 1. 讀檔並解析數據 ===
#     number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
#     data_rows = []

#     with open(filepath, encoding="ISO-8859-1") as f:
#         lines = f.readlines()

#     for line in lines[18:]:  # 從第19行開始
#         numbers = re.findall(number_pattern, line)
#         if len(numbers) >= 2:
#             try:
#                 lateral = float(numbers[0])
#                 angstrom = float(numbers[1])
#                 data_rows.append((lateral, angstrom))
#             except ValueError:
#                 continue

#     if len(data_rows) < 10:
#         print(f"⚠ 資料不足：{filepath}")
#         return

#     df = pd.DataFrame(data_rows, columns=["Lateral um", "Raw Angstrom"])
#     x = df["Lateral um"].to_numpy()
#     z = df["Raw Angstrom"].to_numpy() * 0.0001  # Å → μm

#     # === 2. 基板傾斜校正 ===
#     N = len(x)
#     left_range = slice(0, int(N * 0.1))
#     right_range = slice(int(N * 0.9), N)
#     x_base = np.concatenate([x[left_range], x[right_range]])
#     z_base = np.concatenate([z[left_range], z[right_range]])

#     reg_base = LinearRegression().fit(x_base.reshape(-1, 1), z_base)
#     base_slope = reg_base.coef_[0]
#     base_angle = degrees(atan2(base_slope, 1))
#     print(f"⚙ {os.path.basename(filepath)} 基板傾角補償：{base_angle:.2f}°")

#     theta = atan2(base_slope, 1)
#     cos_t, sin_t = np.cos(-theta), np.sin(-theta)
#     x = x * cos_t - z * sin_t
#     z = x * sin_t + z * cos_t
#     z_inv = -z

#     # === 3. 偵測波峰波谷 ===
#     peaks, _ = find_peaks(z_inv, distance=300, prominence=100)
#     valleys, _ = find_peaks(-z_inv, distance=300, prominence=100)

#     pairs = []
#     for p in peaks:
#         v_after = valleys[valleys > p]
#         if len(v_after) > 0:
#             pairs.append((p, v_after[0]))
#     for v in valleys:
#         p_after = peaks[peaks > v]
#         if len(p_after) > 0:
#             pairs.append((v, p_after[0]))

#     # === 4. 擬合斜邊、計算角度 ===
#     M = 30
#     results = []
#     for idx1, idx2 in pairs:
#         if idx2 - idx1 < 2 * M:
#             continue
#         idx1_adj = idx1 + M
#         idx2_adj = idx2 - M
#         x_seg = x[idx1_adj : idx2_adj + 1].reshape(-1, 1)
#         z_seg = z[idx1_adj : idx2_adj + 1]
#         if len(x_seg) == 0:
#             continue
#         reg = LinearRegression().fit(x_seg, z_seg)
#         slope = reg.coef_[0]
#         angle = degrees(atan2(slope, 1))
#         slope_type = "Upward" if slope > 0 else "Downward"

#         # === 與底邊角度 ===
#         peak_before = peaks[peaks < idx1_adj]
#         peak_after = peaks[peaks > idx2_adj]
#         if len(peak_before) == 0 or len(peak_after) == 0:
#             angle_w_base = np.nan
#         else:
#             xb1, zb1 = x[peak_before[-1]], z[peak_before[-1]]
#             xb2, zb2 = x[peak_after[0]], z[peak_after[0]]
#             v1 = np.array([x[idx2_adj] - x[idx1_adj], z[idx2_adj] - z[idx1_adj]])
#             v2 = np.array([xb2 - xb1, zb2 - zb1])
#             unit_v1 = v1 / np.linalg.norm(v1)
#             unit_v2 = v2 / np.linalg.norm(v2)
#             cos_theta = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
#             angle_w_base = degrees(acos(cos_theta))

#         results.append(
#             {
#                 "From": idx1_adj,
#                 "To": idx2_adj,
#                 "Slope Type": slope_type,
#                 "Angle (°)": abs(angle),
#                 "Angle w.r.t Base (°)": angle_w_base,
#             }
#         )

#     angle_df = pd.DataFrame(results)

#     # === 5. 畫圖 ===
#     plt.figure(figsize=(8, 8))
#     plt.plot(x, z, color="lightgray", linewidth=1, label="Original Profile")

#     # 畫底邊虛線
#     for i in range(len(peaks) - 1):
#         plt.plot(
#             [x[peaks[i]], x[peaks[i + 1]]],
#             [z[peaks[i]], z[peaks[i + 1]]],
#             color="purple",
#             linestyle="--",
#             linewidth=1.5,
#             alpha=0.6,
#         )

#     # 畫擬合線
#     for row in results:
#         x_fit = x[row["From"] : row["To"] + 1]
#         z_fit = z[row["From"] : row["To"] + 1]
#         model = LinearRegression().fit(x_fit.reshape(-1, 1), z_fit)
#         z_pred = model.predict(x_fit.reshape(-1, 1))
#         plt.plot(
#             x_fit,
#             z_pred,
#             label=f'{row["Slope Type"]} {row["Angle (°)"]:.1f}°',
#             linewidth=2,
#         )

#     plt.xlabel("X (μm)")
#     plt.ylabel("Z (μm)")
#     plt.title(f"Slope Angle by Linear Regression\n{os.path.basename(filepath)}")
#     plt.legend()
#     plt.axis("equal")
#     plt.tight_layout()

#     # 儲存圖檔
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         fig_path = os.path.join(
#             save_dir, os.path.splitext(os.path.basename(filepath))[0] + ".png"
#         )
#         plt.savefig(fig_path)
#         print(f"📷 儲存圖檔：{fig_path}")

#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

#     return angle_df
