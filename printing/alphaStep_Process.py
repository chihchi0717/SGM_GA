import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from math import atan2, degrees
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")

# === 載入資料 ===
df = pd.read_excel(
    r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202507\0728\0.9_0.9_60_01.xlsx",
    sheet_name="01",
)

# === 單位轉換 ===
x = df["X(mm)"].to_numpy() * 1000  # mm → μm
z = df["Z(nm)"].to_numpy() * 0.001  # nm → μm
z_inv = -z  # 反轉Z，使峰變高點（方便偵測）

# === 1. 偵測波峰，計算稜鏡寬度 ===
peaks, _ = find_peaks(z_inv, distance=300, prominence=1)
widths = np.diff(x[peaks])
print("=== Prism Structure Widths (μm) ===")
for i, w in enumerate(widths):
    if w > 500:  # 過濾合理寬度
        print(f"Structure {i+1}: {w:.2f} μm")

# === 2. 偵測波峰與波谷，配對成 Upward / Downward 區間 ===
valleys, _ = find_peaks(-z_inv, distance=300, prominence=1)

# === 建立 peak→valley 與 valley→peak 對 ===
pairs = []
for p in peaks:
    v_after = valleys[valleys > p]
    if len(v_after) > 0:
        pairs.append((p, v_after[0]))

for v in valleys:
    p_after = peaks[peaks > v]
    if len(p_after) > 0:
        pairs.append((v, p_after[0]))

# === 3. 擬合每對邊並計算角度 ===
M = 80  # 往內縮的點數
angles, types, from_indices, to_indices = [], [], [], []
for idx1, idx2 in pairs:
    if idx2 - idx1 < 2 * M:
        continue

    idx1_adj = idx1 + M
    idx2_adj = idx2 - M

    x_seg = x[idx1_adj : idx2_adj + 1].reshape(-1, 1)
    z_seg = z[idx1_adj : idx2_adj + 1]  # <<< 用原始 z 擬合

    if len(x_seg) == 0:
        continue

    reg = LinearRegression().fit(x_seg, z_seg)
    slope = reg.coef_[0]
    angle = degrees(atan2(slope, 1))  # 正斜率代表 Upward

    slope_type = "Upward" if slope > 0 else "Downward"

    types.append(slope_type)
    angles.append(abs(angle))
    from_indices.append(idx1_adj)
    to_indices.append(idx2_adj)

# === 4. 整理成資料表 ===
angle_df = pd.DataFrame(
    {"From": from_indices, "To": to_indices, "Slope Type": types, "Angle (°)": angles}
)

# === 顯示結果 ===
print("\n=== Filtered Slope Angles ===")
print(angle_df)

# === 5. 平均角度統計 ===
avg_angles = angle_df.groupby("Slope Type")["Angle (°)"].mean()
print("\n=== Average Angle by Slope Type ===")
for slope_type in avg_angles.index:
    print(f"{slope_type}: {avg_angles[slope_type]:.2f}°")


# === 6. 畫圖：原始資料與擬合線 ===
plt.figure(figsize=(8, 8))
plt.plot(x, z, color="lightgray", linewidth=1, label="Original Profile")

# 👉 新增：底邊（波谷→波谷）
# === 繪製波峰之間的上邊線（Peak-to-Peak） ===
peaks_sorted = np.sort(peaks)
for i in range(len(peaks_sorted) - 1):
    idx1 = peaks_sorted[i]
    idx2 = peaks_sorted[i + 1]
    x1, z1 = x[idx1], z[idx1]
    x2, z2 = x[idx2], z[idx2]
    plt.plot(
        [x1, x2], [z1, z2], color="purple", linestyle="--", linewidth=1.5, alpha=0.7
    )


for idx, row in angle_df.iterrows():
    x_fit = x[row["From"] : row["To"] + 1]
    z_fit = z[row["From"] : row["To"] + 1]
    model = LinearRegression().fit(x_fit.reshape(-1, 1), z_fit)
    z_pred = model.predict(x_fit.reshape(-1, 1))
    plt.plot(
        x_fit, z_pred, label=f'{row["Slope Type"]} {row["Angle (°)"]:.1f}°', linewidth=2
    )

plt.xlabel("X (μm)")
plt.ylabel("Z (μm)")
plt.title("Slope Angle by Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()

# === 7. 計算每條斜邊與底邊的夾角 ===
base_angles = []  # 用來存夾角結果

for idx, row in angle_df.iterrows():
    x1, z1 = x[row["From"]], z[row["From"]]
    x2, z2 = x[row["To"]], z[row["To"]]

    # === 斜邊向量 ===
    dx1 = x2 - x1
    dz1 = z2 - z1
    v1 = np.array([dx1, dz1])

    # === 找最近的兩個波峰，用來定義底邊 ===
    peak_before = peaks[peaks < row["From"]]
    peak_after = peaks[peaks > row["To"]]
    if len(peak_before) == 0 or len(peak_after) == 0:
        base_angles.append(np.nan)
        continue

    idx_p1 = peak_before[-1]
    idx_p2 = peak_after[0]
    xb1, zb1 = x[idx_p1], z[idx_p1]
    xb2, zb2 = x[idx_p2], z[idx_p2]
    dx2 = xb2 - xb1
    dz2 = zb2 - zb1
    v2 = np.array([dx2, dz2])

    # === 計算夾角（單位向量內積 + arccos）===
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    cos_theta = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    theta = degrees(np.arccos(cos_theta))  # in degrees

    base_angles.append(theta)

# === 加入資料表 ===
angle_df["Angle w.r.t Base (°)"] = base_angles

# === 顯示更新後的資料表 ===
print("\n=== Slope Angles vs Base Line ===")
print(angle_df[["Slope Type", "Angle (°)", "Angle w.r.t Base (°)"]])
