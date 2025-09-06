import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from math import atan2, degrees
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding="utf-8")
# === 載入資料 ===
df = pd.read_excel(
    r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202507\0728\0.9_0.9_60_01.xlsx", sheet_name="01"
)
x = df["X(mm)"].to_numpy()
z = df["Z(nm)"].to_numpy()
z_inv = -z  # 反轉Z，使峰變高點

# === 1. 偵測波峰，計算稜鏡寬度 ===
peaks, _ = find_peaks(z_inv, distance=200, prominence=1)
widths = np.diff(x[peaks])
print("=== Prism Structure Widths (μm) ===")
for i, w in enumerate(widths):
    if w > 500:  # 過濾合理寬度
        print(f"Structure {i+1}: {w:.2f} μm")

# === 2. 偵測波峰與波谷，計算斜邊角度 ===
valleys, _ = find_peaks(-z_inv, distance=10, prominence=5)
features = np.sort(np.concatenate([peaks, valleys]))

angles = []
types = []
from_indices = []
to_indices = []

M = 180 # 往內縮的點數

from sklearn.linear_model import LinearRegression

for i in range(len(features) - 1):
    idx1 = features[i]
    idx2 = features[i + 1]

    idx1_adj = min(idx1 + M, idx2 - 1)
    idx2_adj = max(idx2 - M, idx1 + 1)

    # 擬合該區間的直線斜率
    x_seg = x[idx1_adj : idx2_adj + 1].reshape(-1, 1)
    z_seg = z_inv[idx1_adj : idx2_adj + 1]

    reg = LinearRegression().fit(x_seg, z_seg)
    slope = reg.coef_[0]  # 擬合斜率
    angle = degrees(atan2(slope, 1))  # 斜率與水平線夾角

    types.append("Upward" if slope > 0 else "Downward")
    angles.append(abs(angle))
    from_indices.append(idx1_adj)
    to_indices.append(idx2_adj)


angle_df = pd.DataFrame(
    {"From": from_indices, "To": to_indices, "Slope Type": types, "Angle (°)": angles}
)

# === 3. 過濾不合理角度（44°~84°）並刪除首尾 ===
# angle_df = angle_df[
#     (angle_df["Angle (°)"] >= 44) & (angle_df["Angle (°)"] <= 84)
# ].reset_index(drop=True)
# angle_df = angle_df.iloc[1:-1].reset_index(drop=True)

# 顯示處理後斜邊角度
print("\n=== Filtered Slope Angles ===")
print(angle_df)

# === 4. 分別計算 Upward / Downward 平均角度 ===
avg_angles = angle_df.groupby("Slope Type")["Angle (°)"].mean()
print("\n=== Average Angle by Slope Type ===")
for slope_type in avg_angles.index:
    print(f"{slope_type}: {avg_angles[slope_type]:.2f}°")

# === 繪圖：原始資料與擬合線 ===
plt.figure(figsize=(12, 5))
plt.plot(x, z, color="lightgray", linewidth=1, label="Original Profile")

for idx, row in angle_df.iterrows():
    x_fit = x[row["From"] : row["To"] + 1]
    z_fit = z[row["From"] : row["To"] + 1]

    # 線性回歸擬合再繪圖（注意用原始 z）
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
