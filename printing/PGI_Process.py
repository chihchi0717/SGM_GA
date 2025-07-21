import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from math import atan2, degrees
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding="utf-8")
# === 載入資料 ===
df = pd.read_excel(
    r"C:\Users\cchih\Desktop\20250716\20250716_clean.xlsx", sheet_name="01"
)
x = df["X(um)"].to_numpy()
z = df["Z(um)"].to_numpy()
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

for i in range(len(features) - 1):
    x1, z1 = x[features[i]], z_inv[features[i]]
    x2, z2 = x[features[i + 1]], z_inv[features[i + 1]]
    dx = x2 - x1
    dz = z2 - z1
    angle = degrees(atan2(dz, dx))
    types.append("Upward" if dz > 0 else "Downward")
    angles.append(abs(angle))
    from_indices.append(features[i])
    to_indices.append(features[i + 1])

angle_df = pd.DataFrame(
    {"From": from_indices, "To": to_indices, "Slope Type": types, "Angle (°)": angles}
)

# === 3. 過濾不合理角度（44°~84°）並刪除首尾 ===
angle_df = angle_df[
    (angle_df["Angle (°)"] >= 44) & (angle_df["Angle (°)"] <= 84)
].reset_index(drop=True)
angle_df = angle_df.iloc[1:-1].reset_index(drop=True)

# 顯示處理後斜邊角度
print("\n=== Filtered Slope Angles ===")
print(angle_df)

# === 4. 分別計算 Upward / Downward 平均角度 ===
avg_angles = angle_df.groupby("Slope Type")["Angle (°)"].mean()
print("\n=== Average Angle by Slope Type ===")
for slope_type in avg_angles.index:
    print(f"{slope_type}: {avg_angles[slope_type]:.2f}°")
