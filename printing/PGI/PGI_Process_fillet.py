import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from math import atan2, degrees
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding="utf-8")


# === 設定參數 ===
filename = r"C:\Users\cchih\Desktop\20250716\20250716_clean.xlsx"
sheet = "01"
N = 15  # 計算圓角半徑時使用 ±N 個點

# === 讀取資料 ===
df = pd.read_excel(filename, sheet_name=sheet)
x = df["X(um)"].to_numpy()
z = df["Z(um)"].to_numpy()
z_inv = -z  # 反轉 Z 軸，讓波峰為高點

# === 1. 偵測波峰，計算稜鏡寬度 ===
peaks, _ = find_peaks(z_inv, distance=10, prominence=1)
widths = np.diff(x[peaks])
print("=== Prism Structure Widths (μm) ===")
for i, w in enumerate(widths):
    if w > 500:  # 過濾合理寬度
        print(f"Structure {i+1}: {w:.2f} μm")

# === 2. 偵測波峰與波谷，計算斜邊角度 ===
valleys, _ = find_peaks(-z_inv, distance=10, prominence=1)
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

# === 3. 過濾不合理角度並去除首尾 ===
angle_df = angle_df[
    (angle_df["Angle (°)"] >= 44) & (angle_df["Angle (°)"] <= 84)
].reset_index(drop=True)
angle_df = angle_df.iloc[1:-1].reset_index(drop=True)

# === 4. 分別對 Upward / Downward 求平均角度 ===
avg_angles = angle_df.groupby("Slope Type")["Angle (°)"].mean()
print("\n=== Average Angle by Slope Type ===")
for slope_type in avg_angles.index:
    print(f"{slope_type}: {avg_angles[slope_type]:.2f}°")


# === 5. 計算圓角半徑 ===
def curvature_radius(x_segment, z_segment):
    dx = np.gradient(x_segment)
    dz = np.gradient(z_segment, x_segment)
    ddz = np.gradient(dz, x_segment)

    numerator = (1 + dz**2) ** 1.5
    denominator = np.abs(ddz)
    denominator[denominator == 0] = np.nan  # 避免除以 0

    R = numerator / denominator
    return np.mean(R[np.isfinite(R)])  # 去除除以0的情況


# 整理結果
curvature_results = []

for idx in peaks:
    if idx - N >= 0 and idx + N < len(x):
        R = curvature_radius(x[idx - N : idx + N + 1], z_inv[idx - N : idx + N + 1])
        curvature_results.append(
            {"Index": idx, "X (μm)": x[idx], "Type": "Peak", "Radius (μm)": R}
        )

for idx in valleys:
    if idx - N >= 0 and idx + N < len(x):
        R = curvature_radius(x[idx - N : idx + N + 1], z_inv[idx - N : idx + N + 1])
        curvature_results.append(
            {"Index": idx, "X (μm)": x[idx], "Type": "Valley", "Radius (μm)": R}
        )

curvature_df = pd.DataFrame(curvature_results)
print("\n=== Radius of Curvature at Peaks and Valleys ===")
print(curvature_df)


# === 6. 分別去除離群值後計算平均圓角半徑 ===
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


# 分開處理 Peak 與 Valley
peak_df = remove_outliers_iqr(
    curvature_df[curvature_df["Type"] == "Peak"], "Radius (μm)"
)
valley_df = remove_outliers_iqr(
    curvature_df[curvature_df["Type"] == "Valley"], "Radius (μm)"
)

# 計算平均
avg_peak_radius = peak_df["Radius (μm)"].mean()
avg_valley_radius = valley_df["Radius (μm)"].mean()

print("\n=== Average Radius of Curvature (without outliers) ===")
print(f"Peak: {avg_peak_radius:.2f} μm")
print(f"Valley: {avg_valley_radius:.2f} μm")


