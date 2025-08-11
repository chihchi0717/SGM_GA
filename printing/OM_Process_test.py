import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from math import atan2, degrees
from sklearn.linear_model import LinearRegression
import os
from skimage.restoration import denoise_tv_chambolle


def analyze_feature_slopes_auto(
    image_path,
    px_size=0.56,
    z_scale=0.56,
    base_prominence_ratio=0.05,  # x方向變化量比例
    base_distance_ratio=0.1,  # z高度比例
    min_segment_points=20,
    adaptive_margin_ratio=0.15,  # segment margin
    r2_tiers=(0.95, 0.90, 0.85),  # 對應長度階層的r²門檻
):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 錯誤：無法讀取影像路徑 {image_path}")
        return

    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # 膨脹幫助上方連接
    binary_dilate = cv2.dilate(binary, kernel, iterations=3)
    binary_close = cv2.morphologyEx(
        binary_dilate, cv2.MORPH_CLOSE, kernel, iterations=5
    )
    binary_clean = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel, iterations=2)
    # plt.imshow(binary_clean, cmap="gray")
    # plt.title("Binary Image (After Otsu Thresholding)")
    # plt.axis("off")
    # plt.show()
    H, W = binary.shape

    # === 1. 擷取所有輪廓 ===
    contours, _ = cv2.findContours(
        binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # === 2. 排除雜訊，挑出主輪廓 ===
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    main_contour = max(filtered, key=cv2.contourArea)

    # === 3. 建立 y → x 對應的右輪廓
    profile_right_px_dict = {}
    for pt in main_contour:
        x, y = pt[0]
        if y not in profile_right_px_dict or x > profile_right_px_dict[y]:
            profile_right_px_dict[y] = x

    # === 4. 同時補左輪廓（基板）或視需要排除
    valid_z_indices = sorted(profile_right_px_dict.keys())
    profile_right_px = [profile_right_px_dict[y] for y in valid_z_indices]
    profile_left_px = [np.where(binary[y] == 255)[0][0] if np.any(binary[y] == 255) else 0 for y in valid_z_indices]

    z_vals = np.array(valid_z_indices) * z_scale
    x_vals_left = np.array(profile_left_px) * px_size
    x_vals_right_raw = np.array(profile_right_px) * px_size

    raw_len = len(x_vals_right_raw)
    target_ratio = 0.03  # 或 0.03 ~ 0.1 看你圖的鋸齒程度
    estimated_window = int(raw_len * target_ratio)
    if estimated_window % 2 == 0:
        estimated_window += 1
    window_length = max(11, min(estimated_window, 201))

    # 平滑右側 profile
    # x_tv = denoise_tv_chambolle(x_vals_right_raw, weight=0.1)
    # x_vals_right_smoothed = savgol_filter(x_tv, window_length=81, polyorder=2)
    x_vals_right_smoothed = savgol_filter(
        x_vals_right_raw,
        window_length=11,
        # window_length=min(171, len(x_vals_right_raw) // 2 * 2 + 1),
        polyorder=3,
        mode="mirror",
    )

    print(f"window_length: {window_length}")

    # 基底線擬合
    reg_base = LinearRegression().fit(z_vals.reshape(-1, 1), x_vals_left.reshape(-1, 1))
    slope_base = reg_base.coef_[0][0]
    angle_base = degrees(atan2(slope_base, 1))

    # === 自動參數化 ===
    x_range = np.max(x_vals_right_smoothed) - np.min(x_vals_right_smoothed)
    z_range = np.max(z_vals) - np.min(z_vals)
    peak_prominence = x_range * base_prominence_ratio
    peak_distance = int(z_range * base_distance_ratio)

    peaks, _ = find_peaks(
        x_vals_right_smoothed, prominence=peak_prominence, distance=peak_distance
    )
    valleys, _ = find_peaks(
        -x_vals_right_smoothed, prominence=peak_prominence, distance=peak_distance
    )

    feature_points = [(p, "Peak") for p in peaks] + [(v, "Valley") for v in valleys]
    feature_points.append((0, "Start"))
    feature_points.append((len(z_vals) - 1, "End"))
    feature_points = sorted(set(feature_points), key=lambda x: x[0])

    fitted_angles = []
    print(f"\n📂 處理檔案: {os.path.basename(image_path)}")
    print(
        f"參數: prominence={peak_prominence:.2f}, distance={peak_distance}, margin_ratio={adaptive_margin_ratio}"
    )

    for i in range(len(feature_points) - 1):
        idx1, label1 = feature_points[i]
        idx2, label2 = feature_points[i + 1]
        segment_length = idx2 - idx1
        if segment_length < min_segment_points:
            continue

        fit_idx1 = idx1 + int(segment_length * adaptive_margin_ratio)
        fit_idx2 = idx2 - int(segment_length * adaptive_margin_ratio)
        if fit_idx2 <= fit_idx1:
            continue

        z_seg = z_vals[fit_idx1 : fit_idx2 + 1]
        # x_seg = x_vals_right_smoothed[fit_idx1 : fit_idx2 + 1]
        x_seg = x_vals_right_raw[fit_idx1 : fit_idx2 + 1]
        reg = LinearRegression().fit(z_seg.reshape(-1, 1), x_seg.reshape(-1, 1))
        r2 = reg.score(z_seg.reshape(-1, 1), x_seg.reshape(-1, 1))

        # === 適應性 r2 門檻 ===
        if segment_length < 60:
            min_r2 = r2_tiers[0]
        elif segment_length < 150:
            min_r2 = r2_tiers[1]
        else:
            min_r2 = r2_tiers[2]

        if r2 >= min_r2:
            slope = reg.coef_[0][0]
            angle = degrees(atan2(slope, 1))
            delta = abs(angle - angle_base)
            label = "Up" if slope > 0 else "Down"
            fitted_angles.append((fit_idx1, fit_idx2, angle, label, delta))

    # === 繪圖 ===
    plt.figure(figsize=(8, 10))
    plt.plot(x_vals_right_raw, z_vals, ":", color="gray", label="Raw Right Profile")
    plt.plot(
        x_vals_right_smoothed,
        z_vals,
        "--",
        color="black",
        label="Smoothed Right Contour",
        linewidth=1.5,
    )
    plt.plot(x_vals_left, z_vals, "g--", label="Left Profile", alpha=0.7)

    x_fit_base = reg_base.predict(z_vals.reshape(-1, 1))
    plt.plot(x_fit_base, z_vals, "g--", label=f"Base Fit {angle_base:.1f}°")

    plt.scatter(x_vals_right_smoothed[peaks], z_vals[peaks], color="red", label="Peaks")
    plt.scatter(
        x_vals_right_smoothed[valleys], z_vals[valleys], color="blue", label="Valleys"
    )

    for idx1, idx2, angle, label, delta in fitted_angles:
        z_endpoints = z_vals[[idx1, idx2]]
        model = LinearRegression().fit(
            z_vals[idx1 : idx2 + 1].reshape(-1, 1),
            x_vals_right_smoothed[idx1 : idx2 + 1].reshape(-1, 1),
            # x_vals_right_raw[idx1 : idx2 + 1].reshape(-1, 1),
        )
        x_pred = model.predict(z_endpoints.reshape(-1, 1))
        # x_pred = reg.predict(z_endpoints.reshape(-1, 1))
        color = "blue" if label == "Up" else "orange"
        plt.plot(
            x_pred,
            z_endpoints,
            linewidth=3,
            color=color,
            label=f"{label} {angle:.1f}°, Δ={delta:.1f}°",
        )

    plt.xlabel("X (μm)")
    plt.ylabel("Z (μm)")
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(os.path.basename(image_path))
    plt.show()


image_folder = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\RB"
for file in os.listdir(image_folder):
    if file.endswith(".png"):
        image_path = os.path.join(image_folder, file)
        analyze_feature_slopes_auto(image_path)
