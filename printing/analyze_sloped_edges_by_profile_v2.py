# 放入檔案：analyze_sloped_edges_by_profile_v2.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from math import atan2, degrees
from sklearn.linear_model import LinearRegression


def analyze_sloped_edges_by_profile_v2(
    image_path,
    px_size=0.56,
    z_scale=0.56,
    rdp_tolerance=15.0,
    smooth_window=51,
    peak_prominence=20,
    peak_distance=100,
    margin_ratio_x=0.1,
    margin_ratio_y=0.05,
    fit_margin=10,
    min_r2=0.90,
    display=True,
):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_dilate = cv2.dilate(binary, kernel, iterations=3)
    binary_close = cv2.morphologyEx(
        binary_dilate, cv2.MORPH_CLOSE, kernel, iterations=5
    )
    binary_clean = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(
        binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        print("❌ No contour found")
        return

    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour.reshape(-1, 2)
    x_vals, y_vals = contour_points[:, 0], contour_points[:, 1]
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)

    x_margin = (x_max - x_min) * margin_ratio_x
    y_margin = (y_max - y_min) * margin_ratio_y

    filtered_points = contour_points[
        (x_vals > x_min + x_margin)
        & (y_vals > y_min + y_margin)
        & (y_vals < y_max - y_margin)
    ]

    if len(filtered_points) < smooth_window:
        print("⚠️ 濾除後輪廓點太少")
        return

    y_unique = np.unique(filtered_points[:, 1])
    upper_profile, lower_profile = [], []
    for y in y_unique:
        row_x = filtered_points[filtered_points[:, 1] == y][:, 0]
        if len(row_x) < 2:
            continue
        x_left = np.min(row_x)
        x_right = np.max(row_x)
        upper_profile.append([x_left, y])
        lower_profile.append([x_right, y])

    upper_profile = np.array(upper_profile)
    lower_profile = np.array(lower_profile)

    def adjust_window_length(profile_length, desired_window):
        max_valid = profile_length if profile_length % 2 == 1 else profile_length - 1
        return max(5, min(desired_window, max_valid))

    smooth_window_adj = adjust_window_length(len(upper_profile), smooth_window)

    upper_x = np.array([p[0] for p in upper_profile])
    lower_x = np.array([p[0] for p in lower_profile])

    # 動態調整平滑視窗長度
    upper_len = len(upper_x)
    lower_len = len(lower_x)

    upper_window = min(smooth_window, upper_len)
    if upper_window % 2 == 0:
        upper_window -= 1
    if upper_window < 3:
        upper_window = 3

    lower_window = min(smooth_window, lower_len)
    if lower_window % 2 == 0:
        lower_window -= 1
    if lower_window < 3:
        lower_window = 3

    if upper_window > len(upper_x):
        print(f"⚠️ Upper profile 長度不足，無法平滑處理：{image_path}")
        return
    if lower_window > len(lower_x):
        print(f"⚠️ Lower profile 長度不足，無法平滑處理：{image_path}")
        return

    upper_x_smooth = savgol_filter(upper_x, window_length=upper_window, polyorder=2)
    lower_x_smooth = savgol_filter(lower_x, window_length=lower_window, polyorder=2)



    profile_y = upper_profile[:, 1]

    x_inv = -lower_x_smooth
    peaks, _ = find_peaks(
        lower_x_smooth, prominence=peak_prominence, distance=peak_distance
    )
    valleys, _ = find_peaks(x_inv, prominence=peak_prominence, distance=peak_distance)

    segments_info = []
    for v in valleys:
        left_peaks = peaks[peaks < v]
        right_peaks = peaks[peaks > v]
        if len(left_peaks) == 0 or len(right_peaks) == 0:
            continue
        left = left_peaks[-1]
        right = right_peaks[0]

        def fit_line(y_segment, x_segment):
            if len(y_segment) < 2:
                return None, None, None
            model = LinearRegression().fit(y_segment.reshape(-1, 1), x_segment)
            r2 = model.score(y_segment.reshape(-1, 1), x_segment)
            slope = model.coef_[0]
            angle = degrees(atan2(slope * z_scale, 1))
            return slope, angle, r2

        upper_y = profile_y[left + fit_margin : v - fit_margin]
        upper_x = upper_x_smooth[left + fit_margin : v - fit_margin]
        lower_y = profile_y[v + fit_margin : right - fit_margin]
        lower_x = lower_x_smooth[v + fit_margin : right - fit_margin]

        upper_slope, upper_angle, r2_upper = fit_line(upper_y, upper_x)
        lower_slope, lower_angle, r2_lower = fit_line(lower_y, lower_x)

        segments_info.append(
            {
                "valley_index": v,
                "upper_angle": upper_angle,
                "lower_angle": lower_angle,
                "r2_upper": r2_upper,
                "r2_lower": r2_lower,
                "upper_pts": (upper_x, upper_y),
                "lower_pts": (lower_x, lower_y),
            }
        )

    fig = None
    if display:
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap="gray")
        plt.plot(upper_x_smooth, profile_y, "r-", label="Upper Profile")
        plt.plot(lower_x_smooth, profile_y, "b-", label="Lower Profile")
        plt.plot(lower_x_smooth[peaks], profile_y[peaks], "r^", label="Peaks")
        plt.plot(lower_x_smooth[valleys], profile_y[valleys], "bv", label="Valleys")

        for seg in segments_info:
            ux, uy = seg["upper_pts"]
            lx, ly = seg["lower_pts"]
            plt.plot(
                ux,
                uy,
                "red",
                linewidth=3,
                label="Upper Edge" if seg == segments_info[0] else "",
            )
            plt.plot(
                lx,
                ly,
                "purple",
                linewidth=3,
                label="Lower Edge" if seg == segments_info[0] else "",
            )
            if seg["upper_angle"] is not None:
                plt.text(
                    ux[len(ux) // 2],
                    uy[len(ux) // 2] - 10,
                    f"{seg['upper_angle']:.1f}°",
                    color="red",
                    fontsize=8,
                    ha="center",
                )
            if seg["lower_angle"] is not None:
                plt.text(
                    lx[len(lx) // 2],
                    ly[len(lx) // 2] + 10,
                    f"{seg['lower_angle']:.1f}°",
                    color="purple",
                    fontsize=8,
                    ha="center",
                )

        plt.legend()
        plt.title("Upper and Lower Sloped Edges by Cleaned Contour")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.tight_layout()

    return segments_info if not display else (segments_info, fig)
