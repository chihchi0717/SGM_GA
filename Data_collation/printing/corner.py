import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import degrees
from skimage.measure import approximate_polygon
from os.path import basename
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


def calculate_signed_area(contour):
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def vector_angle(v1, v2):
    """計算兩向量夾角 (0~180°)"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return np.nan
    angle = degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
    return angle


def analyze_contour_final(
    image_path, px_size=0.56, rdp_tolerance=15.0, output_dir=None
):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_dilate = cv2.dilate(binary, kernel, iterations=3)
    binary_close = cv2.morphologyEx(
        binary_dilate, cv2.MORPH_CLOSE, kernel, iterations=10
    )
    binary_clean = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("❌ Error: No contours found.")
        return

    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour.reshape(-1, 2).astype(np.float32)
    corner_points = approximate_polygon(contour_points, tolerance=rdp_tolerance)

    if not np.allclose(corner_points[0], corner_points[-1]):
        corner_points = np.vstack([corner_points, corner_points[0]])

    corner_points_scaled = corner_points * px_size

    # === 合併相同方向的連續段 ===
    merged_segments = []
    i = 0
    while i < len(corner_points_scaled) - 1:
        p_start = corner_points_scaled[i]
        p_curr = p_start
        j = i + 1

        dx = corner_points_scaled[j][0] - p_curr[0]
        dz = corner_points_scaled[j][1] - p_curr[1]
        angle = degrees(np.arctan2(-dz, dx))
        direction = "Up" if angle > 10 else "Down" if angle < -10 else "Flat"

        angle_sum = angle
        seg_count = 1

        while j < len(corner_points_scaled) - 1:
            p_next = corner_points_scaled[j + 1]
            dx_next = p_next[0] - p_curr[0]
            dz_next = p_next[1] - p_curr[1]
            angle_next = degrees(np.arctan2(-dz_next, dx_next))

            direction_next = (
                "Up" if angle_next > 10 else "Down" if angle_next < -10 else "Flat"
            )
            if direction_next != direction or abs(angle_next - angle) > 10:
                break

            angle_sum += angle_next
            seg_count += 1
            p_curr = p_next
            j += 1

        p_end = corner_points_scaled[i + seg_count]
        avg_angle = angle_sum / seg_count
        merged_segments.append((p_start, p_end, direction, avg_angle))
        i += seg_count

    # === 尋找基板段（方向為 Down 且角度接近垂直）===
    substrate_seg = None
    for seg in merged_segments:
        p1, p2, direction, ang = seg
        if direction == "Down" and abs(abs(ang) - 90) < 5:
            substrate_seg = seg
            break
    if substrate_seg is None:
        print("⚠️ Substrate segment not found")
        return

    sub_vec = np.array(substrate_seg[1]) - np.array(substrate_seg[0])

    # === 分群結構 ===
    y_values = [((p1[1] + p2[1]) / 2) for p1, p2, _, _ in merged_segments]
    y_min, y_max = min(y_values), max(y_values)
    estimated_prism_height = 100  # μm
    num_structures = max(1, int((y_max - y_min) / estimated_prism_height))
    structure_groups = [[] for _ in range(num_structures)]

    for seg in merged_segments:
        p1, p2, direction, avg_angle = seg
        mid_y = (p1[1] + p2[1]) / 2
        idx = int((mid_y - y_min) / (y_max - y_min) * num_structures)
        idx = max(0, min(num_structures - 1, idx))
        structure_groups[idx].append(seg)

    # === 尋找主斜邊 ===
    highlight_segments = []  # List of (label, segment, interior_angle)
    for group in structure_groups:
        upper_slopes, lower_slopes = [], []

        for seg in group:
            p1, p2, _, _ = seg
            vec = np.array(p2) - np.array(p1)
            angle_with_substrate = 180 - vector_angle(vec, sub_vec)

            slope = (p2[0] - p1[0]) / (p2[1] - p1[1] + 1e-6)

            if slope > 0:
                upper_slopes.append((seg, angle_with_substrate))
            elif slope < 0:
                lower_slopes.append((seg, angle_with_substrate))

        def seg_length(seg):
            return np.linalg.norm(np.array(seg[1]) - np.array(seg[0]))

        if upper_slopes:
            best_upper = max(upper_slopes, key=lambda x: seg_length(x[0]))
            highlight_segments.append(("Upper", best_upper[0], best_upper[1]))
        if lower_slopes:
            best_lower = max(lower_slopes, key=lambda x: seg_length(x[0]))
            highlight_segments.append(("Lower", best_lower[0], best_lower[1]))

    # === 繪圖 ===
    plt.figure(figsize=(10, 10))
    plt.plot(
        contour_points[:, 0] * px_size,
        contour_points[:, 1] * px_size,
        "g-",
        alpha=0.4,
        label="Full Contour",
    )

    for p1, p2, direction, angle_deg in merged_segments:
        color = (
            "blue" if direction == "Up" else "orange" if direction == "Down" else "gray"
        )
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1)

    for label, seg, angle in highlight_segments:
        p1, p2, _, _ = seg
        color = "red" if label == "Upper" else "purple"
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.5)
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        plt.text(
            mid_x,
            mid_y,
            f"{label} {angle:.1f}°",
            fontsize=9,
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor=color),
        )

    plt.title("Contour Simplification and Interior Angle w.r.t Substrate")
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.legend(["Full Contour", "Reconstructed Lines"])
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.axis("equal")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{base}_result.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import degrees
from skimage.measure import approximate_polygon
from os.path import basename
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


def vector_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return np.nan
    angle = degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
    return angle


def analyze_contour_final_v2(
    image_path, px_size=0.56, rdp_tolerance=15.0, output_dir=None
):
    def compute_angle(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return degrees(np.arctan2(-dy, dx))  # upward = positive angle

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.dilate(binary, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=10)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Error: No contours found.")
        return

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    corner_points = approximate_polygon(contour, tolerance=rdp_tolerance)
    if not np.allclose(corner_points[0], corner_points[-1]):
        corner_points = np.vstack([corner_points, corner_points[0]])
    corner_points = corner_points * px_size

    # === 合併角度相近方向 ===
    merged_segments = []
    i = 0
    while i < len(corner_points) - 2:
        p_start = corner_points[i]
        p_curr = corner_points[i + 1]
        angle_ref = compute_angle(p_start, p_curr)
        length_accum = np.linalg.norm(p_curr - p_start)
        j = i + 1
        while j < len(corner_points) - 1:
            p_next = corner_points[j + 1]
            angle_next = compute_angle(p_curr, p_next)
            angle_diff = abs(angle_next - angle_ref)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff < 10:
                length_accum += np.linalg.norm(p_next - p_curr)
                p_curr = p_next
                j += 1
            else:
                break
        p_end = corner_points[j]
        avg_angle = compute_angle(p_start, p_end)
        direction = "Up" if avg_angle > 10 else "Down" if avg_angle < -10 else "Flat"
        if length_accum > 50:
            merged_segments.append((p_start, p_end, direction, avg_angle))
        i = j

    # === 找 Substrate ===
    substrate_seg = None
    for seg in merged_segments:
        _, _, direction, ang = seg
        if direction == "Down" and abs(abs(ang) - 90) < 5:
            substrate_seg = seg
            break
    if substrate_seg is None:
        print("⚠️ Substrate segment not found")
        return
    sub_vec = np.array(substrate_seg[1]) - np.array(substrate_seg[0])

    # === 分群構造 ===
    y_values = [((p1[1] + p2[1]) / 2) for p1, p2, _, _ in merged_segments]
    y_min, y_max = min(y_values), max(y_values)
    estimated_prism_height = 100
    num_structures = max(1, int((y_max - y_min) / estimated_prism_height))
    structure_groups = [[] for _ in range(num_structures)]
    for seg in merged_segments:
        p1, p2, direction, avg_angle = seg
        mid_y = (p1[1] + p2[1]) / 2
        idx = int((mid_y - y_min) / (y_max - y_min) * num_structures)
        idx = max(0, min(num_structures - 1, idx))
        structure_groups[idx].append(seg)

    # === 取得主斜邊 ===
    highlight_segments = []
    for group in structure_groups:
        upper_slopes, lower_slopes = [], []
        for seg in group:
            p1, p2, _, _ = seg
            vec = np.array(p2) - np.array(p1)
            angle_with_sub = 180 - vector_angle(vec, sub_vec)
            slope = (p2[0] - p1[0]) / (p2[1] - p1[1] + 1e-6)
            if slope > 0:
                upper_slopes.append((seg, angle_with_sub))
            elif slope < 0:
                lower_slopes.append((seg, angle_with_sub))

        def seg_len(s):
            return np.linalg.norm(np.array(s[0][1]) - np.array(s[0][0]))

        if upper_slopes:
            best = max(upper_slopes, key=lambda s: seg_len(s))
            highlight_segments.append(("Upper", best[0], best[1]))
        if lower_slopes:
            best = max(lower_slopes, key=lambda s: seg_len(s))
            highlight_segments.append(("Lower", best[0], best[1]))

    # === Plot ===
    plt.figure(figsize=(10, 10))
    plt.plot(
        contour[:, 0] * px_size,
        contour[:, 1] * px_size,
        "g-",
        alpha=0.4,
        label="Full Contour",
    )
    for label, seg, angle in highlight_segments:
        p1, p2, _, _ = seg
        color = "red" if label == "Upper" else "purple"
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.5)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        plt.text(
            mx,
            my,
            f"{label} {angle:.1f}°",
            fontsize=9,
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor=color),
        )
    plt.title("Contour Simplification and Interior Angle w.r.t Substrate")
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{base}_result.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()


# --- 執行 ---

input_folder = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\RB"
output_folder = (
    r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\RB\ResultFigures"
)


def batch_process_folder(folder_path, output_dir, px_size=0.56, rdp_tolerance=15.0):
    supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_ext)
    ]

    if not image_files:
        print("⚠️ 沒有找到圖片檔案")
        return

    for idx, image_path in enumerate(sorted(image_files)):
        print(
            f"\n=== [{idx+1}/{len(image_files)}] Processing: {os.path.basename(image_path)} ==="
        )
        try:
            analyze_contour_final_v2(
                image_path,
                px_size=px_size,
                rdp_tolerance=rdp_tolerance,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")


batch_process_folder(input_folder, output_folder, rdp_tolerance=1, px_size=1.12)
