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

    # 膨脹幫助上方連接
    binary_dilate = cv2.dilate(binary, kernel, iterations=3)
    binary_close = cv2.morphologyEx(binary_dilate, cv2.MORPH_CLOSE, kernel, iterations=5)
    binary_clean = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel, iterations=2)

    # binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=4)
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

    # --- 畫圖 ---
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
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=4)

        vec = np.array(p2) - np.array(p1)
        ang_between = vector_angle(vec, sub_vec)
        interior_angle = 180 - ang_between

        if (p1 == substrate_seg[0]).all() and (p2 == substrate_seg[1]).all():
            label_text = f"Substrate {abs(angle_deg):.1f}°"
        else:
            label_text = f"Interior {interior_angle:.1f}°"

        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        plt.text(
            mid_x,
            mid_y,
            label_text,
            fontsize=8,
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor=color),
        )

    plt.title("Contour Simplification and Interior Angle w.r.t Substrate")
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.legend(["Full Contour", "Reconstructed Lines"])
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.axis("equal")

    # --- 儲存圖像 ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{base}_result.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()


def analyze_contour_final_v2(
    image_path, px_size=0.56, rdp_tolerance=15.0, output_dir=None
):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from math import degrees
    from skimage.measure import approximate_polygon
    import os

    def vector_angle(v1, v2):
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return np.nan
        angle = degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
        return angle

    # === 1. 讀圖與增強對比 ===
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    # === 2. 二值化與形態學 ===
    binary = cv2.adaptiveThreshold(
        img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # === 3. 輪廓偵測與篩選 ===
    contours, _ = cv2.findContours(
        binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not contours:
        print("❌ No valid contours.")
        return

    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour.reshape(-1, 2).astype(np.float32)
    corner_points = approximate_polygon(contour_points, tolerance=rdp_tolerance)

    if not np.allclose(corner_points[0], corner_points[-1]):
        corner_points = np.vstack([corner_points, corner_points[0]])

    corner_points_scaled = corner_points * px_size

    # === 4. 線段合併與分類 ===
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
        length = np.linalg.norm(p_end - p_start)

        if length >= 20:  # 過短段略過
            merged_segments.append((p_start, p_end, direction, avg_angle))

        i += seg_count

    # === 5. 找基板線段 ===
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

    # === 6. 畫圖 ===
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
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=4)

        vec = np.array(p2) - np.array(p1)
        ang_between = vector_angle(vec, sub_vec)
        interior_angle = 180 - ang_between

        if (p1 == substrate_seg[0]).all() and (p2 == substrate_seg[1]).all():
            label_text = f"Substrate {abs(angle_deg):.1f}°"
        else:
            label_text = f"Interior {interior_angle:.1f}°"

        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        plt.text(
            mid_x,
            mid_y,
            label_text,
            fontsize=8,
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor=color),
        )

    plt.title("Contour Simplification and Interior Angle w.r.t Substrate")
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.legend(["Full Contour", "Reconstructed Lines"])
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.axis("equal")

    # === 7. 儲存圖檔 ===
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


batch_process_folder(input_folder, output_folder, rdp_tolerance=2)
