import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math
from sklearn.linear_model import LinearRegression, RANSACRegressor
from skimage.measure import approximate_polygon


def calculate_angle(v1, v2):
    """計算兩個向量之間的角度（度）。"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cosine_angle = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(cosine_angle)


def nearest_idx(contour, pt):
    """在輪廓上尋找與給定點最近的點的索引。"""
    d = np.linalg.norm(contour - pt, axis=1)
    return int(np.argmin(d))


def slice_contour(contour, i0, i1):
    """根據索引切割輪廓，處理環繞情況。"""
    if i0 <= i1:
        return contour[i0 : i1 + 1]
    else:  # 處理閉合輪廓的環繞情況
        return np.vstack([contour[i0:], contour[: i1 + 1]])


def fit_line_to_segment(points_xy, extension_length=0):
    """
    穩健地擬合一條線到點集，並可選擇性地向兩端延伸。
    """
    if len(points_xy) < 5:
        return None

    try:
        x_range = np.ptp(points_xy[:, 0])
        y_range = np.ptp(points_xy[:, 1])
        is_vertical = y_range > x_range

        if is_vertical:
            X = points_xy[:, 1].reshape(-1, 1)
            y = points_xy[:, 0]
        else:
            X = points_xy[:, 0].reshape(-1, 1)
            y = points_xy[:, 1]

        ransac = RANSACRegressor(
            LinearRegression(), min_samples=2, residual_threshold=5.0, max_trials=100
        )
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_

        if np.sum(inlier_mask) < 2:
            return None

        clean_X = X[inlier_mask]
        final_reg = LinearRegression()
        final_reg.fit(clean_X, y[inlier_mask])

        pt1_np, pt2_np = None, None
        if is_vertical:
            y_min, y_max = np.min(clean_X), np.max(clean_X)
            x_min_pred, x_max_pred = final_reg.predict(np.array([[y_min], [y_max]]))
            pt1_np = np.array([x_min_pred, y_min])
            pt2_np = np.array([x_max_pred, y_max])
        else:
            x_min, x_max = np.min(clean_X), np.max(clean_X)
            y_min_pred, y_max_pred = final_reg.predict(np.array([[x_min], [x_max]]))
            pt1_np = np.array([x_min, y_min_pred])
            pt2_np = np.array([x_max, y_max_pred])

        if extension_length > 0 and pt1_np is not None and pt2_np is not None:
            direction_vector = pt2_np - pt1_np
            vector_length = np.linalg.norm(direction_vector)
            if vector_length > 1e-6:
                unit_vector = direction_vector / vector_length
                pt1_extended = pt1_np - unit_vector * extension_length
                pt2_extended = pt2_np + unit_vector * extension_length
                return tuple(pt1_extended), tuple(pt2_extended)

        if pt1_np is not None and pt2_np is not None:
            return tuple(pt1_np), tuple(pt2_np)
        return None
    except ValueError:
        return None


def find_intersection(line1, line2):
    """計算兩條線段定義的直線的交點。"""
    p1, p2 = line1
    p3, p4 = line2
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None
    else:
        ix = (b2 * c1 - b1 * c2) / determinant
        iy = (a1 * c2 - a2 * c1) / determinant
        return (ix, iy)


def find_apexes_with_convexity(contour, img_shape):
    """透過分析凸性來穩健地尋找和分類頂點（峰和谷）。"""
    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
    contour_diagonal = np.sqrt(w_c**2 + h_c**2)

    best_corners = None
    for tol_percent in [0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003]:
        tolerance = contour_diagonal * tol_percent
        corners = approximate_polygon(contour, tolerance=tolerance)
        if len(corners) >= 4:
            best_corners = corners
            if len(corners) > 8:
                break

    if best_corners is None:
        return []

    corners = best_corners.reshape(-1, 2)
    x_gate = img_shape[1] * 0.35
    corners_on_right = [pt for pt in corners if pt[0] > x_gate]

    if len(corners_on_right) < 3:
        return []

    apexes = []
    original_indices_map = {tuple(pt): i for i, pt in enumerate(corners)}
    for i in range(len(corners_on_right)):
        p_curr = corners_on_right[i]
        original_idx = original_indices_map.get(tuple(p_curr))
        if original_idx is None:
            continue

        p_prev = corners[(original_idx - 1 + len(corners)) % len(corners)]
        p_next = corners[(original_idx + 1 + len(corners)) % len(corners)]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        if abs(cross_product) < 1e-6:
            continue

        apex_type = "valley" if cross_product > 0 else "peak"
        apexes.append({"point": p_curr, "type": apex_type, "vec_in": v1, "vec_out": v2})

    return apexes


def fit_left_substrate(binary_image):
    """擬合左側基板的線條。"""
    h, w = binary_image.shape
    y_coords, x_coords = [], []
    for y in range(0, h, 5):
        row = binary_image[y, : int(w * 0.3)]
        white_pixels = np.where(row > 0)[0]
        if len(white_pixels) > 0:
            x_coords.append(white_pixels[0])
            y_coords.append(y)
    if len(x_coords) < 20:
        return None
    points = np.array([x_coords, y_coords]).T
    try:
        ransac = RANSACRegressor(
            LinearRegression(), min_samples=10, residual_threshold=5.0, max_trials=100
        )
        ransac.fit(points[:, 1].reshape(-1, 1), points[:, 0])
        inliers = points[ransac.inlier_mask_]
        if len(inliers) < 20:
            return None
        y1, y2 = float(inliers[:, 1].min()), float(inliers[:, 1].max())
        x1_pred, x2_pred = ransac.predict(np.array([[y1], [y2]]))
        angle = math.degrees(math.atan2(y2 - y1, float(x1_pred) - float(x2_pred)))
        return (float(x1_pred), y1), (float(x2_pred), y2), angle
    except Exception:
        return None


def analyze_prism_image(file_path, output_folder):
    """主分析函數，尋找斜邊交點並標註長度。"""
    print(f"Processing file: {file_path}")

    try:
        with open(file_path, "rb") as f:
            nparr = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        print(f"  Error: Could not read or decode file. Reason: {e}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig, ax = plt.subplots(figsize=(10, 15))

    if not contours:
        print("  Analysis failed: Could not find any contours.")
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.close(fig)
        return False

    main_contour = max(contours, key=cv2.contourArea)
    main_contour_reshaped = main_contour.reshape(-1, 2)

    ax.fill(
        main_contour_reshaped[:, 0], main_contour_reshaped[:, 1], color="gray", zorder=1
    )
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray")

    apexes_from_contour = find_apexes_with_convexity(main_contour_reshaped, img.shape)

    if len(apexes_from_contour) < 3:
        print(f"  Analysis failed: Found only {len(apexes_from_contour)} apexes.")
        plt.close(fig)
        return False

    print("--- Analysis Summary ---")

    apexes_by_y = sorted(apexes_from_contour, key=lambda a: a["point"][1])

    # 擬合斜邊、儲存並繪製延伸線
    extended_lines = []
    for i in range(len(apexes_by_y) - 1):
        p1_apex = apexes_by_y[i]
        p2_apex = apexes_by_y[i + 1]

        idx1 = nearest_idx(main_contour_reshaped, p1_apex["point"])
        idx2 = nearest_idx(main_contour_reshaped, p2_apex["point"])

        segment_points = slice_contour(main_contour_reshaped, idx1, idx2)

        if len(segment_points) > len(main_contour_reshaped) * 0.6:
            segment_points = slice_contour(main_contour_reshaped, idx2, idx1)

        # 增加延伸長度以確保線條相交
        fitted_line = fit_line_to_segment(segment_points, extension_length=500)

        if fitted_line:
            extended_lines.append(fitted_line)
            pt1, pt2 = fitted_line
            ax.plot(
                [pt1[0], pt2[0]], [pt1[1], pt2[1]], "c--", lw=1.5, alpha=0.8, zorder=2
            )

    # 計算並繪製相鄰延伸線的交點
    intersection_points = []
    for i in range(len(extended_lines) - 1):
        line1 = extended_lines[i]
        line2 = extended_lines[i + 1]
        intersection = find_intersection(line1, line2)
        if intersection:
            intersection_points.append(intersection)
            ax.plot(intersection[0], intersection[1], "r*", markersize=15, zorder=12)
            print(
                f"  Intersection found at: ({intersection[0]:.1f}, {intersection[1]:.1f})"
            )

    # *** 新增：計算並標註交點之間的長度 ***
    for i in range(len(intersection_points) - 1):
        p1 = np.array(intersection_points[i])
        p2 = np.array(intersection_points[i + 1])

        # 計算長度
        length = np.linalg.norm(p2 - p1)

        # 計算中點以放置文字
        mid_point = (p1 + p2) / 2

        # 繪製長度標註
        ax.text(
            mid_point[0] + 10,
            mid_point[1],
            f"{length:.1f}",
            color="orange",
            fontsize=14,
            fontweight="bold",
            zorder=15,
        )

        # 繪製連接線以突顯
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="orange",
            linestyle="-",
            linewidth=2,
            zorder=10,
        )
        print(f"  Length between intersection {i+1} and {i+2}: {length:.1f}")

    # 繪製左側基板線
    left_line_data = fit_left_substrate(binary)
    if left_line_data:
        p1, p2, angle = left_line_data
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="yellow", linewidth=3, zorder=3)
        ax.text(
            p1[0] + 20,
            p1[1] + (p2[1] - p1[1]) / 2,
            f"Left {abs(90-angle):.1f} deg",
            color="yellow",
            fontsize=14,
            fontweight="bold",
        )

    print("-" * 24 + "\n")
    ax.set_title(os.path.basename(file_path).split(".")[0], fontsize=16)
    ax.set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    output_path = os.path.join(output_folder, f"analyzed_{os.path.basename(file_path)}")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Result saved to: {output_path}")
    return True


# --- 主執行區塊 ---
if __name__ == "__main__":
    try:
        import skimage, sklearn
    except ImportError:
        print("Required packages 'scikit-image' or 'scikit-learn' are missing.")
        print("Please run 'pip install scikit-image scikit-learn' to install them.")
        exit()

    # --- 重要：請在此設定您的圖片資料夾路徑 ---
    input_directory = (
        r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\DOE_RB\0.6_0.9"
    )
    output_directory = os.path.join(input_directory, "results_analyzed_final")

    if not os.path.isdir(input_directory):
        print(f"Error: Directory not found: '{input_directory}'")
        exit()

    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(glob.glob(os.path.join(input_directory, ext)))

    if not image_files:
        print(f"No image files found in '{input_directory}'.")
    else:
        print(f"Found {len(image_files)} images. Starting analysis...")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created output directory: {output_directory}")

        success_count = 0
        fail_count = 0
        for file in image_files:
            try:
                if analyze_prism_image(file, output_directory):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing {os.path.basename(file)}: {e}"
                )
                fail_count += 1

        print("\n--- BATCH PROCESSING COMPLETE ---")
        print(f"  Successfully processed: {success_count} images")
        print(f"  Failed to process:    {fail_count} images")
        print("---------------------------------")
