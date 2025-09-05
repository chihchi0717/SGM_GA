import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math
from sklearn.linear_model import LinearRegression, RANSACRegressor
from skimage.measure import approximate_polygon


# --- 輔助函式 (無變動) ---
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cosine_angle = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(cosine_angle)


def nearest_idx(contour, pt):
    d = np.linalg.norm(contour - pt, axis=1)
    return int(np.argmin(d))


def slice_contour(contour, i0, i1):
    if i0 <= i1:
        return contour[i0 : i1 + 1]
    else:
        return np.vstack([contour[i0:], contour[: i1 + 1]])


def perpendicular_distance(point, line_start, line_end):
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    p, a, b = np.array(point), np.array(line_start), np.array(line_end)
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    t = np.clip(t, 0, 1)
    closest_point = a + t * (b - a)
    return np.linalg.norm(p - closest_point)


# <--- NEW VISUALIZATION MANAGER START --->
class VisualizationManager:
    """一個專門用來管理 Douglas-Peucker 演算法視覺化過程的類別"""

    def __init__(self, full_segment, tolerance, base_output_path):
        self.full_segment = full_segment
        self.tolerance = tolerance
        self.base_output_path = base_output_path
        self.step_counter = 0
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        # 預先計算圖片的邊界，讓每張圖的視野都一樣
        x_min, y_min = full_segment.min(axis=0)
        x_max, y_max = full_segment.max(axis=0)
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        self.xlims = (x_min - x_pad, x_max + x_pad)
        self.ylims = (y_min - y_pad, y_max + y_pad)

    def draw_and_save_step(self, current_points, furthest_point, max_dist, decision):
        """繪製並儲存演算法的單一步驟"""
        self.step_counter += 1
        self.ax.clear()

        # 1. 繪製背景：完整的輪廓片段
        self.ax.plot(
            self.full_segment[:, 0],
            self.full_segment[:, 1],
            color="lightgray",
            lw=2,
            label="Full Contour Segment",
        )

        # 2. 繪製當前正在處理的片段
        self.ax.plot(
            current_points[:, 0],
            current_points[:, 1],
            "k-",
            lw=1.5,
            label="Current Segment",
        )

        # 3. 繪製基準線
        start, end = current_points[0], current_points[-1]
        self.ax.plot(
            [start[0], end[0]], [start[1], end[1]], "--", color="gray", label="Baseline"
        )

        # 4. 標示最遠點和距離
        color = "red" if decision == "split" else "green"
        self.ax.scatter(
            furthest_point[0], furthest_point[1], color=color, s=100, zorder=5
        )
        self.ax.text(
            furthest_point[0] + 15,
            furthest_point[1],
            f"Dist: {max_dist:.1f}",
            color=color,
            fontsize=12,
        )

        # 5. 加上標題和圖例
        title = f"Step {self.step_counter}: Check segment from ({start[0]},{start[1]}) to ({end[0]},{end[1]})\n"
        title += f"Max Distance = {max_dist:.1f} | Tolerance = {self.tolerance:.1f} -> Decision: {decision.upper()}"
        self.ax.set_title(title, fontsize=12)

        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")
        self.ax.set_xlim(self.xlims)
        self.ax.set_ylim(self.ylims)
        self.ax.legend(loc="upper right")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.set_aspect("equal", adjustable="box")
        plt.gca().invert_yaxis()

        # 6. 儲存圖片
        output_path = f"{self.base_output_path}_Step_{self.step_counter:02d}.png"
        self.fig.savefig(output_path, dpi=100, bbox_inches="tight")
        print(
            f"  - Saved DP step {self.step_counter} to: {os.path.basename(output_path)}"
        )

    def close(self):
        """關閉繪圖視窗，釋放資源"""
        plt.close(self.fig)


def douglas_peucker_recursive_viz(points, tolerance, manager):
    """遞迴函式，現在透過 manager 來進行視覺化"""
    if len(points) < 2:
        return []

    start, end = points[0], points[-1]

    distances = [perpendicular_distance(pt, start, end) for pt in points]
    max_dist_idx = np.argmax(distances)
    max_dist = distances[max_dist_idx]

    if max_dist > tolerance:
        # 決策：分割
        manager.draw_and_save_step(points, points[max_dist_idx], max_dist, "split")

        results1 = douglas_peucker_recursive_viz(
            points[: max_dist_idx + 1], tolerance, manager
        )
        results2 = douglas_peucker_recursive_viz(
            points[max_dist_idx:], tolerance, manager
        )

        return results1 + results2[1:]
    else:
        # 決策：保留
        manager.draw_and_save_step(points, points[max_dist_idx], max_dist, "keep")
        return [start, end]


# <--- NEW VISUALIZATION MANAGER END --->


def fit_line_to_segment(points_xy, extension_length=0):
    if len(points_xy) < 5:
        return None
    try:
        x_range = np.ptp(points_xy[:, 0])
        y_range = np.ptp(points_xy[:, 1])
        is_vertical = y_range > x_range
        X = (
            points_xy[:, 1].reshape(-1, 1)
            if is_vertical
            else points_xy[:, 0].reshape(-1, 1)
        )
        y = points_xy[:, 0] if is_vertical else points_xy[:, 1]
        ransac = RANSACRegressor(
            LinearRegression(), min_samples=2, residual_threshold=5.0, max_trials=100
        )
        ransac.fit(X, y)
        if np.sum(ransac.inlier_mask_) < 2:
            return None
        clean_X = X[ransac.inlier_mask_]
        final_reg = LinearRegression().fit(clean_X, y[ransac.inlier_mask_])
        if is_vertical:
            y_min, y_max = np.min(clean_X), np.max(clean_X)
            preds = final_reg.predict(np.array([[y_min], [y_max]])).flatten()
            pt1_np = np.array([preds[0], y_min])
            pt2_np = np.array([preds[1], y_max])
        else:
            x_min, x_max = np.min(clean_X), np.max(clean_X)
            preds = final_reg.predict(np.array([[x_min], [x_max]])).flatten()
            pt1_np = np.array([x_min, preds[0]])
            pt2_np = np.array([x_max, preds[1]])
        if extension_length > 0:
            direction_vector = pt2_np - pt1_np
            vector_length = np.linalg.norm(direction_vector)
            if vector_length > 1e-6:
                unit_vector = direction_vector / vector_length
                return tuple(pt1_np - unit_vector * extension_length), tuple(
                    pt2_np + unit_vector * extension_length
                )
        return tuple(pt1_np), tuple(pt2_np)
    except ValueError:
        return None


def find_intersection(line1, line2):
    p1_orig, p2_orig = line1
    p3_orig, p4_orig = line2
    p1, p2 = (p1_orig, p2_orig) if p1_orig[1] < p2_orig[1] else (p2_orig, p1_orig)
    p3, p4 = (p3_orig, p4_orig) if p3_orig[1] < p4_orig[1] else (p4_orig, p3_orig)
    d1 = np.array(p2) - np.array(p1)
    d2 = np.array(p4) - np.array(p3)
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
    if abs(determinant) < 1e-6:
        return None
    ix = (b2 * c1 - b1 * c2) / determinant
    iy = (a1 * c2 - a2 * c1) / determinant
    raw_angle = calculate_angle(d1, d2)
    cross_product_z = d1[0] * d2[1] - d1[1] * d2[0]
    final_angle = 180.0 - raw_angle if cross_product_z > 0 else raw_angle
    return (ix, iy), final_angle


def find_apexes_with_convexity(contour, img_shape):
    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
    contour_diagonal = np.sqrt(w_c**2 + h_c**2)
    best_corners = None
    debug_steps = []
    tolerance_percents = [0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003]
    for tol_percent in tolerance_percents:
        tolerance = contour_diagonal * tol_percent
        corners = approximate_polygon(contour, tolerance=tolerance)
        step_info = {
            "tol_percent": tol_percent,
            "corners": corners,
            "is_selected": False,
        }
        debug_steps.append(step_info)
        if len(corners) >= 10:
            best_corners = corners
            step_info["is_selected"] = True
            if len(corners) > 11:
                break
    if best_corners is None:
        if debug_steps:
            best_step = max(debug_steps, key=lambda x: len(x["corners"]))
            best_corners = best_step["corners"]
            best_step["is_selected"] = True
        else:
            return [], debug_steps
    corners = best_corners.reshape(-1, 2)
    x_gate = img_shape[1] * 0.35
    corners_on_right = [pt for pt in corners if pt[0] > x_gate]
    if len(corners_on_right) < 3:
        return [], debug_steps
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
    return apexes, debug_steps


def fit_left_substrate(binary_image):
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
        preds = ransac.predict(np.array([[y1], [y2]])).flatten()
        angle = math.degrees(math.atan2(y2 - y1, preds[0] - preds[1]))
        return (preds[0], y1), (preds[1], y2), angle
    except Exception:
        return None


def analyze_prism_image(file_path, output_folder):
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

    if not contours:
        print("  Analysis failed: Could not find any contours.")
        return False

    main_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    # --- 主分析圖的設定 ---
    fig_main, ax_main = plt.subplots(figsize=(12, 18))
    ax_main.plot(
        main_contour[:, 0], main_contour[:, 1], color="white", lw=1.5, zorder=4
    )
    ax_main.fill(main_contour[:, 0], main_contour[:, 1], color="gray", zorder=1)
    ax_main.set_xlim(0, img.shape[1])
    ax_main.set_ylim(0, img.shape[0])
    ax_main.grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray")

    apexes_from_contour, debug_info = find_apexes_with_convexity(
        main_contour, img.shape
    )

    colors = plt.cm.viridis(np.linspace(0, 1, len(debug_info)))
    for i, step in enumerate(debug_info):
        corners = step["corners"].reshape(-1, 2)
        label = f"Tol: {step['tol_percent']*100:.1f}% ({len(corners)} corners)"
        if step["is_selected"]:
            label += " (Selected)"
        ax_main.scatter(
            corners[:, 0],
            corners[:, 1],
            s=(120 if step["is_selected"] else 50),
            color=colors[i],
            label=label,
            zorder=(10 if step["is_selected"] else 5),
            marker=("P" if step["is_selected"] else "o"),
            edgecolors=("black" if step["is_selected"] else "none"),
            alpha=0.9,
        )
    ax_main.legend(
        title="Corner Detection Steps", bbox_to_anchor=(1.02, 1), loc="upper left"
    )

    if len(apexes_from_contour) < 3:
        print(f"  Analysis failed: Found only {len(apexes_from_contour)} apexes.")
        output_path = os.path.join(
            output_folder, f"analyzed_DEBUG_{os.path.basename(file_path)}"
        )
        fig_main.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig_main)
        return False

    print("--- Analysis Summary ---")
    apexes_by_y = sorted(apexes_from_contour, key=lambda a: a["point"][1])

    # <--- MODIFICATION: 呼叫新的分步視覺化流程 --->
    if len(apexes_by_y) >= 4:
        # 選取中間一段有代表性的輪廓
        p_start_apex = apexes_by_y[len(apexes_by_y) // 2]
        p_end_apex = apexes_by_y[len(apexes_by_y) // 2 + 1]
        idx_start = nearest_idx(main_contour, p_start_apex["point"])
        idx_end = nearest_idx(main_contour, p_end_apex["point"])
        segment_to_viz = slice_contour(main_contour, idx_start, idx_end)

        selected_tolerance_value = 0
        for step in debug_info:
            if step["is_selected"]:
                x_min, y_min, w, h = cv2.boundingRect(main_contour)
                diag = np.sqrt(w**2 + h**2)
                selected_tolerance_value = diag * step["tol_percent"]
                break

        if selected_tolerance_value > 0 and len(segment_to_viz) > 1:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            viz_base_path = os.path.join(output_folder, f"{base_name}_DP")

            print("--- Starting Douglas-Peucker Step-by-Step Visualization ---")
            manager = VisualizationManager(
                segment_to_viz, selected_tolerance_value, viz_base_path
            )
            douglas_peucker_recursive_viz(
                segment_to_viz, selected_tolerance_value, manager
            )
            manager.close()
            print("--- Step-by-Step Visualization Finished ---")
    # <--- END OF MODIFICATION --->

    extended_lines = []
    for i in range(len(apexes_by_y) - 1):
        p1_apex, p2_apex = apexes_by_y[i], apexes_by_y[i + 1]
        idx1, idx2 = nearest_idx(main_contour, p1_apex["point"]), nearest_idx(
            main_contour, p2_apex["point"]
        )
        segment_points = slice_contour(main_contour, idx1, idx2)
        if len(segment_points) > len(main_contour) * 0.6:
            segment_points = slice_contour(main_contour, idx2, idx1)
        fitted_line = fit_line_to_segment(segment_points, extension_length=500)
        if fitted_line:
            extended_lines.append(fitted_line)
            pt1, pt2 = fitted_line
            ax_main.plot(
                [pt1[0], pt2[0]], [pt1[1], pt2[1]], "c--", lw=1.5, alpha=0.8, zorder=2
            )

    intersection_points = []
    for i in range(len(extended_lines) - 1):
        result = find_intersection(extended_lines[i], extended_lines[i + 1])
        if result:
            intersection, angle = result
            intersection_points.append(intersection)
            ax_main.plot(
                intersection[0], intersection[1], "r*", markersize=15, zorder=12
            )
            ax_main.text(
                intersection[0] + 15,
                intersection[1] - 15,
                f"{angle:.1f}°",
                color="cyan",
                fontsize=14,
                fontweight="bold",
                zorder=15,
            )

    for i in range(len(intersection_points) - 1):
        p1, p2 = np.array(intersection_points[i]), np.array(intersection_points[i + 1])
        length = np.linalg.norm(p2 - p1)
        mid_point = (p1 + p2) / 2
        ax_main.text(
            mid_point[0] + 10,
            mid_point[1],
            f"{length:.1f}",
            color="orange",
            fontsize=14,
            fontweight="bold",
            zorder=15,
        )
        ax_main.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="orange",
            linestyle="-",
            linewidth=2,
            zorder=10,
        )

    left_line_data = fit_left_substrate(binary)
    if left_line_data:
        p1, p2, angle = left_line_data
        ax_main.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], color="yellow", linewidth=3, zorder=3
        )
        ax_main.text(
            p1[0] + 20,
            p1[1] + (p2[1] - p1[1]) / 2,
            f"Left {abs(90-angle):.1f} deg",
            color="yellow",
            fontsize=14,
            fontweight="bold",
        )

    ax_main.set_title(os.path.basename(file_path).split(".")[0], fontsize=16)
    ax_main.set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    output_path = os.path.join(
        output_folder, f"analyzed_VISUALIZED_{os.path.basename(file_path)}"
    )
    fig_main.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_main)
    print(f"Result saved to: {output_path}")
    return True


if __name__ == "__main__":
    try:
        import skimage, sklearn
    except ImportError:
        print("Required packages 'scikit-image' or 'scikit-learn' are missing.")
        exit()
    input_directory = (
        r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202509\0904"
    )
    output_directory = os.path.join(input_directory, "results_analyzed_final")
    if not os.path.isdir(input_directory):
        print(f"Error: Directory not found: '{input_directory}'")
        exit()
    image_files = (
        glob.glob(os.path.join(input_directory, "*.png"))
        + glob.glob(os.path.join(input_directory, "*.jpg"))
        + glob.glob(os.path.join(input_directory, "*.jpeg"))
    )
    if not image_files:
        print(f"No image files found in '{input_directory}'.")
    else:
        print(f"Found {len(image_files)} images. Starting analysis...")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        success_count, fail_count = 0, 0
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
