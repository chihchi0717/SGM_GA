import os
import numpy as np
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*32-bit application should be automated.*")


# === Gaussian 權重函數 ===
def gaussian_weight(
    theta, center, sigma, theta_min=None, theta_max=None, peak=1.0, base=0.0
):
    if theta_min is not None and (theta < theta_min or theta > theta_max):
        return 0.0
    return base + peak * np.exp(-((theta - center) ** 2) / (2 * sigma**2))


def down_gaussian_weight(
    theta, sigma, theta_min=None, theta_max=None, peak=1.0, base=0.0
):
    return -np.exp(-((theta) ** 2) / (2 * sigma**2))


# === Polar txt 讀取 ===
def read_txt_file(file_path):
    try:
        angles = []
        intensities = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        angle = float(parts[0])
                        intensity = float(parts[1])  # 只取第二欄
                        angles.append(angle)
                        intensities.append(intensity)
                    except ValueError:
                        continue  # 跳過非數值行
        return np.array(angles), np.array(intensities)
    except Exception as e:
        print(f"無法處理 {file_path}: {e}")
        return np.array([]), np.array([])


# === 製程回歸模型 ===
def compute_regression_score(S1, S2, A1):
    return (
        -0.067
        + 0.217 * S1
        + 0.275 * S2
        + 0.002 * A1
        - 0.506 * S1 * S2
        - 0.002 * S1 * A1
        - 0.002 * S2 * A1
        + 0.004 * S1 * S2 * A1
    )


# === 加權導光效率計算 ===
def evaluate_fitness(
    folder,
    individual,
    theta_u2=100,
    sigma_up=60,
    sigma_down=15,
    theta_d=22,
    return_uniformity=False,
):
    S1, S2, A1 = individual
    weighted_efficiency_total = 0
    weight_sum = 0
    efficiencies_per_angle = []

    weights = [1, 2, 5, 7, 5, 8.5, 1.5, 2]
    uniformities_per_angle = []

    for idx, angle in enumerate(range(10, 90, 10)):
        txt_path = os.path.join(folder, f"polar-{angle}.txt")
        try:
            angles, intensities = read_txt_file(txt_path)
            if len(angles) == 0:
                continue

            total_energy = np.sum(intensities)
            weighted_energy = 0
            weight_debug_sum = 0  # 為了印出此角度的權重總和

            intensities_up = []
            for theta, flux in zip(angles, intensities):
                # 預設權重為 0，避免意外錯誤導致 w 未定義
                if theta > 90:
                    # 向上光線，θ > 90°
                    # w = gaussian_weight(
                    #     theta,
                    #     center=theta_u2,
                    #     sigma=sigma_up,
                    #     theta_min=90,
                    #     theta_max=180,
                    #     peak=0.5,
                    #     base=0.5,
                    # )
                    w = 1
                else:
                    w = 0

                weighted_energy += flux * w
                weight_debug_sum += w
                if theta > 90:
                    intensities_up.append(flux)
                # print(f"    θ = {theta:6.1f}°, flux = {flux:.4e}, weight = {w:.4f}")

            eff = float(weighted_energy / total_energy) if total_energy > 0 else 0.0
            efficiencies_per_angle.append(eff)
            if intensities_up:
                mean_up = np.mean(intensities_up)
                std_up = np.std(intensities_up)
                cv_up_angle = std_up / mean_up if mean_up > 0 else 1.0
            else:
                cv_up_angle = 1.0
            uniformity_angle = 1.0 - cv_up_angle
            if uniformity_angle < 0:
                uniformity_angle = 0.0
            uniformities_per_angle.append(uniformity_angle)
            weighted_efficiency_total += eff * weights[idx]
            weight_sum += weights[idx]

            # print(
            #     f"Angle {angle}° - Eff: {eff:.4f}, "
            #     f"Total Energy: {total_energy:.4e}, "
            #     f"Weighted Energy: {weighted_energy:.4e}, "
            #     f"Weight Sum: {weight_debug_sum:.2f}"
            # )

        except Exception as e:
            print(f"無法處理 {txt_path}: {e}")
            efficiencies_per_angle.append(0.0)
            uniformities_per_angle.append(0.0)
            continue

    efficiency = weighted_efficiency_total / weight_sum if weight_sum > 0 else 0

    try:
        process_score = compute_regression_score(S1, S2, A1)
    except Exception as e:
        print(f"⚠️ 製程品質評估失敗: {e}")
        process_score = 1.0

    # fitness = efficiency * (1 / (1 + process_score))
    # return fitness, efficiency, process_score, efficiencies_per_angle
    if uniformities_per_angle:
        uniformity = float(np.mean(uniformities_per_angle))
    else:
        uniformity = 0.0

    alpha = 2.0  # 加權係數，可調整均勻度的懲罰強度
    # 將均勻度定義為 1 - CV
    fitness = (efficiency / (1 + process_score)) * uniformity

    print(f"Uniformity: {uniformity:.4f}")

    if return_uniformity:
        return (
            fitness,
            efficiency,
            process_score,
            uniformity,
            efficiencies_per_angle,
            uniformities_per_angle,
        )
    else:
        return fitness, efficiency, process_score, efficiencies_per_angle


# fitness, efficiency, process_score, efficiencies_per_angle = evaluate_fitness(
#     "C:\\Users\\User\\SGM_GA\\GA_population\\P1",
#     [0.4, 0.4, 45],
#     theta_u2=100,
#     sigma_up=20,
#     sigma_down=15,
#     theta_d=90,
# )
# print(f"Fitness: {fitness:.4f}")
# print(f"Efficiency: {efficiency:.4f}")
# print(f"Process Score: {process_score:.4f}")
# efficiencies_per_angle = [float(x) for x in efficiencies_per_angle]
# print(f"Per-angle Efficiency: {efficiencies_per_angle}")
