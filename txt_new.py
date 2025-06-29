import os
import numpy as np


def read_txt_file(file_path):
    try:
        data = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        angle = float(parts[0])
                        intensity = float(parts[1])
                        data.append(intensity)
                    except ValueError:
                        continue  # 跳過標題或非數值行
        return np.array(data)
    except Exception as e:
        print(f"無法處理 {file_path}: {e}")
        return np.array([])


def average_intensity(intensities):
    return np.mean(intensities)


def uniformity(intensities):
    return np.min(intensities) / np.mean(intensities)


def standard_deviation(intensities):
    return np.std(intensities)


def score_data(i):
    folder_path = f"GA_population/P{i+1}"
    all_intensities = []

    for angle in range(10, 81, 10):
        txt_file = os.path.join(folder_path, f"polar-{angle}.txt")
        if not os.path.exists(txt_file):
            continue
        try:
            _, intensity = read_txt_file(txt_file)
            all_intensities.append(intensity)
        except Exception as e:
            print(f"無法處理 {txt_file}: {e}")

    if not all_intensities:
        return 0, 0, 0, 0

    all_data = np.concatenate(all_intensities)
    avg = average_intensity(all_data)
    uni = uniformity(all_data)
    std = standard_deviation(all_data)

    # 自定義加權評分（可根據實驗調整）
    score = avg * uni / (1 + std)

    return score, avg, uni, std


def compute_regression_score(S1, S2, A1):
    return (
        10.1180
        + 0.217 * S1
        + 0.275 * S2
        + 0.002 * A1
        - 0.506 * S1 * S2
        - 0.002 * S1 * A1
        - 0.002 * S2 * A1
        + 0.004 * S1 * S2 * A1
    )


def evaluate_fitness(
    folder,
    individual,
    return_uniformity=False,
    eff_weight=0.7,
    process_weight=0.3,
    uni_weight=0.1,
):
    """Evaluate optical performance for *individual* and return multiple objectives.

    When ``return_uniformity`` is ``True``, the upward energy distribution for
    each measurement angle is analyzed. The standard deviation of the upward
    energy (polar angle > 90°) is computed per angle and returned as
    ``uni_10`` .. ``uni_80``.  The function returns efficiency, process score and
    uniformity without aggregating them into a single fitness value.
    """
    # ``individual`` can sometimes contain more than three values. Only the
    # first three parameters (S1, S2, A1) are relevant for this evaluation.
    S1, S2, A1 = individual[:3]

    weighted_efficiency_total = 0
    weight_sum = 0
    efficiencies_per_angle = []  # store efficiency for each measurement angle
    upward_uni = []  # store upward energy standard deviation per angle when requested

    weights = [1, 2, 5, 7, 5, 8.5, 1.5, 2]

    for idx, angle in enumerate(range(10, 90, 10)):
        txt_path = os.path.join(folder, f"polar-{angle}.txt")
        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()

            data_lines = lines[6:]
            total_energy = 0
            upward_energy = 0
            upward_values = []

            angle_intensities = []
            for line in data_lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    polar_angle = float(parts[0])
                    intensity_col1 = float(parts[1])
                    total_energy += intensity_col1
                    angle_intensities.append(intensity_col1)
                    if polar_angle > 100:
                        upward_energy += intensity_col1
                        if return_uniformity:
                            upward_values.append(intensity_col1)
                except ValueError:
                    continue

            if total_energy > 0:
                eff = upward_energy / total_energy
            else:
                eff = 0.0

            efficiencies_per_angle.append(eff)
            weighted_efficiency_total += eff * weights[idx]
            weight_sum += weights[idx]

            if return_uniformity:
                upward_uni.append(
                    np.mean(upward_values) * (1 / np.std(upward_values))
                    if upward_values
                    else 0.0
                )

        except Exception as e:

            print(f"無法處理 {txt_path}: {e}")
            efficiencies_per_angle.append(0.0)
            continue

    if weight_sum == 0:
        efficiency = 0
    else:
        efficiency = weighted_efficiency_total / weight_sum

    try:
        process_score = compute_regression_score(S1, S2, A1)
    except Exception as e:
        print(f"⚠️ 製程品質評估失敗: {e}")
        process_score = 1.0

    if return_uniformity:
        uniformity = float(np.mean(upward_uni)) if upward_uni else 0.0

    else:
        uniformity = 0.0

    if return_uniformity:
        return (
            efficiency,
            process_score,
            uniformity,
            efficiencies_per_angle,
            upward_uni,
        )
    else:
        return (
            efficiency,
            process_score,
            0.0,
            efficiencies_per_angle,
            [],
        )