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
    S1 = S1 / 1000 
    S2 = S2 / 1000
    return (
        -0.067 +
        0.217 * S1 +
        0.275 * S2 +
        0.002 * A1 -
        0.506 * S1 * S2 -
        0.002 * S1 * A1 -
        0.002 * S2 * A1 +
        0.004 * S1 * S2 * A1
    )

def evaluate_fitness(folder, individual, return_uniformity=False):
    """Evaluate fitness of an individual based on simulation results.

    Parameters
    ----------
    folder : str
        The folder containing ``polar-*.txt`` simulation output files.
    individual : sequence
        The ``(S1, S2, A1)`` parameters for regression scoring.
    return_uniformity : bool, optional
        If ``True`` the returned tuple will additionally contain overall
        uniformity and per-angle uniformities.  When ``False`` only the
        efficiency related metrics are returned for backward compatibility.

    Returns
    -------
    tuple
        When ``return_uniformity`` is ``False`` the return value is::

            (fitness, efficiency, process_score, efficiencies_per_angle)

        When ``True`` the return value becomes::

            (
                fitness,
                efficiency,
                process_score,
                uniformity,
                efficiencies_per_angle,
                uniformities_per_angle,
            )
    """

    S1, S2, A1 = individual

    weighted_efficiency_total = 0
    weight_sum = 0
    efficiencies_per_angle = []  # 儲存各角度效率
    intensities_all = []         # 儲存所有角度亮度資料
    uniformities_per_angle = []  # 儲存各角度均勻度

    weights = [1, 2, 5, 7, 5, 8.5, 1.5, 2]

    for idx, angle in enumerate(range(10, 90, 10)):
        txt_path = os.path.join(folder, f"polar-{angle}.txt")
        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()

            data_lines = lines[6:]
            total_energy = 0
            upward_energy = 0
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
                    if polar_angle > 90:
                        upward_energy += intensity_col1
                except ValueError:
                    continue

            if total_energy > 0:
                eff = upward_energy / total_energy
            else:
                eff = 0.0

            efficiencies_per_angle.append(eff)
            weighted_efficiency_total += eff * weights[idx]
            weight_sum += weights[idx]

            if return_uniformity and angle_intensities:
                intensities_all.append(angle_intensities)
                uniformities_per_angle.append(
                    uniformity(np.array(angle_intensities))
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
    
    w = 2
    fitness = efficiency * (1 / (1 + w * process_score))

    if return_uniformity:
        if intensities_all:
            all_vals = np.concatenate(intensities_all)
            overall_uni = uniformity(all_vals)
        else:
            overall_uni = 0.0
        return (
            fitness,
            efficiency,
            process_score,
            overall_uni,
            efficiencies_per_angle,
            uniformities_per_angle,
        )
    else:
        return fitness, efficiency, process_score, efficiencies_per_angle


