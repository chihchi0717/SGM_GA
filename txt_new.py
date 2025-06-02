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

def evaluate_fitness(folder, individual):
    S1, S2, A1 = individual  # unpack 個體參數

    weighted_efficiency_total = 0
    weight_sum = 0

    # 對應角度10~80的權重，順序要與 range(10, 90, 10) 對齊
    weights = [1, 2, 5, 7, 5, 8.5, 1.5, 2]

    for idx, angle in enumerate(range(10, 90, 10)):
        txt_path = os.path.join(folder, f"polar-{angle}.txt")
        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()

            data_lines = lines[6:]
            total_energy = 0
            upward_energy = 0

            for line in data_lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    polar_angle = float(parts[0])
                    intensity_col1 = float(parts[1])
                    total_energy += intensity_col1
                    if polar_angle > 90:
                        upward_energy += intensity_col1
                except ValueError:
                    continue

            if total_energy > 0:
                efficiency = upward_energy / total_energy
                weight = weights[idx]
                weighted_efficiency_total += efficiency * weight
                weight_sum += weight

        except Exception as e:
            print(f"無法處理 {txt_path}: {e}")
            continue

    if weight_sum == 0:
        return 0

    efficiency = weighted_efficiency_total / weight_sum

    try:
        process_score = compute_regression_score(S1, S2, A1)
    except Exception as e:
        print(f"⚠️ 製程品質評估失敗: {e}")
        process_score = 1.0

    fitness = efficiency * (1 / (1 + process_score))
    return fitness, efficiency, process_score

