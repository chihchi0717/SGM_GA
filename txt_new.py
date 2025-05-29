import os
import numpy as np

# def read_txt_file(file_path):
#     angles = []
#     intensities = []

#     with open(file_path, 'r') as file:
#         lines = file.readlines()[2:]  # Skip headers
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) >= 2:
#                 angles.append(float(parts[0]))
#                 intensities.append(float(parts[1]))

#     return np.array(angles), np.array(intensities)
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

import os

def evaluate_fitness(folder):
    """
    計算一個資料夾中所有 polar-XX.txt 的總能量作為 Fitness。
    """
    total_energy = 0
    for angle in range(10, 90, 10):
        txt_path = os.path.join(folder, f"polar-{angle}.txt")
        try:
            with open(txt_path, "r") as f:
                lines = f.readlines()

            # 從第7行開始取數據 (index 6)，跳過表頭
            data_lines = lines[6:]
            angle_energy = 0
            for line in data_lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # 跳過空行或格式不正確的行
                try:
                    value = float(parts[1])  # 第二欄是強度
                    angle_energy += value
                except ValueError:
                    continue  # 跳過不能轉成數字的欄位

            total_energy += angle_energy

        except Exception as e:
            print(f"無法處理 {txt_path}: {e}")
            continue

    return total_energy

