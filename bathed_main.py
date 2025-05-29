import os
import numpy as np
from PYtoAutocad_New0523_light_center_short import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

# === 參數設定 ===
population_size = 5
save_root = r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\GA_population"
os.makedirs(save_root, exist_ok=True)

# === 工具函式 ===
def generate_individual():
    side_a = np.random.randint(400, 900)
    side_b = np.random.randint(400, 900)
    angle_B = np.random.randint(20, 160)
    return [side_a, side_b, angle_B]

# === 建立初始族群 ===
populations = [generate_individual() for _ in range(population_size)]

# === 建模階段 ===
print("=== 建模階段 ===")
for i, individual in enumerate(populations):
    folder = os.path.join(save_root, f"P{i+1}")
    os.makedirs(folder, exist_ok=True)
    print(f"建模個體 P{i+1}: {individual}")
    Build_model(individual, mode="triangle", folder=folder)

# === 模擬階段 ===
print("\n=== 模擬階段 ===")
for i in range(population_size):
    folder = os.path.join(save_root, f"P{i+1}")
    scm_path = os.path.join(folder, "Sim.scm")
    print(f"模擬個體 P{i+1}")
    tracepro_fast(scm_path)

# === 評估階段 ===
print("\n=== 評估階段 ===")
fitness_values = []
for i in range(population_size):
    folder = os.path.join(save_root, f"P{i+1}")
    fitness = evaluate_fitness(folder)
    fitness_values.append(fitness)
    print(f"P{i+1} fitness: {fitness}")

# 可以進一步選出最佳個體
best_idx = np.argmax(fitness_values)
print(f"\n最佳個體為 P{best_idx+1}: {populations[best_idx]}, Fitness: {fitness_values[best_idx]}")
