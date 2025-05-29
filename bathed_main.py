import os
import numpy as np
import random
from draw_New import draw_
from PYtoAutocad_New0523_light_center_short import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

# === 參數設定 ===
POP_SIZE = 5
N_GENERATIONS = 3
CROSS_RATE = 0.6
MUTATE_RATE = 0.1
SIDE_BOUND = [400, 900]
ANGLE_BOUND = [1, 179]
save_root = r"C:\Users\user\Desktop\NTHU\MasterThesis\GA\SGM_GA\GA_population"
os.makedirs(save_root, exist_ok=True)

# === 工具函式 ===
def generate_valid_population(n):
    population = []
    attempts = 0
    while len(population) < n and attempts < n * 10:
        a = random.randint(*SIDE_BOUND)
        b = random.randint(*SIDE_BOUND)
        A = random.randint(*ANGLE_BOUND)
        param = [a, b, A]
        try:
            success, *_ = draw_(param, 1, 1, -1, -1, 0, 0)
            if success:
                population.append(param)
                print(f"✅ 有效個體: {param}")
            else:
                print(f"❌ draw_() 失敗: {param}")
        except Exception as e:
            print(f"⚠️ 例外: {param} -> {e}")
        attempts += 1
    return np.array(population)

def select(pop, fitness):
    probs = fitness / fitness.sum()
    idx = np.random.choice(np.arange(len(pop)), size=len(pop), p=probs)
    return pop[idx]

def crossover(p1, p2):
    if np.random.rand() < CROSS_RATE:
        mask = np.random.randint(0, 2, len(p1)).astype(bool)
        return np.where(mask, p1, p2)
    return p1  # 不做 crossover 就保留原 parent

    # mask = np.random.randint(0, 2, len(p1)).astype(bool)
    # return np.where(mask, p1, p2)

def mutate(child):
    for i in range(len(child)):
        if np.random.rand() < MUTATE_RATE:
            if i < 2:
                child[i] = random.randint(*SIDE_BOUND)
            else:
                child[i] = random.randint(*ANGLE_BOUND)
    return child

# === 主流程 ===
pop = generate_valid_population(POP_SIZE)

for g in range(N_GENERATIONS):
    print(f"\n=== Generation {g+1} ===")

    # === 建模階段 ===
    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        os.makedirs(folder, exist_ok=True)
        print(f"建模個體 P{i+1}: {individual}")
        Build_model(individual, mode="triangle", folder=folder)

    # === 模擬階段 ===
    for i in range(POP_SIZE):
        folder = os.path.join(save_root, f"P{i+1}")
        scm_path = os.path.join(folder, "Sim.scm")
        print(f"模擬個體 P{i+1}")
        tracepro_fast(scm_path)

    # === 評估階段 ===
    fitness_values = []
    for i in range(POP_SIZE):
        folder = os.path.join(save_root, f"P{i+1}")
        try:
            #fitness = evaluate_fitness(folder)
            fitness = evaluate_fitness(folder, individual)

        except Exception as e:
            print(f"⚠️ P{i+1} 評估錯誤: {e}")
            fitness = 0.01
        fitness_values.append(fitness)
        print(f"P{i+1} fitness: {fitness}")

    fitness_values = np.array(fitness_values)
    best_idx = np.argmax(fitness_values)
    print(f"★ Generation {g+1} 最佳個體為 P{best_idx+1}: {pop[best_idx]}, Fitness: {fitness_values[best_idx]:.2f}")

    # === 產生下一代 ===
    selected = select(pop, fitness_values)
    next_gen = []
    for _ in range(POP_SIZE):
        p1 = selected[np.random.randint(POP_SIZE)]
        p2 = selected[np.random.randint(POP_SIZE)]
        child = crossover(p1.copy(), p2.copy())
        child = mutate(child)
        next_gen.append(child)
    pop = np.array(next_gen)

print("✅ 所有世代完成")
