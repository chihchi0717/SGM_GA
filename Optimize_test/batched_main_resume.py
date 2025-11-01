import os
import numpy as np
import random
import csv
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

# === GA參數設定 ===
POP_SIZE = 5
N_GENERATIONS = 3
CROSS_RATE = 0.6
MUTATE_RATE = 0.2
SIDE_BOUND = [400, 1000]
ANGLE_BOUND = [1, 179]
ELITE_COUNT = 2
VERBOSE = False  # 控制是否輸出細節訊息

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
onedrive_root = r"C:\Users\user\OneDrive - NTHU\home"
save_root = os.path.join(PROJECT_ROOT, "GA_population")
fitness_log_path = os.path.join(onedrive_root, "fitness_log.csv")

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
        except:
            pass
        attempts += 1
    return np.array(population)

def tournament_selection(pop, fitness, k=2):
    selected = []
    for _ in range(len(pop)):
        idxs = np.random.choice(np.arange(len(pop)), size=k, replace=False)
        best_idx = idxs[np.argmax(fitness[idxs])]
        selected.append(pop[best_idx])
    return np.array(selected)

def crossover(p1, p2):
    if np.random.rand() < CROSS_RATE:
        mask = np.random.randint(0, 2, len(p1)).astype(bool)
        return np.where(mask, p1, p2)
    return p1

def mutate(child):
    for i in range(len(child)):
        if np.random.rand() < MUTATE_RATE:
            if i < 2:
                child[i] = random.randint(*SIDE_BOUND)
            else:
                child[i] = random.randint(*ANGLE_BOUND)
    return child

def load_fitness_log():
    if not os.path.exists(fitness_log_path):
        return []
    with open(fitness_log_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_fitness_log(fitness_log):
    with open(fitness_log_path, mode="w", newline="") as f:
        fieldnames = ["S1", "S2", "A1", "fitness", "efficiency", "process_score", "generation"] + \
                     [f"eff_{angle}" for angle in range(10, 90, 10)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fitness_log:
            writer.writerow(row)

def check_if_evaluated(fitness_log, individual):
    S1, S2, A1 = map(str, individual)
    for row in fitness_log:
        if row["S1"] == S1 and row["S2"] == S2 and row["A1"] == A1:
            fitness = float(row["fitness"])
            efficiency = float(row["efficiency"])
            process_score = float(row["process_score"])
            angle_effs = [float(row.get(f"eff_{angle}", 0)) for angle in range(10, 90, 10)]
            return True, (fitness, efficiency, process_score, angle_effs)
    return False, None

def append_fitness(fitness_log, individual, fitness, efficiency, process_score, generation, angle_effs=None):
    S1, S2, A1 = map(str, individual)
    row = {
        "S1": S1,
        "S2": S2,
        "A1": A1,
        "fitness": str(fitness),
        "efficiency": str(efficiency),
        "process_score": str(process_score),
        "generation": str(generation)
    }
    if angle_effs:
        for angle, eff in zip(range(10, 90, 10), angle_effs):
            row[f"eff_{angle}"] = str(eff)
    fitness_log.append(row)
    save_fitness_log(fitness_log)

def get_last_completed_generation():
    fitness_log = load_fitness_log()
    if not fitness_log:
        return 0
    return max(int(row["generation"]) for row in fitness_log)

# === 判斷從哪一代開始 ===
start_gen = get_last_completed_generation()
if start_gen >= N_GENERATIONS:
    exit()
else:
    if VERBOSE:
        print(f"🔁 從第 {start_gen+1} 代執行至第 {N_GENERATIONS} 代")

# === 初始化族群 ===
if start_gen == 0:
    pop = generate_valid_population(POP_SIZE)
else:
    fitness_log = load_fitness_log()
    prev_population = []
    for row in fitness_log:
        if int(row["generation"]) == start_gen:
            prev_population.append([int(row["S1"]), int(row["S2"]), int(row["A1"])])
    pop = np.array(prev_population)

# === 主迴圈 ===
for g in range(start_gen, N_GENERATIONS):
    fitness_log = load_fitness_log()

    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        os.makedirs(folder, exist_ok=True)
        is_evaluated, _ = check_if_evaluated(fitness_log, individual)
        if not is_evaluated:
            try:
                Build_model(individual, mode="triangle", folder=folder)
            except:
                continue

    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        is_evaluated, _ = check_if_evaluated(fitness_log, individual)
        if not is_evaluated:
            scm_path = os.path.join(folder, "Sim.scm")
            tracepro_fast(scm_path)

    fitness_values = []
    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
        if is_evaluated:
            fitness, efficiency, process_score, angle_effs = eval_data
        else:
            try:
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
            except:
                fitness, efficiency, process_score, angle_effs = 0.01, 0.0, 1.0, [0]*8
        append_fitness(fitness_log, individual, fitness, efficiency, process_score, g + 1, angle_effs)
        fitness_values.append(fitness)

    # fitness_values = np.array(fitness_values)
    # best_idx = np.argmax(fitness_values)
    # print(f"★ Generation {g+1} 最佳個體為 P{best_idx+1}: {pop[best_idx]}, Fitness: {fitness_values[best_idx]:.2f}")

    # elite = pop[best_idx].copy()
    # selected = tournament_selection(pop, fitness_values, k=3)
    # next_gen = [elite]
    fitness_values = np.array(fitness_values)
    elite_indices = fitness_values.argsort()[-ELITE_COUNT:][::-1]
    elites = [pop[idx].copy() for idx in elite_indices]
    print(f"★ Generation {g+1} 最佳個體為 P{elite_indices[0]+1}: {pop[elite_indices[0]]}, Fitness: {fitness_values[elite_indices[0]]:.2f}")

    selected = tournament_selection(pop, fitness_values, k=2)
    next_gen = elites.copy()
    while len(next_gen) < POP_SIZE:
        p1, p2 = selected[np.random.randint(POP_SIZE)], selected[np.random.randint(POP_SIZE)]
        child = crossover(p1.copy(), p2.copy()) if np.random.rand() < CROSS_RATE else p1.copy()
        if np.random.rand() < MUTATE_RATE:
            child = mutate(child)
        # 避免重複個體進入族群
        if not any(np.array_equal(child, existing) for existing in next_gen):
            next_gen.append(child)

    pop = np.array(next_gen)

print("所有世代完成")
