import os
import numpy as np
import random
import csv
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

# === GA參數設定 ===
POP_SIZE = 5  # μ
N_GENERATIONS = 3
OFFSPRING_SIZE = POP_SIZE # λ
CROSS_RATE = 0.6
MUTATE_RATE = 0.2
SIDE_BOUND = [400, 1000]
ANGLE_BOUND = [1, 179]
# ELITE_RATIO = 0.05 #5
VERBOSE = False

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

# === 迭代主迴圈 ===
for g in range(start_gen, N_GENERATIONS):
    fitness_log = load_fitness_log()

    # 評估親代
    fitness_values = []
    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        os.makedirs(folder, exist_ok=True)
        is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
        if is_evaluated:
            fitness, efficiency, process_score, angle_effs = eval_data
        else:
            try:
                Build_model(individual, mode="triangle", folder=folder)
                tracepro_fast(os.path.join(folder, "Sim.scm"))
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
            except:
                fitness, efficiency, process_score, angle_effs = 0.01, 0.0, 1.0, [0]*8
        append_fitness(fitness_log, individual, fitness, efficiency, process_score, g + 1, angle_effs)
        fitness_values.append(fitness)

    fitness_values = np.array(fitness_values)

    # 選出 μ 親代中的前 ELITE_COUNT
    # elite_count = max(1, int(POP_SIZE * ELITE_RATIO))
    # elite_indices = fitness_values.argsort()[-elite_count:][::-1]
    # elites = [pop[idx].copy() for idx in elite_indices]

    # 產生 λ 個子代
    selected = tournament_selection(pop, fitness_values, k=2)
    children = []
    while len(children) < OFFSPRING_SIZE:
        p1, p2 = selected[np.random.randint(POP_SIZE)], selected[np.random.randint(POP_SIZE)]
        child = crossover(p1.copy(), p2.copy())
        child = mutate(child)
        children.append(child)
    children = np.array(children)

    # 評估子代
    offspring_fitness = []
    for i, individual in enumerate(children):
        folder = os.path.join(save_root, f"P{i+1}")
        os.makedirs(folder, exist_ok=True)
        is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
        if is_evaluated:
            fitness, efficiency, process_score, angle_effs = eval_data
        else:
            try:
                Build_model(individual, mode="triangle", folder=folder)
                tracepro_fast(os.path.join(folder, "Sim.scm"))
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
            except:
                fitness, efficiency, process_score, angle_effs = 0.01, 0.0, 1.0, [0]*8
        append_fitness(fitness_log, individual, fitness, efficiency, process_score, g + 1, angle_effs)
        offspring_fitness.append(fitness)

    # 合併 μ+λ，選出新 μ 親代
    combined = np.vstack([pop, children])
    combined_fitness = np.hstack([fitness_values, offspring_fitness])
    best_indices = combined_fitness.argsort()[-POP_SIZE:][::-1]
    pop = combined[best_indices]

    print(f"★ Generation {g+1} 最佳個體: {pop[0]}, Fitness: {combined_fitness[best_indices[0]]:.2f}")

print("所有世代完成")
