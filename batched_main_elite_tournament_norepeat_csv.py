import os
import numpy as np
import random
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness
import csv
# Representation : Real-valued vector (S1, S2, A1)
# Crossover : Uniform crossover
# Mutation : Random change within bounds
# Parents Selection : Tournament Selection
# Offsets Selection : (μ + λ) replacement, remain top
# fitness : efficiency * (1/(1 + process_score))

# === GA參數設定 ===
POP_SIZE = 5 
N_GENERATIONS = 3
CROSS_RATE = 0.6
MUTATE_RATE = 0.1
SIDE_BOUND = [400, 1000]
ANGLE_BOUND = [1, 179]

# 取得此檔案所在資料夾的絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 專案根目錄（假設此檔案就放在專案根目錄下）
PROJECT_ROOT = BASE_DIR
onedrive_root = r"C:\Users\user\OneDrive - NTHU\home"
# 相對路徑
save_root       = os.path.join(PROJECT_ROOT, "GA_population")
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
                print(f"✅ 有效個體: {param}")
            else:
                print(f"❌ draw_() 失敗: {param}")
        except Exception as e:
            print(f"⚠️ 例外: {param} -> {e}")
        attempts += 1
    return np.array(population)

def tournament_selection(pop, fitness, k=3):
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

# === fitness_log 相關處理 ===
def load_fitness_log():
    if not os.path.exists(fitness_log_path):
        return []
    with open(fitness_log_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_fitness_log(fitness_log):
    with open(fitness_log_path, mode="w", newline="") as f:
        # fieldnames = ["S1", "S2", "A1", "fitness", "efficiency", "process_score", "generation"]
        fieldnames = [
            "S1",
            "S2",
            "A1",
            "fitness",
            "efficiency",
            "process_score",
            "cv",
            "generation",
        ] + [f"eff_{angle}" for angle in range(10, 90, 10)] + [
            f"cv_{angle}" for angle in range(10, 90, 10)
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fitness_log:
            writer.writerow(row)

def save_fitness_log_backup(fitness_log, generation):
    backup_path = os.path.join(onedrive_root, f"generation_{generation}.csv")
    fieldnames = [
        "S1",
        "S2",
        "A1",
        "fitness",
        "efficiency",
        "process_score",
        "cv",
        "generation",
    ] + [f"eff_{angle}" for angle in range(10, 90, 10)] + [
        f"cv_{angle}" for angle in range(10, 90, 10)
    ]
    with open(backup_path, mode="w", newline="") as f:
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
            cv = float(row.get("cv", 0))
            angle_effs = [float(row.get(f"eff_{angle}", 0)) for angle in range(10, 90, 10)]
            angle_cvs = [float(row.get(f"cv_{angle}", 0)) for angle in range(10, 90, 10)]
            return True, (fitness, efficiency, process_score, cv, angle_effs, angle_cvs)
    return False, None

def append_fitness(
    fitness_log,
    individual,
    fitness,
    efficiency,
    process_score,
    cv,
    generation,
    angle_effs=None,
    angle_cvs=None,
):
    S1, S2, A1 = map(str, individual)
    row = {
        "S1": S1,
        "S2": S2,
        "A1": A1,
        "fitness": str(fitness),
        "efficiency": str(efficiency),
        "process_score": str(process_score),
        "cv": str(cv),
        "generation": str(generation)
    }

    if angle_effs:
        for angle, eff in zip(range(10, 90, 10), angle_effs):
            row[f"eff_{angle}"] = str(eff)
    if angle_cvs:
        for angle, cv_a in zip(range(10, 90, 10), angle_cvs):
            row[f"cv_{angle}"] = str(cv_a)

    fitness_log.append(row)
    save_fitness_log(fitness_log)


# === 初始化族群 ===
pop = generate_valid_population(POP_SIZE)

for g in range(N_GENERATIONS):
    print(f"\n=== Generation {g+1} ===")
    fitness_log = load_fitness_log()

    # === 建模階段 ===
    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        os.makedirs(folder, exist_ok=True)
        is_evaluated, _ = check_if_evaluated(fitness_log, individual)
        if not is_evaluated:
            print(f"🔧 建模個體 P{i+1}")
            #Build_model(individual, mode="triangle", folder=folder)
            try:
                Build_model(individual, mode="triangle", folder=folder)
            except Exception as e:
                print(f"❌ AutoCAD 建模失敗：個體 {individual}，錯誤：{e}")
                continue  # 或用預設 fitness 處理

        else:
            print(f"⏩ 已建模過的個體 P{i+1},{individual}，跳過")

    # === 模擬階段 ===
    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        is_evaluated, _ = check_if_evaluated(fitness_log, individual)
        if not is_evaluated:
            print(f"🔬 模擬個體 P{i+1}")
            scm_path = os.path.join(folder, "Sim.scm")
            tracepro_fast(scm_path)
        else:
            print(f"⏩ 已模擬過的個體 P{i+1}，跳過")

    # === 評估階段 ===
    fitness_values = []
    for i, individual in enumerate(pop):
        folder = os.path.join(save_root, f"P{i+1}")
        is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)

        if is_evaluated:
            fitness, efficiency, process_score, cv, angle_effs, angle_cvs = eval_data
            print(f"📄 讀取已存在的 fitness P{i+1}: {fitness:.4f}")
            append_fitness(
                fitness_log,
                individual,
                fitness,
                efficiency,
                process_score,
                cv,
                g + 1,
                angle_effs,
                angle_cvs,
            )
        else:
            try:
                fitness, efficiency, process_score, cv, angle_effs, angle_cvs = evaluate_fitness(
                    folder, individual, return_cv=True
                )
                print(
                    f"📊 評估完成 P{i+1}: {fitness:.4f} (eff: {efficiency:.4f}, proc: {process_score:.4f}, cv: {cv:.4f})"
                )
            except Exception as e:
                print(f"⚠️ P{i+1} 評估錯誤: {e}")
                fitness, efficiency, process_score, cv, angle_effs, angle_cvs = 0.01, 0.0, 1.0, 0.0, [0] * 8, [0.0] * 8

            append_fitness(
                fitness_log,
                individual,
                fitness,
                efficiency,
                process_score,
                cv,
                g + 1,
                angle_effs,
                angle_cvs,
            )

            # append_fitness(fitness_log, individual, fitness, efficiency, process_score, g + 1)

        fitness_values.append(fitness)
        print(f"P{i+1} pop[i]: {pop[i]}")

    fitness_values = np.array(fitness_values)
    best_idx = np.argmax(fitness_values)
    print(f"★ Generation {g+1} 最佳個體為 P{best_idx+1}: {pop[best_idx]}, Fitness: {fitness_values[best_idx]:.2f}")

    # === 菁英保留 + 建立下一代 ===
    elite = pop[best_idx].copy()
    selected = tournament_selection(pop, fitness_values, k=3)
    next_gen = [elite]
    while len(next_gen) < POP_SIZE:
        p1, p2 = selected[np.random.randint(POP_SIZE)], selected[np.random.randint(POP_SIZE)]
        child = crossover(p1.copy(), p2.copy()) if np.random.rand() < CROSS_RATE else p1.copy()
        if np.random.rand() < MUTATE_RATE:
            child = mutate(child)
        next_gen.append(child)

    pop = np.array(next_gen)
    save_fitness_log_backup(fitness_log, g+1) 

print("所有世代完成")
