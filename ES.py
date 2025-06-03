import os
import numpy as np
import random
import csv
from datetime import datetime
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

# === ES 參數設定 ===
POP_SIZE = 3                 # μ
OFFSPRING_SIZE = POP_SIZE  # λ
N_GENERATIONS = 1

# 基因範圍
SIDE_BOUND = [400, 1000]
ANGLE_BOUND = [1, 179]

# ES 自適應突變學習率 (n=3)
n = 3
TAU_PRIME = 1 / np.sqrt(2 * n)
TAU = 1 / np.sqrt(2 * np.sqrt(n))

# 據點 (seed) 設定（若要重現，可指定一個固定值，或每代都隨機）
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
onedrive_root = r"C:\Users\123\OneDrive - NTHU\411"
save_root = os.path.join(PROJECT_ROOT, "GA_population")

fitness_log_path = os.path.join(onedrive_root, "fitness_log.csv")

# === 工具函式 ===
def generate_valid_population(n_individuals):
    population = []
    attempts = 0
    while len(population) < n_individuals and attempts < n_individuals * 10:
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

def load_fitness_log():
    if not os.path.exists(fitness_log_path):
        return []
    with open(fitness_log_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_fitness_log(fitness_log):
    """
    新增了以下欄位：
      - sigma1, sigma2, sigma3: 策略參數
      - role: 'parent' 或 'offspring'
      - parent_idx1, parent_idx2: 父母索引
      - random_seed: 本筆紀錄所使用的隨機種子（可重現）
      - pop_size, offspring_size, tau_prime, tau: 超參數備註
    """
    fieldnames = [
        "generation", "role", "parent_idx1", "parent_idx2",
        "S1", "S2", "A1",
        "sigma1", "sigma2", "sigma3",
        "fitness", "efficiency", "process_score"
    ] + [f"eff_{angle}" for angle in range(10, 90, 10)] + [
        "random_seed", "pop_size", "offspring_size", "tau_prime", "tau"
    ]

    with open(fitness_log_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fitness_log:
            writer.writerow(row)

def check_if_evaluated(fitness_log, individual):
    """
    舊有行為保持不變：透過 (S1,S2,A1) 字串比對判斷是否已評估。
    """
    S1, S2, A1 = map(str, individual)
    for row in fitness_log:
        if row["S1"] == S1 and row["S2"] == S2 and row["A1"] == A1:
            fitness = float(row["fitness"])
            efficiency = float(row["efficiency"])
            process_score = float(row["process_score"])
            angle_effs = [float(row.get(f"eff_{angle}", 0)) for angle in range(10, 90, 10)]
            return True, (fitness, efficiency, process_score, angle_effs)
    return False, None

def append_fitness(fitness_log, individual, sigma, fitness, efficiency, process_score,
                   generation, angle_effs=None, role="parent", parent_idx1=-1, parent_idx2=-1, seed=None):
    """
    將一筆個體的結果追加到 fitness_log 並存回 CSV。此時要傳入的欄位有：
      - individual: [S1, S2, A1]
      - sigma: [sigma1, sigma2, sigma3]
      - fitness, efficiency, process_score, generation, angle_effs（list 長度 8）
      - role: 'parent' 或 'offspring'
      - parent_idx1, parent_idx2: 如果是 offspring，即父母在舊一代 pop_genes 陣列中的索引；若是 parent，則預設 -1
      - seed: 本筆使用的隨機種子（若 None，則取 GLOBAL_SEED）
    """
    S1, S2, A1 = individual
    sigma1, sigma2, sigma3 = sigma
    row = {
        "generation": generation,
        "role": role,
        "parent_idx1": parent_idx1,
        "parent_idx2": parent_idx2,
        "S1": S1,
        "S2": S2,
        "A1": A1,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "fitness": fitness,
        "efficiency": efficiency,
        "process_score": process_score,
        "random_seed": seed if seed is not None else GLOBAL_SEED,
        "pop_size": POP_SIZE,
        "offspring_size": OFFSPRING_SIZE,
        "tau_prime": TAU_PRIME,
        "tau": TAU
    }
    if angle_effs:
        for angle, eff in zip(range(10, 90, 10), angle_effs):
            row[f"eff_{angle}"] = eff
    fitness_log.append(row)
    save_fitness_log(fitness_log)

def get_last_completed_generation():
    fitness_log = load_fitness_log()
    if not fitness_log:
        return 0
    return max(int(row["generation"]) for row in fitness_log)

def clamp_gene(child):
    child[0] = int(np.clip(child[0], SIDE_BOUND[0], SIDE_BOUND[1]))
    child[1] = int(np.clip(child[1], SIDE_BOUND[0], SIDE_BOUND[1]))
    child[2] = int(np.clip(child[2], ANGLE_BOUND[0], ANGLE_BOUND[1]))
    return child

# === 判斷從哪一代開始 ===
start_gen = get_last_completed_generation()
if start_gen >= N_GENERATIONS:
    exit()
else:
    print(f"🔁 從第 {start_gen+1} 代執行至第 {N_GENERATIONS} 代")

# === 初始化族群（基因 + 步長） ===
if start_gen == 0:
    pop_genes = generate_valid_population(POP_SIZE)  # shape (μ, 3)
    # 初始步長: 各基因對應尺度的 10%
    sigma_side = (SIDE_BOUND[1] - SIDE_BOUND[0]) * 0.1
    sigma_angle = (ANGLE_BOUND[1] - ANGLE_BOUND[0]) * 0.1
    initial_sigmas = np.array([sigma_side, sigma_side, sigma_angle])
    pop_sigmas = np.tile(initial_sigmas, (POP_SIZE, 1))  # shape (μ, 3)
else:
    fitness_log = load_fitness_log()
    prev_population = []
    for row in fitness_log:
        if int(row["generation"]) == start_gen and row["role"] == "parent":
            prev_population.append([int(row["S1"]), int(row["S2"]), int(row["A1"])])
    pop_genes = np.array(prev_population)
    sigma_side = (SIDE_BOUND[1] - SIDE_BOUND[0]) * 0.1
    sigma_angle = (ANGLE_BOUND[1] - ANGLE_BOUND[0]) * 0.1
    initial_sigmas = np.array([sigma_side, sigma_side, sigma_angle])
    pop_sigmas = np.tile(initial_sigmas, (POP_SIZE, 1))

# === 迭代主迴圈 ===
for g in range(start_gen, N_GENERATIONS):
    fitness_log = load_fitness_log()

    # --- 評估父代：先畫 CAD，再全部模擬 ---
    for i, individual in enumerate(pop_genes):
        folder = os.path.join(save_root, f"P{i+1}")
        os.makedirs(folder, exist_ok=True)
        is_evaluated, _ = check_if_evaluated(fitness_log, individual)
        if not is_evaluated:
            try:
                Build_model(individual, mode="triangle", folder=folder)
            except:
                pass

    fitness_values = []
    for i, individual in enumerate(pop_genes):
        folder = os.path.join(save_root, f"P{i+1}")
        is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
        if is_evaluated:
            fitness, efficiency, process_score, angle_effs = eval_data
        else:
            try:
                tracepro_fast(os.path.join(folder, "Sim.scm"))
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
            except:
                fitness, efficiency, process_score, angle_effs = 0.01, 0.0, 1.0, [0]*8

            # 把父代紀錄寫入 CSV (role="parent")
            append_fitness(
                fitness_log=fitness_log,
                individual=individual,
                sigma=pop_sigmas[i],
                fitness=fitness,
                efficiency=efficiency,
                process_score=process_score,
                generation=g + 1,
                angle_effs=angle_effs,
                role="parent",
                parent_idx1=-1,
                parent_idx2=-1,
                seed=random.getrandbits(32)
            )
        fitness_values.append(fitness)
    fitness_values = np.array(fitness_values)

    # --- 產生 λ 個子代 (使用 ES 突變) ---
    children_genes = []
    children_sigmas = []
    children_parent_idxs = []  # 存放 (父 idx1, 父 idx2)，這邊我們只 隨機選 1 位父母 作突變
    for _ in range(OFFSPRING_SIZE):
        idx = random.randint(0, POP_SIZE - 1)
        parent_gene = pop_genes[idx].copy()
        parent_sigma = pop_sigmas[idx].copy()

        # 自適應突變 (log-normal) 更新步長
        new_sigma = parent_sigma * np.exp(
            TAU_PRIME * np.random.randn() + TAU * np.random.randn(n),
        )
        new_sigma = np.maximum(new_sigma, 1e-8)

        # 用更新後的步長做高斯突變
        child_gene = parent_gene + new_sigma * np.random.randn(n)
        child_gene = clamp_gene(child_gene)

        children_genes.append(child_gene.astype(int))
        children_sigmas.append(new_sigma)
        children_parent_idxs.append((idx, -1))  # 只有一位父母

    children_genes = np.array(children_genes)   # shape (λ, 3)
    children_sigmas = np.array(children_sigmas) # shape (λ, 3)

    # --- 評估子代：先畫 CAD，再全部仿真 ---
    for i, individual in enumerate(children_genes):
        folder = os.path.join(save_root, f"P{i+1}")
        os.makedirs(folder, exist_ok=True)
        is_evaluated, _ = check_if_evaluated(fitness_log, individual)
        if not is_evaluated:
            try:
                Build_model(individual, mode="triangle", folder=folder)
            except:
                pass

    offspring_fitness = []
    for i, individual in enumerate(children_genes):
        folder = os.path.join(save_root, f"P{i+1}")
        is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
        if is_evaluated:
            fitness, efficiency, process_score, angle_effs = eval_data
        else:
            try:
                tracepro_fast(os.path.join(folder, "Sim.scm"))
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
            except:
                fitness, efficiency, process_score, angle_effs = 0.01, 0.0, 1.0, [0]*8

            # 把子代紀錄寫入 CSV (role="offspring")
            parent_idx1, parent_idx2 = children_parent_idxs[i]
            append_fitness(
                fitness_log=fitness_log,
                individual=individual,
                sigma=children_sigmas[i],
                fitness=fitness,
                efficiency=efficiency,
                process_score=process_score,
                generation=g + 1,
                angle_effs=angle_effs,
                role="offspring",
                parent_idx1=parent_idx1,
                parent_idx2=parent_idx2,
                seed=random.getrandbits(32)
            )
        offspring_fitness.append(fitness)
    offspring_fitness = np.array(offspring_fitness)

    # --- 合併 μ+λ，選出下代 μ 個父母 ---
    combined_genes = np.vstack([pop_genes, children_genes])
    combined_sigmas = np.vstack([pop_sigmas, children_sigmas])
    combined_fitness = np.hstack([fitness_values, offspring_fitness])

    best_indices = np.argsort(combined_fitness)[-POP_SIZE:]
    pop_genes = combined_genes[best_indices]
    pop_sigmas = combined_sigmas[best_indices]

    print(f"★ Generation {g+1} 最佳個體: {pop_genes[-1]}, Fitness: {combined_fitness[best_indices[-1]]:.2f}")

print("所有世代完成")
