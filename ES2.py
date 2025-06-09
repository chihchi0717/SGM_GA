import os
import numpy as np
import random
import csv
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast, load_macro
from txt_new import evaluate_fitness
import time
import shutil
from pywinauto import application, findwindows

import smtplib
import traceback
from email.message import EmailMessage
# 先定义好全局 log_dir
log_dir = r"C:\Users\User\OneDrive - NTHU\nuc"

def send_error(subject: str, body: str):
    try:
        err_dir = os.path.join(log_dir, "ES2_ErrorLogs")
        os.makedirs(err_dir, exist_ok=True)
        log_path = os.path.join(err_dir, "es2_error.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️ 已把错误日志存到：{log_path}")
    except Exception as write_err:
        # 写入 OneDrive 失败，退回写到当前工作目录
        fallback = os.path.join(os.getcwd(), "es2_error_fallback.log")
        with open(fallback, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️ OneDrive 写入失败，已写入本地：{fallback}")

# === ES 參數設定 ===
POP_SIZE = 5         # μ
OFFSPRING_SIZE = POP_SIZE * 7 # λ
N_GENERATIONS = 100

# 基因範圍
SIDE_BOUND = [0.4, 1]
ANGLE_BOUND = [1, 179]

# ES 自適應突變學習率 (n=3)
n = 3
TAU_PRIME = 1 / np.sqrt(2 * n)
TAU = 1 / np.sqrt(2 * np.sqrt(n))

# 隨機種子（固定值可重現）
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_root = os.path.join(BASE_DIR, "GA_population")
# fitness_log_path = os.path.join(r"C:\Users\User\OneDrive - NTHU\nuc", "fitness_log.csv")
log_dir = os.path.join(r"C:\Users\User\OneDrive - NTHU\nuc")
os.makedirs(log_dir, exist_ok=True)
fitness_log_path = os.path.join(log_dir, "fitness_log.csv")

# === 初始化 SCM 複製 ===
def copy_scm_to_all_folders():
    macro_dir = os.path.join(BASE_DIR, "Macro")
    scm_file = os.path.join(macro_dir, "Sim.scm")
    print(f"複製 SCM 檔案: {scm_file}")
    # 親代資料夾 P1 ~ P5
    for i in range(1, POP_SIZE + 1):
        folder = os.path.join(save_root, f"P{i}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(scm_file, folder)

    # 子代資料夾 C1 ~ C(POP_SIZE*7)
    for i in range(POP_SIZE+1, POP_SIZE+OFFSPRING_SIZE+1 + 1):
        folder = os.path.join(save_root, f"P{i}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(scm_file, folder)

copy_scm_to_all_folders()

# === 工具函式 ===

def generate_valid_population(n_individuals):
    population = []
    attempts = 0
    while len(population) < n_individuals and attempts < n_individuals * 10:
        a = random.uniform(SIDE_BOUND[0], SIDE_BOUND[1])
        b = random.uniform(SIDE_BOUND[0], SIDE_BOUND[1])
        # A = random.randint(*ANGLE_BOUND)
        A = random.uniform(ANGLE_BOUND[0], ANGLE_BOUND[1])
        param = [a, b, A]
        # 先 clamp，再交由 draw_ 檢查是否合法
        param = clamp_gene(param)
        try:
            success, *_ = draw_(param, 1, 1, -1, -1, 0, 0)
            if success:
                population.append(param)
        except Exception as e:
            print(f"generate_valid_population: draw_({param}) 失敗: {e}")
        attempts += 1
    return np.array(population, dtype=float)

def clamp_gene(child):
    # 1) clip 让 child[0], child[1] 落在 [0.4, 1.0] 之间
    child[0] = np.clip(child[0], SIDE_BOUND[0], SIDE_BOUND[1])
    child[1] = np.clip(child[1], SIDE_BOUND[0], SIDE_BOUND[1])
    # 2) 角度保持在 [1,179]，
    child[2] = np.clip(child[2], ANGLE_BOUND[0], ANGLE_BOUND[1])

    return child

def load_fitness_log():
    if not os.path.exists(fitness_log_path):
        return []
    with open(fitness_log_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

# def save_fitness_log(fitness_log):
#     fieldnames = [
#         "generation", "role", "parent_idx1", "parent_idx2",
#         "S1", "S2", "A1",
#         "sigma1", "sigma2", "sigma3",
#         "fitness", "efficiency", "process_score"
#     ] + [f"eff_{angle}" for angle in range(10, 90, 10)] + [
#         "random_seed"
#     ]
#     with open(fitness_log_path, mode="w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in fitness_log:
#             writer.writerow(row)

def save_fitness_log(fitness_log, path):
    fieldnames = [
        "generation", "role", "parent_idx1", "parent_idx2",
        "S1", "S2", "A1",
        "sigma1", "sigma2", "sigma3",
        "fitness", "efficiency", "process_score"
    ] + [f"eff_{angle}" for angle in range(10, 90, 10)] + ["random_seed"]
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fitness_log)

def check_if_evaluated(fitness_log, individual):
    S1, S2, A1 = f"{individual[0]:.2f}", f"{individual[1]:.2f}", str((individual[2]))
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
    S1 = f"{round(individual[0], 2):.2f}"
    S2 = f"{round(individual[1], 2):.2f}"
    A1 = str(int(individual[2]))
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
        "random_seed": seed if seed is not None else GLOBAL_SEED
    }
    if angle_effs:
        for angle, eff in zip(range(10, 90, 10), angle_effs):
            row[f"eff_{angle}"] = eff
    fitness_log.append(row)
    save_fitness_log(fitness_log, fitness_log_path)


def get_last_completed_generation():
    fitness_log = load_fitness_log()
    if not fitness_log:
        return 0
    return max(int(row["generation"]) for row in fitness_log if row["role"] == "parent")

# === 主程式 ===
def main():
    start_gen = get_last_completed_generation()
    if start_gen == 0:
        # 第一次執行：寫 run_config
        config_path = os.path.join(save_root, "run_config.txt")
        os.makedirs(save_root, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as cf:
            cf.write(f"POP_SIZE={POP_SIZE}\n")
            cf.write(f"OFFSPRING_SIZE={OFFSPRING_SIZE}\n")
            cf.write(f"N_GENERATIONS={N_GENERATIONS}\n")
            cf.write(f"TAU_PRIME={TAU_PRIME}\n")
            cf.write(f"TAU={TAU}\n")
            cf.write(f"GLOBAL_SEED={GLOBAL_SEED}\n")
            cf.write(f"SIDE_BOUND={SIDE_BOUND}\n")
            cf.write(f"ANGLE_BOUND={ANGLE_BOUND}\n")
    else:
        print(f"🔁 從第 {start_gen+1} 代執行至第 {N_GENERATIONS} 代")

    # 初始化族群
    if start_gen == 0:
        pop_genes = generate_valid_population(POP_SIZE)  # shape (μ, 3)，float array
        sigma_side = (SIDE_BOUND[1] - SIDE_BOUND[0]) * 0.2
        sigma_angle = 3
        initial_sigmas = np.array([sigma_side, sigma_side, sigma_angle])
        pop_sigmas = np.tile(initial_sigmas, (POP_SIZE, 1))  # shape (μ, 3)
    else:
        fitness_log = load_fitness_log()
        prev_population = []
        for row in fitness_log:
            if int(row["generation"]) == start_gen and row["role"] == "parent":
                prev_population.append([
                    float(row["S1"]),
                    float(row["S2"]),
                    float(row["A1"])
                ])
        pop_genes = np.array(prev_population, dtype=float)
        sigma_side = (SIDE_BOUND[1] - SIDE_BOUND[0]) * 0.2
        sigma_angle = 3
        initial_sigmas = np.array([sigma_side, sigma_side, sigma_angle])
        pop_sigmas = np.tile(np.array([0.05*(SIDE_BOUND[1]-SIDE_BOUND[0]),
                                    0.05*(SIDE_BOUND[1]-SIDE_BOUND[0]),
                                    1.0]), (POP_SIZE, 1))

    # 迭代主迴圈
    for g in range(start_gen, N_GENERATIONS):
        fitness_log = load_fitness_log()

        # --- 評估父代：畫 CAD → 全部模擬 ---
        for i, individual in enumerate(pop_genes):
            folder = os.path.join(save_root, f"P{i+1}")
            os.makedirs(folder, exist_ok=True)
            is_evaluated, _ = check_if_evaluated(fitness_log, individual)
            MAX_RETRY = 3  # 最多重試次數

            if not is_evaluated:
                build_success = False
                attempt = 0
                while build_success == False:
                    try:
                        result, log = Build_model(individual, mode="triangle", folder=folder)
                        for msg in log:
                            print(msg)
                        if result == 1:
                            build_success = True
                            break
                    except Exception as e:
                        print(f"❌ Build_model 第 {attempt+1} 次失敗：{e}")
                    time.sleep(1)  # 等一秒再試（讓 AutoCAD 有時間反應）

                # if not build_success:
                #     print(f"❌ 個體 {individual} 多次建模失敗，跳過後續模擬")
                #     raise 

                    
        app = application.Application().connect(path=r"C:\Program Files (x86)\Lambda Research Corporation\TracePro\TracePro.exe")
        reset_path = os.path.join(BASE_DIR, "Macro", "Reset.scm")
        fitness_values = []
        for i, individual in enumerate(pop_genes):
            folder = os.path.join(save_root, f"P{i+1}")
            is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
            if is_evaluated:
                fitness, efficiency, process_score, angle_effs = eval_data
            else:  
                tracepro_fast(os.path.join(folder, "Sim.scm"))
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
            if(fitness is None or efficiency is None or process_score is None):
                print(f"⚠️ 個體 {individual} 評估失敗，跳過後續模擬")
            else:
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
                    seed=random.randint(0, 2**31)
                )
                fitness_values.append(fitness)
        fitness_values = np.array(fitness_values)

        # --- 產生 λ 個子代 (ES 突變) ---
        children_genes = []
        children_sigmas = []
        children_parent_idxs = []
        for _ in range(OFFSPRING_SIZE):
            idx = random.randint(0, POP_SIZE - 1)
            parent_gene = pop_genes[idx].copy()
            parent_sigma = pop_sigmas[idx].copy()

            # ES 突變
            new_sigma = parent_sigma * np.exp(
                TAU_PRIME * np.random.randn() + TAU * np.random.randn(n)
            )
            new_sigma = np.maximum(new_sigma, 0.02)

            # === 10% 機率，強制 reset 為大突變 ===
            if random.random() < 0.1:
                new_sigma = np.array([0.2, 0.2, 5.0])  # 重新擴大突變幅度

            child_gene = parent_gene + new_sigma * np.random.randn(n)
            child_gene = clamp_gene(child_gene)

            # 列印 debug，確認不為 0
            # print(f"DEBUG (child before clamp) : {parent_gene + new_sigma * np.random.randn(n)}")
            # print(f"DEBUG (child after clamp)  : {child_gene}")

            children_genes.append(child_gene)        # <-- **去掉 .astype(int)**
            children_sigmas.append(new_sigma)
            children_parent_idxs.append((idx, -1))

        children_genes = np.array(children_genes, dtype=float)
        children_sigmas = np.array(children_sigmas)

        # --- 評估子代：畫 CAD → 全部模擬 ---
        for i, individual in enumerate(children_genes):
            folder = os.path.join(save_root, f"P{i+1}")   # <-- 改名稱不與 P 系列重複
            os.makedirs(folder, exist_ok=True)
            is_evaluated, _ = check_if_evaluated(fitness_log, individual)
            MAX_RETRY = 3  # 最多重試次數

            if not is_evaluated:
                build_success = False
                attempt = 0
                while build_success == False:
                    try:
                        result, log = Build_model(individual, mode="triangle", folder=folder)
                        for msg in log:
                            print(msg)
                        if result == 1:
                            build_success = True
                            break
                    except Exception as e:
                        print(f"❌ Build_model 第 {attempt+1} 次失敗：{e}")
                    time.sleep(1)  # 等一秒再試（讓 AutoCAD 有時間反應）

                # if not build_success:
                #     print(f"❌ 個體 {individual} 多次建模失敗，跳過後續模擬")
                #     continue  # 或直接 return / raise Error 

        offspring_fitness = []
        for i, individual in enumerate(children_genes):
            folder = os.path.join(save_root, f"P{i+1}")
            is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
            if is_evaluated:
                fitness, efficiency, process_score, angle_effs = eval_data
            else:
                tracepro_fast(os.path.join(folder, "Sim.scm"))
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)

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
                seed=random.randint(0, 2**31)
            )
            offspring_fitness.append(fitness)
        offspring_fitness = np.array(offspring_fitness)

        # --- 合併 μ+λ，選出下一代 μ，最多保留 2 個相同基因 ---
        combined_genes = np.vstack([pop_genes, children_genes])
        combined_sigmas = np.vstack([pop_sigmas, children_sigmas])
        combined_fitness = np.hstack([fitness_values, offspring_fitness])

        # 先將所有個體依 fitness 排序（由大到小）
        sorted_idx = np.argsort(combined_fitness)[::-1]

        new_parents = []
        new_sigmas = []
        count_dict = {}   # 用來記錄同一組基因已經選過幾次
        MAX_DUPLICATE = 2 # 最多允許同一 (S1,S2,A1) 出現 2 次

        for idx in sorted_idx:
            if len(new_parents) >= POP_SIZE:
                break

            gene = combined_genes[idx]
            # 基因取值用 clamp_gene 相同的精度：S1, S2 取到小數第 2 位，A1 取整數
            key = (round(gene[0], 2), round(gene[1], 2), int(gene[2]))
            count = count_dict.get(key, 0)

            if count < MAX_DUPLICATE:
                new_parents.append(gene)
                new_sigmas.append(combined_sigmas[idx])
                count_dict[key] = count + 1

        # 最後把 new_parents、new_sigmas 轉回 numpy array
        best_indices = np.argsort(combined_fitness)[-POP_SIZE:]
        pop_genes = np.array(new_parents, dtype=float)
        pop_sigmas = np.array(new_sigmas, dtype=float)


        print(f"★ Generation {g+1} 最佳個體: {pop_genes[-1]}, Fitness: {combined_fitness[best_indices[-1]]:.2f}")
            
        # 產生動態檔名並存檔
        # 1) 過濾出這一代所有的 log 列
        this_gen_rows = [row for row in fitness_log if int(row["generation"]) == g+1]
        # 2) 計算最高 fitness
        max_f = max(float(row["fitness"]) for row in this_gen_rows)
        # 3) 組出檔名
        fname = f"fitness_gen{g+1}_max{max_f:.2f}.csv"
        out_path = os.path.join(log_dir, fname)
        # 4) 存檔
        save_fitness_log(fitness_log, out_path)
        print(f"已儲存第 {g+1} 代紀錄到：{fname}")


    print("所有世代完成")

import os
from datetime import datetime
# … 其他 imports …

# 先定义好全局 log_dir
log_dir = r"C:\Users\User\OneDrive - NTHU\nuc"

def send_error(subject: str, body: str):
    try:
        err_dir = os.path.join(log_dir, "ES2_ErrorLogs")
        os.makedirs(err_dir, exist_ok=True)
        log_path = os.path.join(err_dir, "es2_error.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️ 已把错误日志存到：{log_path}")
    except Exception as write_err:
        # 写入 OneDrive 失败，退回写到当前工作目录
        fallback = os.path.join(os.getcwd(), "es2_error_fallback.log")
        with open(fallback, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️ OneDrive 写入失败，已写入本地：{fallback}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print("❌ 捕获到主程式异常：", tb)
        # 强制调用一次测试写日志
        send_error("[TEST] ES2.py 錯誤通知", tb)
        # 重新抛出，让程序看到错误
        raise
