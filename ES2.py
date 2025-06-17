import os
import numpy as np
import random
import csv
import traceback
import shutil
import time

# 假設這些是您自己的模組
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness
import time
import shutil
from pywinauto import application, findwindows

import smtplib
import traceback
from email.message import EmailMessage
# 先定义好全局 log_dir
log_dir = r"C:\Users\User\OneDrive - NTHU\nuc"

# === ES 參數設定 ===
POP_SIZE = 5         # μ (親代數量)
OFFSPRING_SIZE = POP_SIZE *7 # λ (後代數量)
N_GENERATIONS = 100  # 總共要執行的世代數

# 基因範圍
SIDE_BOUND = [0.4, 1.0]
ANGLE_BOUND = [1, 179]

# ES 自適應突變學習率 (n=3)
n = 3
TAU_PRIME = 1 / np.sqrt(2 * n)
TAU = 1 / np.sqrt(2 * np.sqrt(n))

# 隨機種子（固定值可重現）
GLOBAL_SEED = 12
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_root = os.path.join(BASE_DIR, "GA_population")
# fitness_log_path = os.path.join(r"C:\Users\User\OneDrive - NTHU\nuc", "fitness_log.csv")
log_dir = os.path.join(r"C:\Users\User\OneDrive - NTHU\nuc")
os.makedirs(log_dir, exist_ok=True)
fitness_log_path = os.path.join(log_dir, "fitness_log.csv")
os.makedirs(save_root, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# === 工具函式 ===

def clamp_gene(child):
    """將基因限制在合法範圍內並進行精度處理"""
    child[0] = float(round(np.clip(child[0], SIDE_BOUND[0], SIDE_BOUND[1]), 2))
    child[1] = float(round(np.clip(child[1], SIDE_BOUND[0], SIDE_BOUND[1]), 2))
    child[2] = int(np.clip(child[2], ANGLE_BOUND[0], ANGLE_BOUND[1]))
    return child

def load_fitness_log():
    """從 CSV 讀取歷史紀錄"""
    if not os.path.exists(fitness_log_path):
        return []
    try:
        with open(fitness_log_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except (IOError, csv.Error) as e:
        print(f"⚠️ 讀取 fitness log 失敗: {e}. 將回傳空列表。")
        return []

def save_fitness_log(fitness_log):
    """將歷史紀錄寫入 CSV (更穩健的版本)"""
    if not fitness_log:
        return
    fieldnames = [
        "generation", "role", "parent_idx1", "parent_idx2",
        "S1", "S2", "A1",
        "sigma1", "sigma2", "sigma3",
        "fitness", "efficiency", "process_score"
    ] + [f"eff_{angle}" for angle in range(10, 90, 10)] + ["random_seed"]
    
    with open(fitness_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(fitness_log)

def check_if_evaluated(fitness_log, individual):
    """檢查個體是否已經被評估過，並回傳其數據"""
    S1_check, S2_check, A1_check = f"{individual[0]:.2f}", f"{individual[1]:.2f}", str(int(individual[2]))
    for row in reversed(fitness_log):
        if row.get("S1") == S1_check and row.get("S2") == S2_check and row.get("A1") == A1_check:
            try:
                fitness = float(row["fitness"])
                efficiency = float(row["efficiency"])
                process_score = float(row["process_score"])
                angle_effs = [float(row.get(f"eff_{angle}", 0.0)) for angle in range(10, 90, 10)]
                return True, (fitness, efficiency, process_score, angle_effs)
            except (ValueError, KeyError):
                continue
    return False, None

def append_fitness(fitness_log, individual, sigma, fitness, efficiency, process_score, generation, angle_effs, role, parent_indices, seed=None):
    """將新的評估結果加入歷史紀錄"""
    S1_check, S2_check, A1_check = f"{individual[0]:.2f}", f"{individual[1]:.2f}", str(int(individual[2]))
    for row in reversed(fitness_log):
        if (row.get("generation") == str(generation) and
            row.get("role") == role and
            row.get("S1") == S1_check and
            row.get("S2") == S2_check and
            row.get("A1") == A1_check):
            return 

    p_idx1, p_idx2 = parent_indices
    row = {
        "generation": generation,
        "role": role,
        "parent_idx1": p_idx1, "parent_idx2": p_idx2,
        "S1": S1_check, "S2": S2_check, "A1": A1_check,
        "sigma1": f"{sigma[0]:.6f}", "sigma2": f"{sigma[1]:.6f}", "sigma3": f"{sigma[2]:.6f}",
        "fitness": f"{fitness:.6f}", "efficiency": f"{efficiency:.6f}", "process_score": f"{process_score:.6f}",
        "random_seed": seed if seed is not None else GLOBAL_SEED
    }
    if angle_effs:
        for angle, eff in zip(range(10, 90, 10), angle_effs):
            row[f"eff_{angle}"] = f"{eff:.6f}"
    fitness_log.append(row)
    save_fitness_log(fitness_log)

def get_last_completed_generation(fitness_log):
    """從歷史紀錄中獲取最後一個完成的世代編號"""
    if not fitness_log: return 0
    parent_gens = [int(row["generation"]) for row in fitness_log if row.get("role") == "parent" and row.get("generation")]
    return max(parent_gens) if parent_gens else 0

def get_last_completed_generation():
    fitness_log = load_fitness_log()
    if not fitness_log:
        return 0
    return max(int(row["generation"]) for row in fitness_log if row["role"] == "parent")

    if start_gen == 0:
        print(f"🌱 從第 1 代全新開始執行")
        pop_genes = np.zeros((POP_SIZE, n))
        pop_genes[:, 0] = np.random.uniform(SIDE_BOUND[0], SIDE_BOUND[1], size=POP_SIZE)
        pop_genes[:, 1] = np.random.uniform(SIDE_BOUND[0], SIDE_BOUND[1], size=POP_SIZE)
        pop_genes[:, 2] = np.random.uniform(ANGLE_BOUND[0], ANGLE_BOUND[1], size=POP_SIZE)
        for i in range(POP_SIZE):
            pop_genes[i] = clamp_gene(pop_genes[i])
        
        sigma_side = (SIDE_BOUND[1] - SIDE_BOUND[0]) * 0.1
        sigma_angle = (ANGLE_BOUND[1] - ANGLE_BOUND[0]) * 0.1
        pop_sigmas = np.tile([sigma_side, sigma_side, sigma_angle], (POP_SIZE, 1))
        
        parent_eval_data = []
        print("\n--- 建立與評估初始族群 ---")
        # 初始族群仍然採用逐一處理的方式
        for i, individual in enumerate(pop_genes):
            folder = os.path.join(save_root, f"P_init_{i+1}") # 使用不同命名以避免與後代衝突
            os.makedirs(folder, exist_ok=True)
            print(f"建立初始模型 P{i+1}")
            Build_model(individual, mode="triangle", folder=folder)
            print(f"模擬評估初始模型 P{i+1}")
            tracepro_fast(os.path.join(folder, "Sim.scm"))
            fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
            
            eval_data = (fitness, efficiency, process_score, angle_effs)
            parent_eval_data.append(eval_data)
            append_fitness(
                fitness_log, individual, pop_sigmas[i],
                fitness, efficiency, process_score, 1, angle_effs, "parent", (-1, -1)
            )
        start_gen = 1
        
    else:
        print(f"🔁 從第 {start_gen + 1} 代繼續執行至第 {N_GENERATIONS} 代")
        last_gen_parents_rows = [row for row in fitness_log if row.get("role") == "parent" and int(row.get("generation", 0)) == start_gen]
        if len(last_gen_parents_rows) < POP_SIZE:
                print(f"❌ 錯誤：第 {start_gen} 代的親代紀錄不完整 ({len(last_gen_parents_rows)}/{POP_SIZE})。請檢查 log 檔。")
                return

        pop_genes, pop_sigmas, parent_eval_data = [], [], []
        print("\n--- 正在從 Log 恢復上一代親代狀態 ---")
        for i, row in enumerate(last_gen_parents_rows):
            try:
                gene = [float(row["S1"]), float(row["S2"]), float(row["A1"])]
                pop_genes.append(gene)
                sigma = [float(row["sigma1"]), float(row["sigma2"]), float(row["sigma3"])]
                pop_sigmas.append(sigma)
                fitness = float(row["fitness"])
                efficiency = float(row["efficiency"])
                process_score = float(row["process_score"])
                angle_effs = [float(row.get(f"eff_{angle}", 0.0)) for angle in range(10, 90, 10)]
                parent_eval_data.append((fitness, efficiency, process_score, angle_effs))
                append_fitness(
                    fitness_log, gene, sigma,
                    fitness, efficiency, process_score, start_gen+1,angle_effs, "parent", (-1, -1)
                )
                # 恢復時，不需要再 append_fitness，因為紀錄已經存在
                print(f"  [DEBUG] 已恢復親代 {i}: Gene={gene}, Fitness={fitness:.4f}")
            except (ValueError, KeyError) as e:
                print(f"❌ 致命錯誤：恢復親代數據時，log 檔案中的行內容不完整或格式錯誤。錯誤: {e}")
                print(f"  [DEBUG] 問題行: {row}")
                return
        pop_genes = np.array(pop_genes, dtype=float)
        pop_sigmas = np.array(pop_sigmas, dtype=float)

    # --- 演化主迴圈 ---
    for g in range(start_gen, N_GENERATIONS):
        current_gen = g + 1
        print(f"\n{'='*25} GENERATION {current_gen} {'='*25}")
        
        fitness_log = load_fitness_log()
        
        print(f"ℹ️  第 {current_gen} 代開始，已載入 {len(pop_genes)} 個親代，其適應度已知。")

        # --- 產生子代 ---
        children_genes, children_sigmas, children_parent_idxs = [], [], []
        for _ in range(OFFSPRING_SIZE):
            parent_idx = random.randint(0, POP_SIZE - 1)
            parent_gene, parent_sigma = pop_genes[parent_idx].copy(), pop_sigmas[parent_idx].copy()
            new_sigma = parent_sigma * np.exp(TAU_PRIME * np.random.randn() + TAU * np.random.randn(n))
            new_sigma = np.maximum(new_sigma, 0.02)
            if random.random() < 0.1: new_sigma = np.array([0.2, 0.2, 5.0])
            child_gene = clamp_gene(parent_gene + new_sigma * np.random.randn(n))
            children_genes.append(child_gene)
            children_sigmas.append(new_sigma)
            children_parent_idxs.append((parent_idx, -1))

        # =====================================================================
        # === 核心修改：分階段處理子代 ===
        # =====================================================================
        offspring_eval_data = [None] * OFFSPRING_SIZE
        needs_processing_indices = []

        # --- 階段 1：檢查所有子代狀態，找出需要處理的新個體 ---
        print("\n--- 步驟 1/4：檢查子代狀態 ---")
        for i, individual in enumerate(children_genes):
            is_evaluated, eval_data = check_if_evaluated(fitness_log, individual)
            if is_evaluated:
                print(f"  子代 P{i+1} 已在紀錄中，直接使用分數: {eval_data[0]:.4f}")
                offspring_eval_data[i] = eval_data
                fitness, efficiency, process_score, angle_effs = eval_data
                append_fitness(
                    fitness_log,
                    individual,
                    children_sigmas[i],  # 使用當前生成的子代的sigma
                    fitness,
                    efficiency,
                    process_score,
                    current_gen,
                    angle_effs,
                    "offspring",  # 角色是子代
                    children_parent_idxs[i]
                )
            else:
                print(f"  子代 P{i+1} 是新的，排入待處理佇列。")
                needs_processing_indices.append(i)
        
        # --- 階段 2：為所有新個體建立模型 ---
        print("\n--- 步驟 2/4：批次建立新子代模型 ---")
        if not needs_processing_indices:
            print("  所有子代皆已評估過，跳過此步驟。")
        else:
            for i in needs_processing_indices:
                individual = children_genes[i]
                folder = os.path.join(save_root, f"P{i+1}") 
                os.makedirs(folder, exist_ok=True, )
                print(f"  正在建立子代模型 P{i+1}...")
                # Build_model(individual, mode="triangle", folder=folder)
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

        # --- 階段 3：為所有新個體執行模擬 ---
        print("\n--- 步驟 3/4：批次執行新子代模擬 ---")
        if not needs_processing_indices:
            print("  所有子代皆已評估過，跳過此步驟。")
        else:
            for i in needs_processing_indices:
                folder = os.path.join(save_root, f"P{i+1}")
                print(f"  正在模擬評估子代 P{i+1}...")
                tracepro_fast(os.path.join(folder, "Sim.scm"))

        # --- 階段 4：計算新個體的適應度並寫入紀錄 ---
        print("\n--- 步驟 4/4：計算適應度與紀錄 ---")
        if not needs_processing_indices:
            print("  所有子代皆已評估過，跳過此步驟。")
        else:
            for i in needs_processing_indices:
                individual = children_genes[i]
                folder = os.path.join(save_root, f"P{i+1}")
                print(f"  正在計算子代 P{i+1} 的適應度...")
                fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, individual)
                
                eval_data = (fitness, efficiency, process_score, angle_effs)
                offspring_eval_data[i] = eval_data  # 將新計算出的結果填入
                
                # 將新結果寫入日誌
                append_fitness(
                    fitness_log, individual, children_sigmas[i],
                    fitness, efficiency, process_score, current_gen, angle_effs, "offspring",
                    children_parent_idxs[i]
                )
        
        # 安全檢查，確保所有子代都有評估數據
        if any(data is None for data in offspring_eval_data):
            raise RuntimeError(f"嚴重錯誤：在第 {current_gen} 代，並非所有子代都成功獲得評估數據！")

        # --- (μ+λ) 選擇 ---
        print("\n--- 選擇下一代 ---")
        combined_genes = np.vstack([pop_genes, children_genes])
        combined_sigmas = np.vstack([pop_sigmas, children_sigmas])
        combined_eval_data = parent_eval_data + offspring_eval_data
        combined_fitness = [d[0] for d in combined_eval_data]

        sorted_idx = np.argsort(combined_fitness)[::-1]

        new_parents, new_sigmas, new_eval_data, count_dict = [], [], [], {}
        MAX_DUPLICATE = 2
        selected_indices = set()
        
        # 第一輪：根據多樣性選擇
        for idx in sorted_idx:
            if len(new_parents) >= POP_SIZE: break
            gene = combined_genes[idx]
            key = (round(gene[0], 2), round(gene[1], 2), int(gene[2]))
            count = count_dict.get(key, 0)
            if count < MAX_DUPLICATE:
                new_parents.append(gene)
                new_sigmas.append(combined_sigmas[idx])
                new_eval_data.append(combined_eval_data[idx])
                count_dict[key] = count + 1
                selected_indices.add(idx)

        # 核心修正：增加了補位邏輯，確保親代數量永遠足夠
        if len(new_parents) < POP_SIZE:
            print(f"⚠️ 多樣性限制後只選出 {len(new_parents)} 個，將從剩餘最佳個體中補滿。")
            for idx in sorted_idx:
                if len(new_parents) >= POP_SIZE: break
                if idx not in selected_indices:
                    new_parents.append(combined_genes[idx])
                    new_sigmas.append(combined_sigmas[idx])
                    new_eval_data.append(combined_eval_data[idx])

        # 更新族群狀態，為下一個迴圈做準備
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
    except Exception as e:
        print(f"❌ 主程式發生致命錯誤: {e}")
        traceback.print_exc()
