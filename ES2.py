import os
import numpy as np
import random
import csv
import traceback
import shutil
import time
import re
from datetime import datetime
import gc

# 假設這些是您自己的模組
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

# 從您的程式中保留的 imports
from pywinauto import application, findwindows
import smtplib
from email.message import EmailMessage

# === ES 參數設定 ===
POP_SIZE = 10  # μ (親代數量)
OFFSPRING_SIZE = POP_SIZE *7  # λ (後代數量)
N_GENERATIONS = 100  # 總共要執行的世代數

# --- AutoCAD 建模參數 ---
BUILD_MODE = "triangle"
BUILD_FILLET = 2
VERTEX_RADIUS = 0.022
INSIDE_RADIUS = 0.088
LIGHT_SOURCE_SIZE = 0.5

# --- Fitness 評估參數 ---
RETURN_UNIFORMITY = True
EFF_WEIGHT = 1
PROCESS_WEIGHT = 1
UNI_WEIGHT = 1

# 基因範圍
SIDE_BOUND = [0.4, 1.6]
ANGLE_BOUND = [60, 90]

# ES 自適應突變學習率 (n=3)
n = 3
TAU_PRIME = 1 / np.sqrt(2 * n)
TAU = 1 / np.sqrt(2 * np.sqrt(n))
SIGMA_SCALE = 2.0  # >1 會放大突變幅度

# 隨機種子（固定值可重現）
GLOBAL_SEED = 50
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_root = os.path.join(BASE_DIR, "GA_population")
# 【保留您的變數】使用 log_dir 作為日誌的根目錄
log_dir = r"C:\Users\user\OneDrive - NTHU\home"
os.makedirs(save_root, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def write_run_config():
    """將執行設定輸出至 log_dir/run_config.txt"""
    config = {
        "POP_SIZE": POP_SIZE,
        "OFFSPRING_SIZE": OFFSPRING_SIZE,
        "N_GENERATIONS": N_GENERATIONS,
        "BUILD_MODE": BUILD_MODE,
        "BUILD_FILLET": BUILD_FILLET,
        "VERTEX_RADIUS": VERTEX_RADIUS,
        "INSIDE_RADIUS": INSIDE_RADIUS,
        "LIGHT_SOURCE_SIZE": LIGHT_SOURCE_SIZE,
        "SIDE_BOUND": SIDE_BOUND,
        "ANGLE_BOUND": ANGLE_BOUND,
        "TAU_PRIME": TAU_PRIME,
        "TAU": TAU,
        "GLOBAL_SEED": GLOBAL_SEED,
        "RETURN_UNIFORMITY": RETURN_UNIFORMITY,
        "EFF_WEIGHT": EFF_WEIGHT,
        "PROCESS_WEIGHT": PROCESS_WEIGHT,
        "UNI_WEIGHT": UNI_WEIGHT,
        "save_root": save_root,
        "log_dir": log_dir,
    }
    try:
        cfg_path = os.path.join(log_dir, "run_config.txt")
        with open(cfg_path, "w", encoding="utf-8") as f:
            for k, v in config.items():
                f.write(f"{k} = {v}\n")
        print(f"🔧 執行設定已輸出到 {cfg_path}")
    except Exception as e:
        print(f"⚠️  無法寫入 run_config.txt: {e}")


# === 錯誤紀錄函式 (使用您的函式名) ===
def send_error(subject: str, body: str):
    """將錯誤訊息寫入本地檔案"""
    try:
        err_dir = os.path.join(log_dir, "ES_ErrorLogs")
        os.makedirs(err_dir, exist_ok=True)
        log_path = os.path.join(err_dir, "es_error.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️  錯誤日誌已存到：{log_path}")
    except Exception as write_err:
        fallback = os.path.join(os.getcwd(), "es_error_fallback.log")
        with open(fallback, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️  OneDrive 寫入失敗，錯誤日誌已寫入本地：{fallback}")


# === 工具函式 (分代日誌系統) ===


def clamp_gene(child):
    """將基因限制在合法範圍內並進行精度處理"""
    child[0] = float(round(np.clip(child[0], SIDE_BOUND[0], SIDE_BOUND[1]), 2))
    child[1] = float(round(np.clip(child[1], SIDE_BOUND[0], SIDE_BOUND[1]), 2))
    child[2] = int(np.clip(child[2], ANGLE_BOUND[0], ANGLE_BOUND[1]))
    return child


def save_generation_log(generation_data, file_path):
    """將單一世代的歷史紀錄寫入指定的 CSV"""
    if not generation_data:
        return
    fieldnames = (
        [
            "generation",
            "role",
            "parent_idx1",
            "parent_idx2",
            "S1",
            "S2",
            "A1",
            "sigma1",
            "sigma2",
            "sigma3",
            "fitness",
            "efficiency",
            "process_score",
            "uniformity",
        ]
        + [f"eff_{angle}" for angle in range(10, 90, 10)]
        + [f"uni_{angle}" for angle in range(10, 90, 10)]
        + ["random_seed"]
    )

    try:
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(generation_data)
    except IOError as e:
        print(f"❌ 寫入日誌檔 {file_path} 失敗: {e}")


def create_log_row(
    individual, sigma, fitness_data, generation, role, parent_indices, seed=None
):
    """建立一筆日誌紀錄的字典物件"""
    if len(fitness_data) >= 6:
        fitness, efficiency, process_score, uniformity, angle_effs, angle_unis = (
            fitness_data[:6]
        )
    else:
        fitness, efficiency, process_score, angle_effs = fitness_data
        uniformity = 0.0
        angle_unis = []
    p_idx1, p_idx2 = parent_indices
    row = {
        "generation": generation,
        "role": role,
        "parent_idx1": p_idx1,
        "parent_idx2": p_idx2,
        "S1": f"{individual[0]:.2f}",
        "S2": f"{individual[1]:.2f}",
        "A1": str(int(individual[2])),
        "sigma1": f"{sigma[0]:.6f}",
        "sigma2": f"{sigma[1]:.6f}",
        "sigma3": f"{sigma[2]:.6f}",
        "fitness": f"{fitness:.6f}",
        "efficiency": f"{efficiency:.6f}",
        "process_score": f"{process_score:.6f}",
        "uniformity": f"{uniformity:.6f}",
        "random_seed": seed if seed is not None else GLOBAL_SEED,
    }
    if angle_effs:
        for angle, eff in zip(range(10, 90, 10), angle_effs):
            row[f"eff_{angle}"] = f"{eff:.6f}"
    if angle_unis:
        for angle, uni_a in zip(range(10, 90, 10), angle_unis):
            row[f"uni_{angle}"] = f"{uni_a:.6f}"
    return row


def evaluate_objectives(fitness_data):
    """Extract objectives [efficiency, -process_score, uniformity] from fitness data."""
    if len(fitness_data) >= 4:
        _, efficiency, process_score, uniformity = fitness_data[:4]
    else:
        return np.array([0.0, 0.0, 0.0])
    return np.array([efficiency, -process_score, uniformity])


def fast_non_dominated_sort(objectives):
    """Perform non-dominated sorting and return list of fronts (list of indices)."""
    S = [set() for _ in objectives]
    domination_count = [0 for _ in objectives]
    fronts = [[]]
    for p, obj_p in enumerate(objectives):
        for q, obj_q in enumerate(objectives):
            if p == q:
                continue
            if np.all(obj_p >= obj_q) and np.any(obj_p > obj_q):
                S[p].add(q)
            elif np.all(obj_q >= obj_p) and np.any(obj_q > obj_p):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(front_indices, objectives):
    """Compute crowding distance for individuals in one front."""
    if not front_indices:
        return {}
    num_obj = objectives.shape[1]
    distances = {idx: 0.0 for idx in front_indices}
    front_objs = objectives[front_indices]
    for m in range(num_obj):
        obj_values = front_objs[:, m]
        order = np.argsort(obj_values)
        distances[front_indices[order[0]]] = float("inf")
        distances[front_indices[order[-1]]] = float("inf")
        min_v = obj_values[order[0]]
        max_v = obj_values[order[-1]]
        if max_v - min_v == 0:
            continue
        for i in range(1, len(front_indices) - 1):
            prev_v = obj_values[order[i - 1]]
            next_v = obj_values[order[i + 1]]
            distances[front_indices[order[i]]] += (next_v - prev_v) / (max_v - min_v)
    return distances


def find_last_completed_generation(directory):
    """從日誌目錄掃描所有檔案，找到最新的世代編號和對應的檔案路徑"""
    if not os.path.exists(directory):
        return 0, None

    pattern = re.compile(r"fitness_gen(\d+)_max.*\.csv")
    last_gen = 0
    last_gen_filepath = None

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            gen_num = int(match.group(1))
            if gen_num > last_gen:
                last_gen = gen_num
                last_gen_filepath = os.path.join(directory, filename)

    return last_gen, last_gen_filepath


def check_if_evaluated(fitness_log, individual):
    """檢查個體是否已經被評估過，並回傳其數據"""
    S1_check, S2_check, A1_check = (
        f"{individual[0]:.2f}",
        f"{individual[1]:.2f}",
        str(int(individual[2])),
    )
    for row in reversed(fitness_log):
        if (
            row.get("S1") == S1_check
            and row.get("S2") == S2_check
            and row.get("A1") == A1_check
        ):
            try:
                fitness = float(row["fitness"])
                efficiency = float(row["efficiency"])
                process_score = float(row["process_score"])
                cv = float(row.get("cv", 0.0))
                uniformity = float(row.get("uniformity", 1.0 - cv))
                angle_effs = [
                    float(row.get(f"eff_{angle}", 0.0)) for angle in range(10, 90, 10)
                ]
                angle_unis = []
                for angle in range(10, 90, 10):
                    val = row.get(f"uni_{angle}")
                    if val not in (None, ""):
                        try:
                            angle_unis.append(float(val))
                            continue
                        except ValueError:
                            pass
                    cv_val = row.get(f"cv_{angle}")
                    if cv_val not in (None, ""):
                        try:
                            cv_a = float(cv_val)
                        except ValueError:
                            cv_a = 1.0
                    else:
                        cv_a = 1.0
                    angle_unis.append(max(0.0, 1.0 - cv_a))
                return True, (
                    fitness,
                    efficiency,
                    process_score,
                    uniformity,
                    angle_effs,
                    angle_unis,
                )
            except (ValueError, KeyError):
                continue
    return False, None


def build_model_with_retry(individual, folder, max_attempts=3):
    """Build a model with retries to handle transient AutoCAD errors."""
    attempt = 0
    while attempt < max_attempts:
        try:
            result, log = Build_model(
                individual,
                mode=BUILD_MODE,
                folder=folder,
                fillet=BUILD_FILLET,
                radius_vertex=VERTEX_RADIUS,
                radius_inside=INSIDE_RADIUS,
                light_source_length=LIGHT_SOURCE_SIZE,
            )
            for msg in log:
                print(msg)
            if result == 1:
                return True
        except Exception as e:
            print(f"❌ Build_model 第 {attempt+1} 次失敗：{e}")
            time.sleep(1)
        attempt += 1
    return False


def simulate_and_evaluate(folder, individual):
    """Run TracePro simulation and evaluate fitness."""
    while True:
        try:
            tracepro_fast(os.path.join(folder, "Sim.scm"))
            return evaluate_fitness(
                folder,
                individual,
                return_uniformity=RETURN_UNIFORMITY,
                eff_weight=EFF_WEIGHT,
                process_weight=PROCESS_WEIGHT,
                uni_weight=UNI_WEIGHT,
            )
        except Exception as e:
            print(f"⚠️ tracepro/evaluate_fitness(parent {individual}) 失敗: {e}")
            time.sleep(1)


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

    # 子代資料夾 P1 ~ P(POP_SIZE*7)
    for i in range(POP_SIZE + 1, POP_SIZE + OFFSPRING_SIZE + 1 + 1):
        folder = os.path.join(save_root, f"P{i}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(scm_file, folder)


def main():
    copy_scm_to_all_folders()
    """主執行函式"""
    write_run_config()
    start_gen, last_gen_filepath = find_last_completed_generation(log_dir)

    pop_genes = None
    pop_sigmas = None
    parent_eval_data = []

    if start_gen == 0:
        print(f"🌱 找不到任何日誌檔，從第 1 代全新開始執行")
        pop_genes = np.zeros((POP_SIZE, n))
        pop_genes[:, 0] = np.random.uniform(SIDE_BOUND[0], SIDE_BOUND[1], size=POP_SIZE)
        pop_genes[:, 1] = np.random.uniform(SIDE_BOUND[0], SIDE_BOUND[1], size=POP_SIZE)
        pop_genes[:, 2] = np.random.uniform(
            ANGLE_BOUND[0], ANGLE_BOUND[1], size=POP_SIZE
        )
        for i in range(POP_SIZE):
            pop_genes[i] = clamp_gene(pop_genes[i])

        sigma_side = (SIDE_BOUND[1] - SIDE_BOUND[0]) * 0.1
        sigma_angle = (ANGLE_BOUND[1] - ANGLE_BOUND[0]) * 0.1
        pop_sigmas = np.tile([sigma_side, sigma_side, sigma_angle], (POP_SIZE, 1))

        # 【修改】批次處理初始族群
        print("\n--- 建立與評估初始族群 (第 1 代的親代) ---")

        build_results = [False] * POP_SIZE
        print("\n--- 步驟 1/3: 批次建立初始族群模型 ---")
        for i, individual in enumerate(pop_genes):
            folder = os.path.join(save_root, f"P{i+1}")
            os.makedirs(folder, exist_ok=True)
            print(f"  建立初始模型 P{i+1}...")
            build_results[i] = build_model_with_retry(individual, folder)
            if not build_results[i]:
                print(f"❌ 建立模型 P{i+1} 最終失敗。")

        print("\n--- 步驟 2/3: 批次模擬初始族群模型 ---")
        for i, success in enumerate(build_results):
            if success:
                folder = os.path.join(save_root, f"P{i+1}")
                print(f"  模擬初始模型 P{i+1}...")
                simulate_and_evaluate(folder, pop_genes[i])

        print("\n--- 步驟 3/3: 批次評估初始族群適應度 ---")
        initial_gen_log = []
        parent_eval_data = []  # 重新建立
        for i, individual in enumerate(pop_genes):
            folder = os.path.join(save_root, f"P{i+1}")
            if build_results[i]:
                print(f"  評估初始模型 P{i+1}...")
                eval_data = evaluate_fitness(
                    folder,
                    individual,
                    return_uniformity=RETURN_UNIFORMITY,
                    eff_weight=EFF_WEIGHT,
                    process_weight=PROCESS_WEIGHT,
                    uni_weight=UNI_WEIGHT,
                )
            else:
                eval_data = (-999, 0, 0, 0.0, [], [0.0] * 8)  # 給予失敗個體極差的適應度

            parent_eval_data.append(eval_data)
            log_row = create_log_row(
                individual, pop_sigmas[i], eval_data, 1, "parent", (-1, -1)
            )
            initial_gen_log.append(log_row)

        max_fitness_gen1 = (
            max(d[0] for d in parent_eval_data) if parent_eval_data else -999
        )
        gen1_filename = f"fitness_gen1_max{max_fitness_gen1:.2f}.csv"
        save_generation_log(initial_gen_log, os.path.join(log_dir, gen1_filename))
        print(f"★ 第 1 代的親代已評估完成，日誌已存為 {gen1_filename}")

        start_gen = 1

    else:
        print(f"🔁 從日誌檔 {os.path.basename(last_gen_filepath)} 恢復進度。")
        print(f"🔁 將從第 {start_gen + 1} 代繼續執行至第 {N_GENERATIONS} 代")

        try:
            with open(last_gen_filepath, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                last_gen_parents_rows = [
                    row for row in reader if row.get("role") == "parent"
                ]
        except (IOError, csv.Error) as e:
            send_error(f"讀取日誌檔失敗", f"檔案: {last_gen_filepath}\n錯誤: {e}")
            return

        if len(last_gen_parents_rows) < POP_SIZE:
            send_error(
                f"親代紀錄不完整",
                f"檔案: {last_gen_filepath}\n預期 {POP_SIZE} 個, 實際 {len(last_gen_parents_rows)} 個。",
            )
            return

        pop_genes_list, pop_sigmas_list = [], []
        print("\n--- 正在從 Log 恢復上一代親代狀態 ---")
        for i, row in enumerate(last_gen_parents_rows):
            try:
                gene = [float(row["S1"]), float(row["S2"]), float(row["A1"])]
                pop_genes_list.append(gene)
                sigma = [
                    float(row["sigma1"]),
                    float(row["sigma2"]),
                    float(row["sigma3"]),
                ]
                pop_sigmas_list.append(sigma)
                fitness = float(row["fitness"])
                efficiency = float(row["efficiency"])
                process_score = float(row["process_score"])
                cv = float(row.get("cv", 0.0))
                uniformity = float(row.get("uniformity", 1.0 - cv))
                angle_effs = [
                    float(row.get(f"eff_{angle}", 0.0)) for angle in range(10, 90, 10)
                ]
                angle_unis = []
                for angle in range(10, 90, 10):
                    val = row.get(f"uni_{angle}")
                    if val not in (None, ""):
                        try:
                            angle_unis.append(float(val))
                            continue
                        except ValueError:
                            pass
                    cv_val = row.get(f"cv_{angle}")
                    if cv_val not in (None, ""):
                        try:
                            cv_a = float(cv_val)
                        except ValueError:
                            cv_a = 1.0
                    else:
                        cv_a = 1.0
                    angle_unis.append(max(0.0, 1.0 - cv_a))
                parent_eval_data.append(
                    (
                        fitness,
                        efficiency,
                        process_score,
                        uniformity,
                        angle_effs,
                        angle_unis,
                    )
                )
                print(f"  [DEBUG] 已恢復親代 {i}: Gene={gene}, Fitness={fitness:.4f}")
            except (ValueError, KeyError) as e:
                send_error(
                    "恢復親代數據失敗",
                    f"檔案: {last_gen_filepath}\n錯誤行: {row}\n錯誤: {e}",
                )
                return
        pop_genes = np.array(pop_genes_list, dtype=float)
        pop_sigmas = np.array(pop_sigmas_list, dtype=float)

    # --- 演化主迴圈 ---
    for g in range(start_gen, N_GENERATIONS):
        current_gen = g + 1
        print(f"\n{'='*25} GENERATION {current_gen} {'='*25}")

        current_gen_log = []

        # 為了檢查重複，需要載入所有歷史紀錄
        full_history_log = []
        try:
            log_files = [
                f
                for f in os.listdir(log_dir)
                if f.startswith("fitness_gen") and f.endswith(".csv")
            ]
            sorted_log_files = sorted(
                log_files, key=lambda x: int(re.search(r"gen(\d+)", x).group(1))
            )
            for log_file in sorted_log_files:
                with open(os.path.join(log_dir, log_file), "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    full_history_log.extend(list(reader))
        except Exception as e:
            print(f"⚠️  讀取歷史日誌時發生錯誤: {e}")

        children_genes, children_sigmas, children_parent_idxs = [], [], []
        for _ in range(OFFSPRING_SIZE):
            parent_idx = random.randint(0, POP_SIZE - 1)
            parent_gene, parent_sigma = (
                pop_genes[parent_idx].copy(),
                pop_sigmas[parent_idx].copy(),
            )
            new_sigma = parent_sigma * np.exp(
                TAU_PRIME * np.random.randn() + TAU * np.random.randn(n)
            )
            new_sigma *= SIGMA_SCALE
            new_sigma = np.maximum(new_sigma, 0.05)

            if random.random() < 0.6:
                new_sigma = np.array([0.2, 0.2, 5.0])
            child_gene = clamp_gene(parent_gene + new_sigma * np.random.randn(n))
            children_genes.append(child_gene)
            children_sigmas.append(new_sigma)
            children_parent_idxs.append((parent_idx, -1))

        offspring_eval_data = [None] * OFFSPRING_SIZE
        needs_processing_indices = []

        print("\n--- 步驟 1/4：檢查子代狀態 ---")
        for i, individual in enumerate(children_genes):
            is_evaluated, eval_data = check_if_evaluated(full_history_log, individual)
            if is_evaluated:
                print(f"  子代 P{i+1} 已在紀錄中，直接使用分數: {eval_data[0]:.4f}")
                offspring_eval_data[i] = eval_data
                log_row = create_log_row(
                    individual,
                    children_sigmas[i],
                    eval_data,
                    current_gen,
                    "offspring",
                    children_parent_idxs[i],
                )
                current_gen_log.append(log_row)
            else:
                needs_processing_indices.append(i)

        if needs_processing_indices:
            print(
                f"\n--- 步驟 2/4：建立 {len(needs_processing_indices)} 個新子代模型 ---"
            )
            for i in needs_processing_indices:
                folder = os.path.join(save_root, f"P{i+1}")
                os.makedirs(folder, exist_ok=True)
                print(f"  建立子代模型 P{i+1}...")
                build_model_with_retry(children_genes[i], folder)

            print(
                f"\n--- 步驟 3/4：執行 {len(needs_processing_indices)} 次新子代模擬 ---"
            )
            for i in needs_processing_indices:
                folder = os.path.join(save_root, f"P{i+1}")
                print(f"  模擬子代模型 P{i+1}...")
                simulate_and_evaluate(folder, children_genes[i])

            print(
                f"\n--- 步驟 4/4：計算 {len(needs_processing_indices)} 個新子代適應度 ---"
            )
            for i in needs_processing_indices:
                folder = os.path.join(save_root, f"P{i+1}")
                print(f"  評估子代模型 P{i+1}...")
                eval_data = evaluate_fitness(
                    folder,
                    children_genes[i],
                    return_uniformity=RETURN_UNIFORMITY,
                    eff_weight=EFF_WEIGHT,
                    process_weight=PROCESS_WEIGHT,
                    uni_weight=UNI_WEIGHT,
                )
                offspring_eval_data[i] = eval_data
                log_row = create_log_row(
                    children_genes[i],
                    children_sigmas[i],
                    eval_data,
                    current_gen,
                    "offspring",
                    children_parent_idxs[i],
                )
                current_gen_log.append(log_row)

        if any(data is None for data in offspring_eval_data):
            raise RuntimeError(
                f"嚴重錯誤：在第 {current_gen} 代，並非所有子代都成功獲得評估數據！"
            )

        print("\n--- 選擇下一代 ---")
        for i in range(POP_SIZE):
            log_row = create_log_row(
                pop_genes[i],
                pop_sigmas[i],
                parent_eval_data[i],
                current_gen,
                "parent_old",
                (-1, -1),
            )
            current_gen_log.append(log_row)

        combined_genes = np.vstack([pop_genes, children_genes])
        combined_sigmas = np.vstack([pop_sigmas, children_sigmas])
        combined_eval_data = parent_eval_data + offspring_eval_data
        objectives = np.array([evaluate_objectives(d) for d in combined_eval_data])
        combined_fitness = [d[0] for d in combined_eval_data]
        fronts = fast_non_dominated_sort(objectives)

        new_parents, new_sigmas, new_eval_data = [], [], []
        for front in fronts:
            if len(new_parents) + len(front) <= POP_SIZE:
                for idx in front:
                    new_parents.append(combined_genes[idx])
                    new_sigmas.append(combined_sigmas[idx])
                    new_eval_data.append(combined_eval_data[idx])
            else:
                remaining = POP_SIZE - len(new_parents)
                cd = crowding_distance(front, objectives)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                for idx in sorted_front[:remaining]:
                    new_parents.append(combined_genes[idx])
                    new_sigmas.append(combined_sigmas[idx])
                    new_eval_data.append(combined_eval_data[idx])
                break

        pareto_front = fronts[0] if fronts else []
        pareto_rows = []
        for idx in pareto_front:
            row = create_log_row(
                combined_genes[idx],
                combined_sigmas[idx],
                combined_eval_data[idx],
                current_gen,
                "pareto",
                (-1, -1),
            )
            pareto_rows.append(row)
        if pareto_rows:
            pareto_filename = f"pareto_gen{current_gen}.csv"
            save_generation_log(pareto_rows, os.path.join(log_dir, pareto_filename))

        pop_genes = np.array(new_parents, dtype=float)
        pop_sigmas = np.array(new_sigmas, dtype=float)
        parent_eval_data = new_eval_data

        print("\n--- 紀錄新一代親代 ---")
        for i in range(len(pop_genes)):
            log_row = create_log_row(
                pop_genes[i],
                pop_sigmas[i],
                parent_eval_data[i],
                current_gen,
                "parent",
                (-1, -1),
            )
            current_gen_log.append(log_row)

        best_fitness_this_gen = max(combined_fitness)
        output_filename = f"fitness_gen{current_gen}_max{best_fitness_this_gen:.2f}.csv"
        save_generation_log(current_gen_log, os.path.join(log_dir, output_filename))

        print(
            f"★ Generation {current_gen} 完成。本代最佳 Fitness: {best_fitness_this_gen:.4f}"
        )
        print(f"★ 日誌已存為: {output_filename}")
        gc.collect()

    print("\n🎉 所有世代執行完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_subject = f"演化策略主程式發生致命錯誤"
        error_body = f"錯誤類型: {type(e).__name__}\n錯誤訊息: {e}\n\n追蹤訊息:\n{traceback.format_exc()}"
        print(f"❌ {error_subject}")
        send_error(error_subject, error_body)
