# ES2.py — Evolution Strategy (μ+λ) with pure log-normal σ-adaptation, recombination, and robust I/O
# 作者：你我共同整理
# 亮點：
# - σ' = σ * exp(τ'·N(0,1) + τ·N_i(0,1))；無固定覆寫、無固定縮放，僅設每維下限
# - 2 親 intermediate recombination（基因與 σ 都重組）
# - 反射邊界（避免吸邊），內部計算保持連續；僅在寫 CSV 時格式化
# - 避免重複評估：容差比對歷史；只對「新個體」建模+模擬+評估一次
# - 修正子代資料夾索引：P{POP_SIZE+i+1}，不覆寫親代資料
# - (μ+λ) 選擇，並記錄 parent_old / offspring / parent 三類行的日誌

import os
import re
import gc
import csv
import time
import math
import shutil
import random
import traceback
from datetime import datetime
import numpy as np

# === 你的外部模組（可用假模組） ===
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_ES import evaluate_fitness

# pywinauto 不是必需；若無則忽略
try:
    from pywinauto import application, findwindows  # noqa: F401
except Exception:
    pass

# === ES 參數 ===
POP_SIZE = 10  # μ 親代
OFFSPRING_SIZE = POP_SIZE * 7  # λ 子代
N_GENERATIONS = 100

# --- 設計變數邊界 ---
SIDE_BOUND = [0.3, 0.9]  # S1, S2
ANGLE_BOUND = [30.0, 150.0]  # A1（度）
n = 3

# --- log-normal 自適應係數（ES 標準設定）---
TAU_PRIME = 1 / np.sqrt(2 * n)
TAU = 1 / np.sqrt(2 * np.sqrt(n))

# --- 每維 σ 的最小值：設為各維範圍的 1% ---
VAR_RANGES = np.array(
    [
        SIDE_BOUND[1] - SIDE_BOUND[0],
        SIDE_BOUND[1] - SIDE_BOUND[0],
        ANGLE_BOUND[1] - ANGLE_BOUND[0],
    ],
    dtype=float,
)
SIGMA_MIN = VAR_RANGES * 0.01

# --- 隨機性 ---
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# --- 路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_root = os.path.join(BASE_DIR, "GA_population")
# 若你要沿用 OneDrive，改成下行並註解上一行
log_dir = r"C:\Users\user\OneDrive - NTHU\home"
# log_dir = os.path.join(BASE_DIR, "es_run")
os.makedirs(save_root, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ---------- 工具 ----------


def write_run_config():
    cfg = {
        "POP_SIZE": POP_SIZE,
        "OFFSPRING_SIZE": OFFSPRING_SIZE,
        "N_GENERATIONS": N_GENERATIONS,
        "SIDE_BOUND": SIDE_BOUND,
        "ANGLE_BOUND": ANGLE_BOUND,
        "TAU_PRIME": TAU_PRIME,
        "TAU": TAU,
        "GLOBAL_SEED": GLOBAL_SEED,
        "SIGMA_MIN": SIGMA_MIN.tolist(),
        "save_root": save_root,
        "log_dir": log_dir,
    }
    try:
        p = os.path.join(log_dir, "run_config.txt")
        with open(p, "w", encoding="utf-8") as f:
            for k, v in cfg.items():
                f.write(f"{k} = {v}\n")
        print(f"🔧 執行設定已輸出到 {p}")
    except Exception as e:
        print(f"⚠️ 無法寫入 run_config.txt: {e}")


def send_error(subject: str, body: str):
    try:
        err_dir = os.path.join(log_dir, "ES_ErrorLogs")
        os.makedirs(err_dir, exist_ok=True)
        p = os.path.join(err_dir, "es_error.log")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️ 錯誤日誌已存到：{p}")
    except Exception:
        fallback = os.path.join(os.getcwd(), "es_error_fallback.log")
        with open(fallback, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {subject}\n{body}\n{'='*60}\n")
        print(f"⚠️ OneDrive 寫入失敗，錯誤日誌已寫入本地：{fallback}")


def reflect_bounds(x):
    """反射邊界；保持連續，不做四捨五入。"""
    lo = np.array([SIDE_BOUND[0], SIDE_BOUND[0], ANGLE_BOUND[0]], dtype=float)
    hi = np.array([SIDE_BOUND[1], SIDE_BOUND[1], ANGLE_BOUND[1]], dtype=float)
    y = x.copy()
    for k in range(3):
        if y[k] < lo[k]:
            span = hi[k] - lo[k]
            over = (lo[k] - y[k]) % (2 * span)
            y[k] = lo[k] + (over if over <= span else 2 * span - over)
        elif y[k] > hi[k]:
            span = hi[k] - lo[k]
            over = (y[k] - hi[k]) % (2 * span)
            y[k] = hi[k] - (over if over <= span else 2 * span - over)
    return y


def format_for_log(individual):
    """只在寫 CSV 時做格式化（S 保留 2 位，角度四捨五入為整數字串）。"""
    return {
        "S1": f"{individual[0]:.2f}",
        "S2": f"{individual[1]:.2f}",
        "A1": f"{int(round(individual[2]))}",
    }


def save_generation_log(rows, file_path):
    if not rows:
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
        + [f"eff_{a}" for a in range(10, 90, 10)]
        + [f"uni_{a}" for a in range(10, 90, 10)]
        + ["random_seed"]
    )
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def create_log_row(
    individual, sigma, fitness_data, generation, role, parent_indices, seed=None
):
    # fitness_data = (fitness, efficiency, process_score, uniformity, angle_effs, angle_unis)
    fitness, efficiency, process_score = (
        fitness_data[0],
        fitness_data[1],
        fitness_data[2],
    )
    uniformity = fitness_data[3] if len(fitness_data) >= 4 else 0.0
    angle_effs = fitness_data[4] if len(fitness_data) >= 5 else []
    angle_unis = fitness_data[5] if len(fitness_data) >= 6 else []
    fmt = format_for_log(individual)
    row = {
        "generation": generation,
        "role": role,
        "parent_idx1": parent_indices[0],
        "parent_idx2": parent_indices[1],
        "S1": fmt["S1"],
        "S2": fmt["S2"],
        "A1": fmt["A1"],
        "sigma1": f"{sigma[0]:.6f}",
        "sigma2": f"{sigma[1]:.6f}",
        "sigma3": f"{sigma[2]:.6f}",
        "fitness": f"{fitness:.6f}",
        "efficiency": f"{efficiency:.6f}",
        "process_score": f"{process_score:.6f}",
        "uniformity": f"{uniformity:.6f}",
        "random_seed": seed if seed is not None else GLOBAL_SEED,
    }
    for a, eff in zip(range(10, 90, 10), angle_effs):
        row[f"eff_{a}"] = f"{eff:.6f}"
    for a, uni in zip(range(10, 90, 10), angle_unis):
        row[f"uni_{a}"] = f"{uni:.6f}"
    return row


def find_last_completed_generation(directory):
    if not os.path.exists(directory):
        return 0, None
    pat = re.compile(r"fitness_gen(\d+)_max.*\.csv")
    last_gen, last_path = 0, None
    for fn in os.listdir(directory):
        m = pat.match(fn)
        if m:
            g = int(m.group(1))
            if g > last_gen:
                last_gen, last_path = g, os.path.join(directory, fn)
    return last_gen, last_path


def is_duplicate_history(
    full_history_rows, individual, tol=np.array([1e-3, 1e-3, 5e-2])
):
    """容差比對是否曾評估過；回傳 (bool, fitness_tuple or None)。"""
    x = np.array(individual, dtype=float)
    for row in reversed(full_history_rows):
        try:
            y = np.array(
                [
                    float(row.get("S1", "nan")),
                    float(row.get("S2", "nan")),
                    float(row.get("A1", "nan")),
                ],
                dtype=float,
            )
        except Exception:
            continue
        if np.all(np.abs(x - y) <= tol):
            fitness = float(row["fitness"])
            efficiency = float(row["efficiency"])
            process_score = float(row["process_score"])
            uniformity = float(row.get("uniformity", 0.0))
            angle_effs = [
                float(row.get(f"eff_{ang}", 0.0)) for ang in range(10, 90, 10)
            ]
            angle_unis = [
                float(row.get(f"uni_{ang}", 0.0)) for ang in range(10, 90, 10)
            ]
            return True, (
                fitness,
                efficiency,
                process_score,
                uniformity,
                angle_effs,
                angle_unis,
            )
    return False, None


def build_model_with_retry(individual, folder, max_attempts=3):
    for k in range(max_attempts):
        try:
            result, logs = Build_model(
                individual,
                mode="triangle",
                folder=folder,
                fillet=1,
                radius_vertex=0.051,
                radius_inside=0.063,
                light_source_length=0.5,
            )
            for msg in logs:
                print(msg)
            if result == 1:
                return True
        except Exception as e:
            print(f"❌ Build_model 第 {k+1} 次失敗：{e}")
            time.sleep(1)
    return False


def simulate_and_evaluate(folder, individual):
    """一次完成：呼叫 TracePro，接著 evaluate_fitness，回傳 fitness 資料 tuple。"""
    while True:
        try:
            tracepro_fast(os.path.join(folder, "Sim.scm"))
            return evaluate_fitness(
                folder,
                individual,
                return_uniformity=False,
                eff_weight=1,
                process_weight=1,
                uni_weight=1,
            )
        except Exception as e:
            print(f"⚠️ tracepro/evaluate_fitness 失敗，重試：{e}")
            time.sleep(1)


# ---------- 初始化 ----------


def copy_scm_to_all_folders():
    macro_dir = os.path.join(BASE_DIR, "Macro")
    scm_file = os.path.join(macro_dir, "Sim.scm")
    print(f"複製 SCM 檔案: {scm_file}")
    # 親代 P1..Pμ
    for i in range(1, POP_SIZE + 1):
        folder = os.path.join(save_root, f"P{i}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(scm_file, folder)
    # 子代 P(μ+1)..P(μ+λ)
    for i in range(POP_SIZE + 1, POP_SIZE + OFFSPRING_SIZE + 1):
        folder = os.path.join(save_root, f"P{i}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(scm_file, folder)


def init_population():
    pop = np.zeros((POP_SIZE, n), dtype=float)
    pop[:, 0] = np.random.uniform(SIDE_BOUND[0], SIDE_BOUND[1], size=POP_SIZE)
    pop[:, 1] = np.random.uniform(SIDE_BOUND[0], SIDE_BOUND[1], size=POP_SIZE)
    pop[:, 2] = np.random.uniform(ANGLE_BOUND[0], ANGLE_BOUND[1], size=POP_SIZE)
    for i in range(POP_SIZE):
        pop[i] = reflect_bounds(pop[i])
    sigma0 = np.tile((VAR_RANGES * 0.10), (POP_SIZE, 1))  # 初始 10%
    return pop, sigma0


# ---------- 產生子代：雙親重組 + 純自適應 σ ----------


def make_offspring(pop_genes, pop_sigmas):
    children_genes, children_sigmas, parent_pairs = [], [], []
    for _ in range(OFFSPRING_SIZE):
        p1, p2 = random.sample(range(POP_SIZE), 2)
        x1, x2 = pop_genes[p1], pop_genes[p2]
        s1, s2 = pop_sigmas[p1], pop_sigmas[p2]

        # intermediate recombination（基因與 σ）
        alpha = np.random.rand()
        x_bar = alpha * x1 + (1 - alpha) * x2
        s_bar = alpha * s1 + (1 - alpha) * s2

        # 純 log-normal 自適應（無固定覆寫、無額外縮放）
        global_noise = np.random.randn()
        indiv_noise = np.random.randn(n)
        new_sigma = s_bar * np.exp(TAU_PRIME * global_noise + TAU * indiv_noise)
        new_sigma = np.maximum(new_sigma, SIGMA_MIN)

        # 產生子代 & 邊界反射
        child = x_bar + new_sigma * np.random.randn(n)
        child = reflect_bounds(child)

        children_genes.append(child)
        children_sigmas.append(new_sigma)
        parent_pairs.append((p1, p2))
    return children_genes, children_sigmas, parent_pairs


# ---------- 主程式 ----------


def main():
    copy_scm_to_all_folders()
    write_run_config()

    start_gen, last_path = find_last_completed_generation(log_dir)
    parent_eval = []

    if start_gen == 0:
        print("🌱 無日誌，從第 1 代開始")
        pop_genes, pop_sigmas = init_population()

        # 建模 + 模擬 + 評估（一次）
        initial_rows = []
        for i in range(POP_SIZE):
            folder = os.path.join(save_root, f"P{i+1}")
            os.makedirs(folder, exist_ok=True)
            print(f"  建立初始模型 P{i+1}...")
            ok = build_model_with_retry(pop_genes[i], folder)
            if ok:
                eval_data = simulate_and_evaluate(folder, pop_genes[i])
            else:
                eval_data = (-999, 0, 0, 0.0, [], [])
            parent_eval.append(eval_data)
            initial_rows.append(
                create_log_row(
                    pop_genes[i], pop_sigmas[i], eval_data, 1, "parent", (-1, -1)
                )
            )

        best = max(d[0] for d in parent_eval)
        fn = f"fitness_gen1_max{best:.2f}.csv"
        save_generation_log(initial_rows, os.path.join(log_dir, fn))
        print(f"★ 第 1 代完成，存為 {fn}")
        start_gen = 1
    else:
        print(
            f"🔁 從 {os.path.basename(last_path)} 恢復，將續跑至第 {N_GENERATIONS} 代"
        )
        with open(last_path, "r", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("role") == "parent"]
        pop_genes, pop_sigmas = [], []
        for r in rows:
            pop_genes.append([float(r["S1"]), float(r["S2"]), float(r["A1"])])
            pop_sigmas.append(
                [float(r["sigma1"]), float(r["sigma2"]), float(r["sigma3"])]
            )
            parent_eval.append(
                (
                    float(r["fitness"]),
                    float(r["efficiency"]),
                    float(r["process_score"]),
                    float(r.get("uniformity", 0.0)),
                    [float(r.get(f"eff_{a}", 0.0)) for a in range(10, 90, 10)],
                    [float(r.get(f"uni_{a}", 0.0)) for a in range(10, 90, 10)],
                )
            )
        pop_genes = np.array(pop_genes, dtype=float)
        pop_sigmas = np.array(pop_sigmas, dtype=float)

    # === 演化 ===
    for g in range(start_gen, N_GENERATIONS):
        gen = g + 1
        print(f"\n{'='*18} GENERATION {gen} {'='*18}")

        # 載入歷史日誌做重複判定
        history_rows = []
        try:
            files = [
                f
                for f in os.listdir(log_dir)
                if f.startswith("fitness_gen") and f.endswith(".csv")
            ]
            files.sort(key=lambda x: int(re.search(r"gen(\d+)", x).group(1)))
            for f in files:
                with open(os.path.join(log_dir, f), "r", encoding="utf-8") as fh:
                    history_rows.extend(list(csv.DictReader(fh)))
        except Exception as e:
            print(f"⚠️ 讀歷史日誌失敗：{e}")

        # 產生子代
        children_genes, children_sigmas, parent_pairs = make_offspring(
            pop_genes, pop_sigmas
        )

        # 檢查重複；需要新跑的才建模+模擬+評估
        offspring_eval = [None] * OFFSPRING_SIZE
        current_rows = []
        need_build = []

        for i, child in enumerate(children_genes):
            dup, data = is_duplicate_history(history_rows, child)
            if dup:
                offspring_eval[i] = data
                current_rows.append(
                    create_log_row(
                        child,
                        children_sigmas[i],
                        data,
                        gen,
                        "offspring",
                        parent_pairs[i],
                    )
                )
            else:
                need_build.append(i)

        # 建模
        for i in need_build:
            folder = os.path.join(save_root, f"P{POP_SIZE + i + 1}")  # 不覆寫親代
            os.makedirs(folder, exist_ok=True)
            print(f"  建立子代模型 P{POP_SIZE + i + 1}...")
            build_model_with_retry(children_genes[i], folder)

        # 模擬 + 評估
        for i in need_build:
            folder = os.path.join(save_root, f"P{POP_SIZE + i + 1}")
            print(f"  模擬/評估 子代模型 P{POP_SIZE + i + 1}...")
            eval_data = simulate_and_evaluate(folder, children_genes[i])
            offspring_eval[i] = eval_data
            current_rows.append(
                create_log_row(
                    children_genes[i],
                    children_sigmas[i],
                    eval_data,
                    gen,
                    "offspring",
                    parent_pairs[i],
                )
            )

        if any(v is None for v in offspring_eval):
            raise RuntimeError(f"第 {gen} 代仍有子代未得到評估結果")

        # 把上一代親代也寫入（parent_old）
        for i in range(POP_SIZE):
            current_rows.append(
                create_log_row(
                    pop_genes[i],
                    pop_sigmas[i],
                    parent_eval[i],
                    gen,
                    "parent_old",
                    (-1, -1),
                )
            )

        # (μ+λ) 選擇
        combined_genes = np.vstack([pop_genes, children_genes])
        combined_sigmas = np.vstack([pop_sigmas, children_sigmas])
        combined_eval = parent_eval + offspring_eval
        fitness_all = [d[0] for d in combined_eval]
        order = np.argsort(fitness_all)[::-1]  # 最大化

        new_genes, new_sigmas, new_eval = [], [], []
        for idx in order:
            if len(new_genes) >= POP_SIZE:
                break
            new_genes.append(combined_genes[idx])
            new_sigmas.append(combined_sigmas[idx])
            new_eval.append(combined_eval[idx])

        pop_genes = np.array(new_genes, dtype=float)
        pop_sigmas = np.array(new_sigmas, dtype=float)
        parent_eval = new_eval

        # 記錄新一代親代
        for i in range(POP_SIZE):
            current_rows.append(
                create_log_row(
                    pop_genes[i], pop_sigmas[i], parent_eval[i], gen, "parent", (-1, -1)
                )
            )

        best = max(fitness_all)
        out = f"fitness_gen{gen}_max{best:.2f}.csv"
        save_generation_log(current_rows, os.path.join(log_dir, out))
        print(f"★ Generation {gen} 完成，最佳 fitness = {best:.4f}，日誌：{out}")
        gc.collect()

    print("\n🎉 所有世代執行完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        subject = "演化策略主程式發生致命錯誤"
        body = f"錯誤類型: {type(e).__name__}\n錯誤訊息: {e}\n\n追蹤訊息:\n{traceback.format_exc()}"
        print(f"❌ {subject}")
        send_error(subject, body)
