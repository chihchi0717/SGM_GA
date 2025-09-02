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
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import HuberRegressor

# === 你的外部模組（可用假模組） ===
from draw_New import draw_
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_ES import evaluate_fitness

# pywinauto 不是必需；若無則忽略
try:
    from pywinauto import application, findwindows
except Exception:
    pass

# ==============================================================================
# === *** 核心模型定義 (整合自 compensation_models.py) *** ===
# ==============================================================================

FEATURES = [
    "Design_s1(mm)",
    "Design_s2(mm)",
    "Design_s3(mm)",
    "Design_a3(deg)",
    "Design_a1(deg)",
    "Design_a2(deg)",
]


def _augment_features(X_raw: np.ndarray) -> np.ndarray:
    """建立所有二階多項式特徵 (包含平方項與交互作用項)"""
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    return poly.fit_transform(X_raw)


class ModelHuber:
    """一個通用的 Huber 迴歸模型類別，封裝了特徵工程和縮放。"""

    def __init__(self, scale: bool, alpha: float, epsilon: float, max_iter: int):
        self.scaler: Optional[StandardScaler] = None
        self.scale = scale
        self.model = HuberRegressor(
            alpha=alpha, epsilon=epsilon, max_iter=int(max_iter)
        )

    def _augment(self, X_raw: np.ndarray) -> np.ndarray:
        return _augment_features(X_raw)

    def fit(self, df: pd.DataFrame, target: str):
        """訓練模型，並設定內部的特徵縮放器。"""
        d = df[FEATURES + [target]].dropna().copy()
        X_raw = d[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw)

        if self.scale:
            self.scaler = StandardScaler()
            X_aug = self.scaler.fit_transform(X_aug)

        self.model.fit(X_aug, d[target].to_numpy(dtype=float))

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """使用訓練好的模型和縮放器進行預測。"""
        X_raw = df_features[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw)
        if self.scale and self.scaler:
            X_aug = self.scaler.transform(X_aug)
        return self.model.predict(X_aug)


# ==============================================================================
# === 整合結束 ===
# ==============================================================================


# === 路徑設定 ===
# *** 修正 ***: 更新為您提供的 Excel 檔案路徑
TRAIN_DATA_PATH = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\GA\SGM_GA\printing\regression\analysis_results0831.xlsx"

# --- 全域模型物件 ---
model_s2: Optional[ModelHuber] = None
model_s3: Optional[ModelHuber] = None
model_ang: Optional[ModelHuber] = None


# === ES 參數 ===
POP_SIZE = 10
OFFSPRING_SIZE = POP_SIZE * 7
N_GENERATIONS = 100
SIDE_BOUND = [0.3, 0.9]
ANGLE_BOUND = [30.0, 150.0]
n = 3
TAU_PRIME = 1 / np.sqrt(2 * n)
TAU = 1 / np.sqrt(2 * np.sqrt(n))
VAR_RANGES = np.array(
    [
        SIDE_BOUND[1] - SIDE_BOUND[0],
        SIDE_BOUND[1] - SIDE_BOUND[0],
        ANGLE_BOUND[1] - ANGLE_BOUND[0],
    ],
    dtype=float,
)
SIGMA_MIN = VAR_RANGES * 0.01
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_root = os.path.join(BASE_DIR, "GA_population")
log_dir = r"C:\Users\cchih\OneDrive - NTHU\msi"
os.makedirs(save_root, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# ---------- 工具函式 ----------


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


def predict_shrinkage_and_angle(design_params: Dict) -> Tuple[float, float, float]:
    """
    使用全域的模型物件進行預測，此方法會自動處理特徵工程和縮放。
    """
    if model_s2 is None or model_s3 is None or model_ang is None:
        raise RuntimeError("模型尚未初始化。")

    df_input = pd.DataFrame([design_params])
    df_features = df_input.rename(
        columns={
            "s1": "Design_s1(mm)",
            "s2": "Design_s2(mm)",
            "s3": "Design_s3(mm)",
            "a1": "Design_a1(deg)",
            "a2": "Design_a2(deg)",
            "a3": "Design_a3(deg)",
        }
    )[FEATURES]

    pred_s2 = model_s2.predict(df_features)[0]
    pred_s3 = model_s3.predict(df_features)[0]
    pred_a3 = model_ang.predict(df_features)[0]

    return pred_s2, pred_s3, pred_a3


def calculate_final_geometry_from_predictions(s2_pred, s3_pred, a3_pred_deg):
    a3_pred_rad = math.radians(a3_pred_deg)
    s3_sin_a3 = s3_pred * math.sin(a3_pred_rad)
    s3_cos_a3 = s3_pred * math.cos(a3_pred_rad)

    denominator = s2_pred - s3_cos_a3
    if abs(denominator) < 1e-9:
        print("️⚠️ 幾何無解，tan(a1) 計算分母為零。")
        return None, None

    a1_pred_rad = math.atan2(s3_sin_a3, denominator)
    a1_pred_deg = math.degrees(a1_pred_rad)

    if a1_pred_deg <= 0 or a1_pred_deg >= 180:
        print(f"️⚠️ 幾何無解，計算出的 a1_pred ({a1_pred_deg:.2f}) 超出範圍。")
        return None, None

    sin_a1 = math.sin(a1_pred_rad)
    if abs(sin_a1) < 1e-9:
        print("️⚠️ 幾何無解，sin(a1) 為零。")
        return None, None

    s1_pred = s3_pred * math.sin(a3_pred_rad) / sin_a1

    return s1_pred, a1_pred_deg


def calculate_dependent_variables(individual):
    s1, s2, a1_deg = individual[0], individual[1], individual[2]
    a1_rad = math.radians(a1_deg)
    s3 = math.sqrt(pow(s1, 2) + pow(s2, 2) - 2 * s1 * s2 * math.cos(a1_rad))

    if abs(s1) < 1e-9 or abs(s3) < 1e-9:
        return {"s1": s1, "s2": s2, "s3": s3, "a1": a1_deg, "is_valid": False}

    cos_a2_arg = (pow(s1, 2) + pow(s3, 2) - pow(s2, 2)) / (2 * s1 * s3)
    cos_a2_arg = max(-1.0, min(1.0, cos_a2_arg))
    a2_rad = math.acos(cos_a2_arg)
    a2_deg = math.degrees(a2_rad)
    a3_deg = 180.0 - a1_deg - a2_deg
    design_params = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "a1": a1_deg,
        "a2": a2_deg,
        "a3": a3_deg,
    }

    s2_shrink_rate, s3_shrink_rate, a3_pred = predict_shrinkage_and_angle(design_params)
    s2_pred = s2 * (1 - s2_shrink_rate)
    s3_pred = s3 * (1 - s3_shrink_rate)

    if s2_pred <= 0 or s3_pred <= 0:
        print(
            f"️⚠️ 物理無解：預測長度為負或零 (s2_pred: {s2_pred:.2f}, s3_pred: {s3_pred:.2f})。"
        )
        design_params.update(
            {
                "is_valid": False,
                "s2_predicted": s2_pred,
                "s3_predicted": s3_pred,
                "a3_predicted": a3_pred,
            }
        )
        return design_params

    design_params.update(
        {"s2_predicted": s2_pred, "s3_predicted": s3_pred, "a3_predicted": a3_pred}
    )
    s1_pred, a1_pred = calculate_final_geometry_from_predictions(
        s2_pred, s3_pred, a3_pred
    )

    if s1_pred is None:
        design_params["is_valid"] = False
    else:
        design_params.update(
            {"is_valid": True, "s1_predicted": s1_pred, "a1_predicted": a1_pred}
        )
    return design_params


def evaluate_individual(individual: np.ndarray, folder: str) -> Tuple[Tuple, Dict]:
    penalty_fitness = (-9999.0, 0, 0, [], [])

    design_params = calculate_dependent_variables(individual)

    if not design_params.get("is_valid", False):
        print(f"  -> 幾何/物理無解，指定懲罰 fitness。")
        return penalty_fitness, design_params

    params_for_build = [
        float(design_params["s1_predicted"]),
        float(design_params["s2_predicted"]),
        float(design_params["a1_predicted"]),
    ]

    build_success = False
    for attempt in range(3):
        try:
            result, logs = Build_model(
                params_for_build,
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
                build_success = True
                break
        except Exception as e:
            print(f"❌ Build_model 第 {attempt+1} 次失敗：{e}")
            time.sleep(1)

    if not build_success:
        print(f"  -> 建模失敗，指定懲罰 fitness。")
        return penalty_fitness, design_params

    print(f"  -> 模型建立成功，進行光學模擬/評估...")
    try:
        tracepro_fast(os.path.join(folder, "Sim.scm"))
        fitness_data = evaluate_fitness(
            folder,
            individual,
            return_uniformity=False,
            eff_weight=1,
            process_weight=1,
            uni_weight=1,
        )
        return fitness_data, design_params
    except Exception as e:
        print(f"⚠️ tracepro/evaluate_fitness 失敗，重試：{e}")
        return penalty_fitness, design_params


def reflect_bounds(x):
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
            "S3_calc",
            "A2_calc",
            "A3_calc",
            "s1_pred",
            "s2_pred",
            "s3_pred",
            "a1_pred",
            "a3_pred",
            "sigma1",
            "sigma2",
            "sigma3",
            "fitness",
            "efficiency",
            "process_score",
            "uniformity",
            "is_valid",
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
    individual,
    sigma,
    fitness_data,
    generation,
    role,
    parent_indices,
    design_params=None,
    seed=None,
):
    if design_params is None:
        design_params = calculate_dependent_variables(individual)

    fit, eff, proc = fitness_data[0], fitness_data[1], fitness_data[2]
    uni = 0.0  # return_uniformity=False
    angle_effs = fitness_data[3] if len(fitness_data) >= 4 else []
    angle_unis = fitness_data[4] if len(fitness_data) >= 5 else []

    fmt = format_for_log(individual)
    row = {
        "generation": generation,
        "role": role,
        "parent_idx1": parent_indices[0],
        "parent_idx2": parent_indices[1],
        "S1": fmt["S1"],
        "S2": fmt["S2"],
        "A1": fmt["A1"],
        "S3_calc": f"{design_params.get('s3', -1):.3f}",
        "A2_calc": f"{design_params.get('a2', -1):.2f}",
        "A3_calc": f"{design_params.get('a3', -1):.2f}",
        "s1_pred": f"{design_params.get('s1_predicted', -1):.3f}",
        "s2_pred": f"{design_params.get('s2_predicted', -1):.3f}",
        "s3_pred": f"{design_params.get('s3_predicted', -1):.3f}",
        "a1_pred": f"{design_params.get('a1_predicted', -1):.2f}",
        "a3_pred": f"{design_params.get('a3_predicted', -1):.2f}",
        "sigma1": f"{sigma[0]:.6f}",
        "sigma2": f"{sigma[1]:.6f}",
        "sigma3": f"{sigma[2]:.6f}",
        "fitness": f"{fit:.6f}",
        "efficiency": f"{eff:.6f}",
        "process_score": f"{proc:.6f}",
        "uniformity": f"{uni:.6f}",
        "is_valid": design_params.get("is_valid", False),
        "random_seed": seed if seed is not None else GLOBAL_SEED,
    }
    for a, e in zip(range(10, 90, 10), angle_effs):
        row[f"eff_{a}"] = f"{e:.6f}"
    for a, u in zip(range(10, 90, 10), angle_unis):
        row[f"uni_{a}"] = f"{u:.6f}"
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
            # *** 修正 ***: 安全地將可能為空字串的欄位轉換為 float
            fitness = float(row.get("fitness") or 0.0)
            efficiency = float(row.get("efficiency") or 0.0)
            process_score = float(row.get("process_score") or 0.0)
            uniformity = float(row.get("uniformity") or 0.0)
            angle_effs = [float(row.get(f"eff_{a}") or 0.0) for a in range(10, 90, 10)]
            angle_unis = [float(row.get(f"uni_{a}") or 0.0) for a in range(10, 90, 10)]

            return True, (fitness, efficiency, process_score, angle_effs, angle_unis)
    return False, None


def copy_scm_to_all_folders():
    macro_dir = os.path.join(BASE_DIR, "Macro")
    scm_file = os.path.join(macro_dir, "Sim.scm")
    print(f"複製 SCM 檔案: {scm_file}")
    for i in range(1, POP_SIZE + OFFSPRING_SIZE + 1):
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
    sigma0 = np.tile((VAR_RANGES * 0.10), (POP_SIZE, 1))
    return pop, sigma0


def make_offspring(pop_genes, pop_sigmas):
    children_genes, children_sigmas, parent_pairs = [], [], []
    for _ in range(OFFSPRING_SIZE):
        p1, p2 = random.sample(range(POP_SIZE), 2)
        x1, x2, s1, s2 = pop_genes[p1], pop_genes[p2], pop_sigmas[p1], pop_sigmas[p2]
        alpha = np.random.rand()
        x_bar, s_bar = alpha * x1 + (1 - alpha) * x2, alpha * s1 + (1 - alpha) * s2
        global_noise, indiv_noise = np.random.randn(), np.random.randn(n)
        new_sigma = s_bar * np.exp(TAU_PRIME * global_noise + TAU * indiv_noise)
        new_sigma = np.maximum(new_sigma, SIGMA_MIN)
        child = reflect_bounds(x_bar + new_sigma * np.random.randn(n))
        children_genes.append(child)
        children_sigmas.append(new_sigma)
        parent_pairs.append((p1, p2))
    return children_genes, children_sigmas, parent_pairs


def main():
    global model_s2, model_s3, model_ang

    print("\n--- 初始化並訓練補償模型 ---")
    model_s2 = ModelHuber(scale=True, alpha=1.0, epsilon=1000, max_iter=10000)
    model_s3 = ModelHuber(scale=True, alpha=1.0, epsilon=1000, max_iter=10000)
    model_ang = ModelHuber(scale=True, alpha=0.1, epsilon=1000, max_iter=10000)

    try:
        df_train = pd.read_excel(TRAIN_DATA_PATH)
        print(f"✅ 成功讀取訓練資料 '{TRAIN_DATA_PATH}'。")

        print("🏋️ 正在訓練 delta_s2 模型...")
        model_s2.fit(df_train, target="delta_s2")

        print("🏋️ 正在訓練 delta_s3 模型...")
        model_s3.fit(df_train, target="delta_s3")

        print("🏋️ 正在訓練 DIP_a3(deg) 模型...")
        model_ang.fit(df_train, target="DIP_a3(deg)")

        print("✅ 所有模型訓練完成。")
    except FileNotFoundError:
        print(f"❌ 嚴重錯誤: 找不到訓練資料 '{TRAIN_DATA_PATH}'。無法訓練模型。")
        return
    except Exception as e:
        print(f"❌ 讀取或訓練時發生錯誤: {e}")
        return
    print("---------------------------\n")

    copy_scm_to_all_folders()
    write_run_config()
    start_gen, last_path = find_last_completed_generation(log_dir)
    parent_eval = []

    if start_gen == 0:
        print("🌱 無日誌，從第 1 代開始")
        pop_genes, pop_sigmas = init_population()
        initial_rows = []
        for i in range(POP_SIZE):
            folder = os.path.join(save_root, f"P{i+1}")
            print(f"  處理初始個體 P{i+1}...")
            eval_data, design_params = evaluate_individual(pop_genes[i], folder)
            parent_eval.append(eval_data)
            initial_rows.append(
                create_log_row(
                    pop_genes[i],
                    pop_sigmas[i],
                    eval_data,
                    1,
                    "parent",
                    (-1, -1),
                    design_params=design_params,
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
        pop_genes = np.array(
            [[float(r["S1"]), float(r["S2"]), float(r["A1"])] for r in rows]
        )
        pop_sigmas = np.array(
            [[float(r["sigma1"]), float(r["sigma2"]), float(r["sigma3"])] for r in rows]
        )
        # *** 修正 ***: 這裡的解析也需要與 create_log_row 的新結構匹配
        parent_eval_from_log = []
        for r in rows:
            fit = float(r.get("fitness") or 0.0)
            eff = float(r.get("efficiency") or 0.0)
            proc = float(r.get("process_score") or 0.0)
            angle_effs = [float(r.get(f"eff_{a}") or 0.0) for a in range(10, 90, 10)]
            angle_unis = [float(r.get(f"uni_{a}") or 0.0) for a in range(10, 90, 10)]
            parent_eval_from_log.append((fit, eff, proc, angle_effs, angle_unis))
        parent_eval = parent_eval_from_log

    for g in range(start_gen, N_GENERATIONS):
        gen = g + 1
        print(f"\n{'='*18} GENERATION {gen} {'='*18}")
        history_rows = []
        try:
            files = sorted(
                [
                    f
                    for f in os.listdir(log_dir)
                    if f.startswith("fitness_gen") and f.endswith(".csv")
                ],
                key=lambda x: int(re.search(r"gen(\d+)", x).group(1)),
            )
            for f in files:
                with open(os.path.join(log_dir, f), "r", encoding="utf-8") as fh:
                    history_rows.extend(list(csv.DictReader(fh)))
        except Exception as e:
            print(f"⚠️ 讀歷史日誌失敗：{e}")

        children_genes, children_sigmas, parent_pairs = make_offspring(
            pop_genes, pop_sigmas
        )
        offspring_eval, current_rows, need_eval = [None] * OFFSPRING_SIZE, [], []

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
                need_eval.append(i)

        for i in need_eval:
            folder = os.path.join(save_root, f"P{POP_SIZE + i + 1}")
            print(f"  處理子代個體 P{POP_SIZE + i + 1}...")
            eval_data, design_params = evaluate_individual(children_genes[i], folder)
            offspring_eval[i] = eval_data
            current_rows.append(
                create_log_row(
                    children_genes[i],
                    children_sigmas[i],
                    eval_data,
                    gen,
                    "offspring",
                    parent_pairs[i],
                    design_params=design_params,
                )
            )

        if any(v is None for v in offspring_eval):
            raise RuntimeError(f"第 {gen} 代仍有子代未得到評估結果")

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

        combined_genes = np.vstack([pop_genes, children_genes])
        combined_sigmas = np.vstack([pop_sigmas, children_sigmas])
        combined_eval = parent_eval + offspring_eval
        fitness_all = [d[0] for d in combined_eval]
        order = np.argsort(fitness_all)[::-1]

        new_genes, new_sigmas, new_eval = [], [], []
        for idx in order[:POP_SIZE]:
            new_genes.append(combined_genes[idx])
            new_sigmas.append(combined_sigmas[idx])
            new_eval.append(combined_eval[idx])

        pop_genes, pop_sigmas, parent_eval = (
            np.array(new_genes),
            np.array(new_sigmas),
            new_eval,
        )

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
