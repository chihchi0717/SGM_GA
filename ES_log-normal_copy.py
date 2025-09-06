import os
import re
import gc
import csv
import time
import math
import shutil
import random
import asyncio
import argparse
import traceback
import threading
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

# === 你的外部模組（可用假模組） ===
# from draw_New import draw_ # 假設此模組存在
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_ES import evaluate_fitness

# pywinauto 和 keyboard 不是必需；若無則忽略
try:
    from pywinauto import application, findwindows
except Exception:
    pass

try:
    import keyboard
except ImportError:
    keyboard = None

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


def augment_feats_for_lengths_custom_interactions(
    X_raw: np.ndarray,
    add_ratios: bool,
    add_sincos: bool,
    return_names: bool = False,
    add_interactions: bool = False,
    add_aa_interact: bool = False,
):
    """
    對長度模型的特徵進行增強，交互作用項被拆分為獨立開關。
    """
    s1, s2, s3, a3, a1, a2 = (X_raw[:, i] for i in range(6))
    final_feats = [s1, s2, s3, a3, a1, a2]
    final_names = FEATURES.copy()

    if add_ratios:
        eps = 1e-9
        final_feats += [
            s1 / np.clip(s2, eps, None),
            s1 / np.clip(s3, eps, None),
            s2 / np.clip(s3, eps, None),
        ]
        final_names += ["r12", "r13", "r23"]

    if add_sincos:
        r1, r2, r3 = np.deg2rad(a1), np.deg2rad(a2), np.deg2rad(a3)
        final_feats += [
            np.sin(r1),
            np.cos(r1),
            np.sin(r2),
            np.cos(r2),
            np.sin(r3),
            np.cos(r3),
        ]
        final_names += ["sin_a1", "cos_a1", "sin_a2", "cos_a2", "sin_a3", "cos_a3"]

    s_features = {"s1": X_raw[:, 0], "s2": X_raw[:, 1], "s3": X_raw[:, 2]}
    a_features = {"a3": X_raw[:, 3], "a1": X_raw[:, 4], "a2": X_raw[:, 5]}

    if add_interactions:
        # 邊長 * 邊長
        for s_i, s_j in combinations(s_features.keys(), 2):
            final_feats.append(s_features[s_i] * s_features[s_j])
            final_names.append(f"{s_i}*{s_j}")
        # 邊長 * 角度
        for s_key in s_features:
            for a_key in a_features:
                final_feats.append(s_features[s_key] * a_features[a_key])
                final_names.append(f"{s_key}*{a_key}")

    if add_aa_interact:
        for a_i, a_j in combinations(a_features.keys(), 2):
            final_feats.append(a_features[a_i] * a_features[a_j])
            final_names.append(f"{a_i}*{a_j}")

    X_aug = np.column_stack(final_feats)

    if return_names:
        return X_aug, final_names
    return X_aug


class ModelHuber:
    """一個通用的 Huber 迴歸模型類別，封裝了特徵工程和縮放。"""

    def __init__(
        self,
        scale: bool,
        alpha: float,
        epsilon: float,
        max_iter: int,
        add_ratios: bool = False,
        add_sincos: bool = False,
        add_interactions: bool = False,
        add_aa_interact: bool = False,
    ):
        self.scaler: Optional[StandardScaler] = None
        self.scale = scale
        self.model = HuberRegressor(
            alpha=alpha, epsilon=epsilon, max_iter=int(max_iter)
        )
        self.add_ratios = add_ratios
        self.add_sincos = add_sincos
        self.add_interactions = add_interactions
        self.add_aa_interact = add_aa_interact
        self.feature_names_: List[str] = []

    def _augment(self, X_raw: np.ndarray, fit_mode: bool = False) -> np.ndarray:
        if fit_mode:
            X_aug, names = augment_feats_for_lengths_custom_interactions(
                X_raw,
                add_ratios=self.add_ratios,
                add_sincos=self.add_sincos,
                return_names=True,
                add_interactions=self.add_interactions,
                add_aa_interact=self.add_aa_interact,
            )
            self.feature_names_ = names
            return X_aug
        else:
            return augment_feats_for_lengths_custom_interactions(
                X_raw,
                add_ratios=self.add_ratios,
                add_sincos=self.add_sincos,
                add_interactions=self.add_interactions,
                add_aa_interact=self.add_aa_interact,
            )

    def fit(self, df: pd.DataFrame, target: str):
        d = df[FEATURES + [target]].dropna().copy()
        X_raw = d[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw, fit_mode=True)

        if self.scale:
            self.scaler = StandardScaler()
            X_aug = self.scaler.fit_transform(X_aug)

        self.model.fit(X_aug, d[target].to_numpy(dtype=float))

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        X_raw = df_features[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw, fit_mode=False)
        if self.scale and self.scaler:
            X_aug = self.scaler.transform(X_aug)
        return self.model.predict(X_aug)

    def get_coefficients_df(self) -> pd.DataFrame:
        """取得模型係數並回傳 DataFrame"""
        if not self.feature_names_ or not hasattr(self.model, "coef_"):
            return pd.DataFrame()

        s = pd.Series(self.model.coef_, index=self.feature_names_, name="coefficient")
        s["_intercept"] = self.model.intercept_
        return s.to_frame()


# ==============================================================================
# === 整合結束 ===
# ==============================================================================


# === 路徑設定 ===
TRAIN_DATA_PATH = r".\printing\regression\analysis_results_0.6_0.9.xlsx"

# --- 全域模型物件 ---
model_s2: Optional[ModelHuber] = None
model_s3: Optional[ModelHuber] = None
model_ang: Optional[ModelHuber] = None


# === 全域開關 ===
USE_COMPENSATION_MODEL = True
immediate_stop_event = threading.Event()
graceful_stop_event = threading.Event()


# === ES 參數 ===
POP_SIZE = 10
OFFSPRING_PARENT_RATIO = 10
OFFSPRING_SIZE = POP_SIZE * OFFSPRING_PARENT_RATIO
INITIAL_SIGMA_FACTOR = 0.15
N_GENERATIONS = 100
SIDE_BOUND = [0.6, 0.9]
ANGLE_BOUND = [30.0, 90.0]

# === 並行處理設定 ===
MAX_WORKERS = os.cpu_count() or 4
autocad_lock = threading.Lock()
tracepro_lock = threading.Lock()


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


def safe_float(value, default=0.0):
    """安全地將值轉換為浮點數，如果轉換失敗則回傳預設值。"""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def setup_keyboard_hooks():
    """設定鍵盤快速鍵：'q' 立即停止, 'f' 完成當前世代後停止。"""
    print("\n*** 按下 'f' 鍵可完成當前世代後停止，按下 'q' 鍵可立即中止 ***\n")

    def stop_immediately():
        if not immediate_stop_event.is_set():
            print("\n*** 收到 'q' 立即停止訊號！將盡快中止目前所有任務... ***")
            immediate_stop_event.set()
            graceful_stop_event.set()

    def stop_gracefully():
        if not graceful_stop_event.is_set():
            print(
                "\n*** 收到 'f' 停止訊號！將在目前這一代 (Generation) 完成後停止... ***"
            )
            graceful_stop_event.set()

    keyboard.add_hotkey("q", stop_immediately)
    keyboard.add_hotkey("f", stop_gracefully)


def save_model_report(models: Dict[str, ModelHuber], output_path: str):
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name, model in models.items():
                df_coef = model.get_coefficients_df()
                if not df_coef.empty:
                    df_coef.to_excel(writer, sheet_name=name)
        print(f"✅ 模型報告已成功儲存至：{output_path}")
    except Exception as e:
        print(f"❌ 儲存模型報告失敗：{e}")


def write_run_config():
    cfg = {
        "USE_COMPENSATION_MODEL": USE_COMPENSATION_MODEL,
        "POP_SIZE": POP_SIZE,
        "OFFSPRING_SIZE": OFFSPRING_SIZE,
        "OFFSPRING_PARENT_RATIO": OFFSPRING_PARENT_RATIO,
        "INITIAL_SIGMA_FACTOR": INITIAL_SIGMA_FACTOR,
        "N_GENERATIONS": N_GENERATIONS,
        "MAX_WORKERS": MAX_WORKERS,
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
        return None, None
    a1_pred_rad = math.atan2(s3_sin_a3, denominator)
    a1_pred_deg = math.degrees(a1_pred_rad)
    if a1_pred_deg <= 0 or a1_pred_deg >= 180:
        return None, None
    sin_a1 = math.sin(a1_pred_rad)
    if abs(sin_a1) < 1e-9:
        return None, None
    s1_pred = s3_pred * math.sin(a3_pred_rad) / sin_a1
    return s1_pred, a1_pred_deg


def calculate_dependent_variables(individual):
    s1, s2, a1_deg = individual[0], individual[1], individual[2]
    a1_rad = math.radians(a1_deg)
    s3 = math.sqrt(pow(s1, 2) + pow(s2, 2) - 2 * s1 * s2 * math.cos(a1_rad))
    if abs(s1) < 1e-9 or abs(s3) < 1e-9:
        return {"s1": s1, "s2": s2, "s3": s3, "a1": a1_deg, "is_valid": False}
    cos_a2_arg = max(
        -1.0, min(1.0, (pow(s1, 2) + pow(s3, 2) - pow(s2, 2)) / (2 * s1 * s3))
    )
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

    if USE_COMPENSATION_MODEL:
        s2_shrink, s3_shrink, a3_pred = predict_shrinkage_and_angle(design_params)
        s2_pred = s2 * (1 - s2_shrink)
        s3_pred = s3 * (1 - s3_shrink)
        if s2_pred <= 0 or s3_pred <= 0:
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
    else:
        design_params.update(
            {
                "is_valid": True,
                "s1_predicted": s1,
                "s2_predicted": s2,
                "s3_predicted": s3,
                "a1_predicted": a1_deg,
                "a3_predicted": a3_deg,
            }
        )
    return design_params


def evaluate_individual(
    individual: np.ndarray, folder: str, idx: int
) -> Tuple[Tuple, Dict]:
    aborted_fitness = (-9998.0, 0, 0, [], [])
    aborted_params = {"is_valid": False, "aborted": True}

    if immediate_stop_event.is_set():
        print(f"  [Worker {idx}] 偵測到立即停止訊號，中止評估。")
        return aborted_fitness, aborted_params

    penalty_fitness = (-9999.0, 0, 0, [], [])
    print(f"  [Worker {idx}] 開始評估個體...")
    design_params = calculate_dependent_variables(individual)

    if immediate_stop_event.is_set():
        print(f"  [Worker {idx}] 偵測到立即停止訊號，中止評估。")
        return aborted_fitness, design_params

    if not design_params.get("is_valid", False):
        print(f"  [Worker {idx}] -> 幾何/物理無解，指定懲罰 fitness。")
        return penalty_fitness, design_params

    params_for_build = [
        float(design_params["s1_predicted"]),
        float(design_params["s2_predicted"]),
        float(design_params["a1_predicted"]),
    ]

    build_success = False
    with autocad_lock:
        if immediate_stop_event.is_set():
            print(f"  [Worker {idx}] 偵測到立即停止訊號，跳過建模。")
            return aborted_fitness, design_params

        print(f"  [Worker {idx}] 取得 AutoCAD 鎖，開始建模...")
        for attempt in range(3):
            if immediate_stop_event.is_set():
                print(f"  [Worker {idx}] 偵測到立即停止訊號，取消建模嘗試。")
                break
            try:
                result, _ = Build_model(
                    params_for_build,
                    mode="triangle",
                    folder=folder,
                    fillet=2,
                    radius_vertex=0.036,
                    radius_inside=0.053,
                    light_source_length=1,
                )
                if result == 1:
                    build_success = True
                    break
            except Exception as e:
                print(f"❌ [Worker {idx}] Build_model 第 {attempt+1} 次失敗：{e}")
                time.sleep(1)
        print(f"  [Worker {idx}] 釋放 AutoCAD 鎖。")

    if not build_success:
        if immediate_stop_event.is_set():
            return aborted_fitness, design_params
        print(f"  [Worker {idx}] -> 建模失敗，指定懲罰 fitness。")
        return penalty_fitness, design_params

    if immediate_stop_event.is_set():
        print(f"  [Worker {idx}] 偵測到立即停止訊號，跳過模擬。")
        return aborted_fitness, design_params

    simulation_success = False
    with tracepro_lock:
        if immediate_stop_event.is_set():
            print(f"  [Worker {idx}] 偵測到立即停止訊號，跳過模擬。")
            return aborted_fitness, design_params

        print(f"  [Worker {idx}] 取得 TracePro 鎖，開始光學模擬...")
        try:
            tracepro_fast(os.path.join(folder, "Sim.scm"))
            simulation_success = True
        except Exception as e:
            print(f"⚠️ [Worker {idx}] tracepro_fast 失敗：{e}")
        print(f"  [Worker {idx}] 釋放 TracePro 鎖。")

    if not simulation_success:
        if immediate_stop_event.is_set():
            return aborted_fitness, design_params
        print(f"  [Worker {idx}] -> 光學模擬失敗，指定懲罰 fitness。")
        return penalty_fitness, design_params

    if immediate_stop_event.is_set():
        print(f"  [Worker {idx}] 偵測到立即停止訊號，跳過 fitness 計算。")
        return aborted_fitness, design_params

    try:
        print(f"  [Worker {idx}] -> 開始計算 fitness...")
        fitness_data = evaluate_fitness(
            folder,
            individual,
            return_uniformity=False,
            eff_weight=1,
            process_weight=1,
            uni_weight=1,
        )
        print(f"  [Worker {idx}] -> 評估完成。")
        return fitness_data, design_params
    except Exception as e:
        print(f"⚠️ [Worker {idx}] evaluate_fitness 失敗：{e}")
        return penalty_fitness, design_params


async def evaluate_individual_async(executor, individual, folder, idx):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, evaluate_individual, individual, folder, idx
    )


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
    angle_effs = fitness_data[3] if len(fitness_data) >= 4 else []
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
        "uniformity": f"{0.0:.6f}",
        "is_valid": design_params.get("is_valid", False),
        "random_seed": seed if seed is not None else GLOBAL_SEED,
    }

    all_angles = range(10, 90, 10)
    for a in all_angles:
        row[f"eff_{a}"] = "0.000000"
        row[f"uni_{a}"] = "0.000000"

    for a, e in zip(all_angles, angle_effs):
        row[f"eff_{a}"] = f"{e:.6f}"

    angle_unis = fitness_data[4] if len(fitness_data) >= 5 else []
    for a, u in zip(all_angles, angle_unis):
        row[f"uni_{a}"] = f"{u:.6f}"

    return row


def is_duplicate_history(
    full_history_rows, individual, tol=np.array([1e-3, 1e-3, 5e-2])
):
    x = np.array(individual, dtype=float)
    for row in reversed(full_history_rows):
        try:
            y = np.array(
                [float(row.get(k, "nan")) for k in ["S1", "S2", "A1"]], dtype=float
            )
        except Exception:
            continue
        if np.all(np.abs(x - y) <= tol):
            fit = (
                float(row.get(k) or 0.0)
                for k in ["fitness", "efficiency", "process_score"]
            )
            effs = [float(row.get(f"eff_{a}") or 0.0) for a in range(10, 90, 10)]
            return True, (*fit, effs, [])
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
    sigma0 = np.tile((VAR_RANGES * INITIAL_SIGMA_FACTOR), (POP_SIZE, 1))
    return pop, sigma0


def make_offspring(pop_genes, pop_sigmas):
    children_genes, children_sigmas, parent_pairs = [], [], []
    for _ in range(OFFSPRING_SIZE):
        p1, p2 = random.sample(range(POP_SIZE), 2)
        x1, x2, s1, s2 = pop_genes[p1], pop_genes[p2], pop_sigmas[p1], pop_sigmas[p2]
        alpha = np.random.rand()
        x_bar, s_bar = alpha * x1 + (1 - alpha) * x2, alpha * s1 + (1 - alpha) * s2
        noise = np.random.randn()
        new_sigma = s_bar * np.exp(TAU_PRIME * noise + TAU * np.random.randn(n))
        new_sigma = np.maximum(new_sigma, SIGMA_MIN)
        child = reflect_bounds(x_bar + new_sigma * np.random.randn(n))
        children_genes.append(child)
        children_sigmas.append(new_sigma)
        parent_pairs.append((p1, p2))
    return children_genes, children_sigmas, parent_pairs


def load_latest_state():
    """檢查最新日誌檔以決定啟動模式（全新、接續、或從中斷點恢復）。"""
    if not os.path.exists(log_dir):
        return 1, None, None, None, None

    log_files = [
        f
        for f in os.listdir(log_dir)
        if f.startswith("fitness_gen") and f.endswith(".csv")
    ]
    if not log_files:
        return 1, None, None, None, None

    latest_gen_num = -1
    last_path = ""
    for fn in log_files:
        m = re.search(r"fitness_gen(\d+)", fn)
        if m:
            gen_num = int(m.group(1))
            if gen_num > latest_gen_num:
                latest_gen_num = gen_num
                last_path = os.path.join(log_dir, fn)

    if latest_gen_num == -1:
        return 1, None, None, None, None

    try:
        df = pd.read_csv(last_path)
        parent_rows = df[df["role"] == "parent"]

        if len(parent_rows) >= POP_SIZE:
            start_gen = latest_gen_num + 1
            print(
                f"🔁 偵測到已完成的第 {latest_gen_num} 代，將從第 {start_gen} 代開始。"
            )
            pop_genes = parent_rows[["S1", "S2", "A1"]].to_numpy(dtype=float)
            pop_sigmas = parent_rows[["sigma1", "sigma2", "sigma3"]].to_numpy(
                dtype=float
            )
            parent_eval = [
                (
                    safe_float(r["fitness"]),
                    safe_float(r["efficiency"]),
                    safe_float(r["process_score"]),
                    [safe_float(r.get(f"eff_{a}")) for a in range(10, 90, 10)],
                    [],
                )
                for _, r in parent_rows.iterrows()
            ]
            return start_gen, pop_genes, pop_sigmas, parent_eval, None
        else:
            start_gen = latest_gen_num
            # FIX: Special handling for incomplete Generation 1
            if start_gen == 1:
                print(f"⚠️ 偵測到第 1 代未完成。將從頭開始重新評估第 1 代。")
                return 1, None, None, None, None

            print(f"🔁 偵測到未完成的第 {start_gen} 代，將從此代繼續。")
            parent_old_rows = df[df["role"] == "parent_old"]
            if len(parent_old_rows) < POP_SIZE:
                print(
                    f"⚠️ 第 {start_gen} 代日誌損毀 (找不到完整的 'parent_old' 資訊)，將從頭開始。"
                )
                return 1, None, None, None, None

            pop_genes = parent_old_rows[["S1", "S2", "A1"]].to_numpy(dtype=float)
            pop_sigmas = parent_old_rows[["sigma1", "sigma2", "sigma3"]].to_numpy(
                dtype=float
            )
            parent_eval = [
                (
                    safe_float(r["fitness"]),
                    safe_float(r["efficiency"]),
                    safe_float(r["process_score"]),
                    [safe_float(r.get(f"eff_{a}")) for a in range(10, 90, 10)],
                    [],
                )
                for _, r in parent_old_rows.iterrows()
            ]
            return start_gen, pop_genes, pop_sigmas, parent_eval, df
    except Exception as e:
        print(f"⚠️ 讀取日誌檔 '{last_path}' 失敗: {e}。將從頭開始。")
        return 1, None, None, None, None


async def main_async(args):
    global USE_COMPENSATION_MODEL, model_s2, model_s3, model_ang
    USE_COMPENSATION_MODEL = not args.no_compensation
    print(f"收縮補償模型: {'🟢 啟用' if USE_COMPENSATION_MODEL else '🔴 禁用'}")

    print("\n--- 初始化並訓練補償模型 ---")
    model_kwargs = {
        "scale": True,
        "alpha": 1.0,
        "epsilon": 1000,
        "max_iter": 10000,
        "add_ratios": args.add_ratios,
        "add_sincos": args.add_sincos,
        "add_interactions": args.add_interactions,
        "add_aa_interact": args.add_aa_interact,
    }
    model_s2 = ModelHuber(**model_kwargs)
    model_s3 = ModelHuber(**model_kwargs)
    model_ang = ModelHuber(**{**model_kwargs, "alpha": 0.1})
    try:
        df_train = pd.read_excel(TRAIN_DATA_PATH)
        print(f"✅ 成功讀取訓練資料 '{TRAIN_DATA_PATH}'。")
        model_s2.fit(df_train, "delta_s2")
        model_s3.fit(df_train, "delta_s3")
        model_ang.fit(df_train, "DIP_a3(deg)")
        print("✅ 所有模型訓練完成。")
    except Exception as e:
        print(f"❌ 模型訓練失敗: {e}")
        return
    if args.save_report:
        save_model_report(
            {"model_s2": model_s2, "model_s3": model_s3, "model_ang": model_ang},
            args.save_report,
        )
        if args.report_only:
            print("報告已儲存，程式結束。")
            return

    print("---------------------------\n")
    copy_scm_to_all_folders()
    write_run_config()

    start_gen, pop_genes, pop_sigmas, parent_eval, incomplete_df = load_latest_state()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    if pop_genes is None:
        print(
            f"🌱 無有效歷史紀錄，從第 1 代全新開始 (使用 {MAX_WORKERS} 個並行 workers)"
        )
        start_gen = 1
        pop_genes, pop_sigmas = init_population()
        tasks = [
            evaluate_individual_async(
                executor, pop_genes[i], os.path.join(save_root, f"P{i+1}"), i + 1
            )
            for i in range(POP_SIZE)
        ]
        results = await asyncio.gather(*tasks)
        initial_rows = [
            create_log_row(
                pop_genes[i], pop_sigmas[i], res[0], 1, "parent", (-1, -1), res[1]
            )
            for i, res in enumerate(results)
        ]
        parent_eval = [res[0] for res in results]
        best = max(d[0] for d in parent_eval) if parent_eval else -9999
        fn = f"fitness_gen1_max{best:.2f}.csv"
        save_generation_log(initial_rows, os.path.join(log_dir, fn))
        print(f"★ 第 1 代完成，存為 {fn}")
        start_gen = 2

    for gen in range(start_gen, N_GENERATIONS + 1):
        if graceful_stop_event.is_set() or immediate_stop_event.is_set():
            print(f"在第 {gen} 代開始前偵測到停止訊號，準備結束程式。")
            break

        print(
            f"\n{'='*18} GENERATION {gen} (使用 {MAX_WORKERS} 個並行 workers) {'='*18}"
        )

        tasks_to_run, indices_to_run = [], []
        offspring_eval = [None] * OFFSPRING_SIZE
        current_rows = []

        if incomplete_df is not None and gen == incomplete_df["generation"].iloc[0]:
            print(f"  -> 恢復第 {gen} 代的未完成工作...")
            offspring_rows = incomplete_df[
                incomplete_df["role"] == "offspring"
            ].reset_index()
            children_genes = offspring_rows[["S1", "S2", "A1"]].to_numpy(dtype=float)
            children_sigmas = offspring_rows[["sigma1", "sigma2", "sigma3"]].to_numpy(
                dtype=float
            )
            parent_pairs = list(
                zip(
                    offspring_rows["parent_idx1"].astype(int),
                    offspring_rows["parent_idx2"].astype(int),
                )
            )

            for i in range(len(offspring_rows)):
                row = offspring_rows.iloc[i]
                fitness = safe_float(row.get("fitness"), -10000)
                if fitness > -9990:
                    offspring_eval[i] = (
                        fitness,
                        safe_float(row.get("efficiency")),
                        safe_float(row.get("process_score")),
                        [safe_float(row.get(f"eff_{a}")) for a in range(10, 90, 10)],
                        [],
                    )
                    current_rows.append(row.to_dict())
                else:
                    folder = os.path.join(save_root, f"P{POP_SIZE + i + 1}")
                    tasks_to_run.append(
                        evaluate_individual_async(
                            executor, children_genes[i], folder, POP_SIZE + i + 1
                        )
                    )
                    indices_to_run.append(i)
            incomplete_df = None
        else:
            history_rows = []
            try:
                files = sorted(
                    [f for f in os.listdir(log_dir) if f.startswith("fitness_gen")],
                    key=lambda x: int(re.search(r"gen(\d+)", x).group(1)),
                )
                for f in files:
                    history_rows.extend(
                        list(
                            csv.DictReader(
                                open(os.path.join(log_dir, f), "r", encoding="utf-8")
                            )
                        )
                    )
            except Exception as e:
                print(f"⚠️ 讀歷史日誌失敗：{e}")

            children_genes, children_sigmas, parent_pairs = make_offspring(
                pop_genes, pop_sigmas
            )
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
                    folder = os.path.join(save_root, f"P{POP_SIZE + i + 1}")
                    tasks_to_run.append(
                        evaluate_individual_async(
                            executor, child, folder, POP_SIZE + i + 1
                        )
                    )
                    indices_to_run.append(i)

        if tasks_to_run:
            print(f"  -> 正在並行評估 {len(tasks_to_run)} 個新子代...")
            results = await asyncio.gather(*tasks_to_run)
            for task_idx, (eval_data, design_params) in enumerate(results):
                original_idx = indices_to_run[task_idx]
                offspring_eval[original_idx] = eval_data
                if not design_params.get("aborted"):
                    current_rows.append(
                        create_log_row(
                            children_genes[original_idx],
                            children_sigmas[original_idx],
                            eval_data,
                            gen,
                            "offspring",
                            parent_pairs[original_idx],
                            design_params,
                        )
                    )

        if any(v is None for v in offspring_eval):
            raise RuntimeError(f"第 {gen} 代有子代未評估")

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

        combined_genes = np.vstack([pop_genes, np.array(children_genes)])
        combined_sigmas = np.vstack([pop_sigmas, np.array(children_sigmas)])
        combined_eval = parent_eval + offspring_eval
        fitness_all = [d[0] for d in combined_eval]
        order = np.argsort(fitness_all)[::-1]

        pop_genes = np.array([combined_genes[i] for i in order[:POP_SIZE]])
        pop_sigmas = np.array([combined_sigmas[i] for i in order[:POP_SIZE]])
        parent_eval = [combined_eval[i] for i in order[:POP_SIZE]]

        for i in range(POP_SIZE):
            current_rows.append(
                create_log_row(
                    pop_genes[i], pop_sigmas[i], parent_eval[i], gen, "parent", (-1, -1)
                )
            )

        best_fitness = max((f for f in fitness_all if f > -9990), default=-9999)
        out = f"fitness_gen{gen}_max{best_fitness:.2f}.csv"
        save_generation_log(current_rows, os.path.join(log_dir, out))
        print(
            f"★ Generation {gen} 完成，最佳 fitness = {best_fitness:.4f}，日誌：{out}"
        )
        gc.collect()

    executor.shutdown()
    if graceful_stop_event.is_set() or immediate_stop_event.is_set():
        print("\n🛑 程式已由使用者安全停止。")
    else:
        print("\n🎉 所有世代執行完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="演化策略優化器")
    parser.add_argument("--add-ratios", action="store_true", help="加入邊長比例特徵")
    parser.add_argument(
        "--add-sincos", action="store_true", help="加入角度的 sin/cos 特徵"
    )
    parser.add_argument(
        "--no-interactions",
        dest="add_interactions",
        action="store_false",
        help="禁用邊長與角度交互作用 (s*s, s*a)",
    )
    parser.add_argument(
        "--no-aa-interact",
        dest="add_aa_interact",
        action="store_false",
        help="禁用角度間交互作用 (a*a)",
    )
    parser.add_argument(
        "--save-report", type=str, metavar="FILENAME", help="將模型係數儲存至 Excel"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="僅儲存報告，不執行優化"
    )
    parser.add_argument(
        "--no-compensation", action="store_true", help="禁用收縮補償模型"
    )
    parser.set_defaults(add_interactions=True, add_aa_interact=True)
    cli_args = parser.parse_args()

    if keyboard:
        setup_keyboard_hooks()
    else:
        print(
            "\n⚠️  未安裝 'keyboard' 模組，無法使用快速鍵停止。請執行 'pip install keyboard'。"
        )

    try:
        asyncio.run(main_async(cli_args))
    except KeyboardInterrupt:
        print("\n🛑 偵測到 Ctrl+C，正在準備立即停止...")
        immediate_stop_event.set()
        graceful_stop_event.set()
    except Exception as e:
        subject = "演化策略主程式發生致命錯誤"
        body = f"錯誤類型: {type(e).__name__}\n錯誤訊息: {e}\n\n追蹤訊息:\n{traceback.format_exc()}"
        print(f"❌ {subject}")
        send_error(subject, body)
    finally:
        print("\n程式已結束。")
