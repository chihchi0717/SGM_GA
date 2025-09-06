import os
import re
import csv
import shutil
import threading
from datetime import datetime

import numpy as np
import pandas as pd

import config

try:
    import keyboard
except ImportError:
    keyboard = None

# === 全域停止事件 ===
immediate_stop_event = threading.Event()
graceful_stop_event = threading.Event()


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


def save_model_report(models, output_path: str):
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name, model in models.items():
                df_coef = model.get_coefficients_df()
                if not df_coef.empty:
                    df_coef.to_excel(writer, sheet_name=name)
        print(f"✅ 模型報告已成功儲存至：{output_path}")
    except Exception as e:
        print(f"❌ 儲存模型報告失敗：{e}")


def write_run_config(use_compensation_model):
    cfg = {
        "USE_COMPENSATION_MODEL": use_compensation_model,
        "POP_SIZE": config.POP_SIZE,
        "OFFSPRING_SIZE": config.OFFSPRING_SIZE,
        "OFFSPRING_PARENT_RATIO": config.OFFSPRING_PARENT_RATIO,
        "INITIAL_SIGMA_FACTOR": config.INITIAL_SIGMA_FACTOR,
        "N_GENERATIONS": config.N_GENERATIONS,
        "MAX_WORKERS": config.MAX_WORKERS,
        "SIDE_BOUND": config.SIDE_BOUND,
        "ANGLE_BOUND": config.ANGLE_BOUND,
        "TAU_PRIME": config.TAU_PRIME,
        "TAU": config.TAU,
        "GLOBAL_SEED": config.GLOBAL_SEED,
        "SIGMA_MIN": config.SIGMA_MIN.tolist(),
        "save_root": config.SAVE_ROOT,
        "log_dir": config.LOG_DIR,
    }
    try:
        p = os.path.join(config.LOG_DIR, "run_config.txt")
        with open(p, "w", encoding="utf-8") as f:
            for k, v in cfg.items():
                f.write(f"{k} = {v}\n")
        print(f"🔧 執行設定已輸出到 {p}")
    except Exception as e:
        print(f"⚠️ 無法寫入 run_config.txt: {e}")


def send_error(subject: str, body: str):
    try:
        err_dir = os.path.join(config.LOG_DIR, "ES_ErrorLogs")
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
        from evolution import calculate_dependent_variables

        design_params = calculate_dependent_variables(individual)

    fit, eff, proc = fitness_data[0], fitness_data[1], fitness_data[2]
    angle_effs = fitness_data[3] if len(fitness_data) >= 4 else []

    from evolution import format_for_log

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
        "random_seed": seed if seed is not None else config.GLOBAL_SEED,
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
    macro_dir = os.path.join(config.BASE_DIR, "Macro")
    scm_file = os.path.join(macro_dir, "Sim.scm")
    print(f"複製 SCM 檔案: {scm_file}")
    for i in range(1, config.POP_SIZE + config.OFFSPRING_SIZE + 1):
        folder = os.path.join(config.SAVE_ROOT, f"P{i}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(scm_file, folder)


def load_latest_state():
    """檢查最新日誌檔以決定啟動模式（全新、接續、或從中斷點恢復）。"""
    if not os.path.exists(config.LOG_DIR):
        return 1, None, None, None, None

    log_files = [
        f
        for f in os.listdir(config.LOG_DIR)
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
                last_path = os.path.join(config.LOG_DIR, fn)

    if latest_gen_num == -1:
        return 1, None, None, None, None

    try:
        df = pd.read_csv(last_path)
        is_aborted = (df["fitness"].apply(safe_float) == -9998.0).any()

        parent_rows = df[df["role"] == "parent"]
        is_complete = len(parent_rows) >= config.POP_SIZE and not is_aborted

        if is_complete:
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
            print(f"🔁 偵測到未完成的第 {start_gen} 代，將從此代繼續。")

            if start_gen == 1:
                return 1, None, None, None, df

            parent_old_rows = df[df["role"] == "parent_old"]
            if len(parent_old_rows) < config.POP_SIZE:
                print(f"⚠️ 第 {start_gen} 代日誌損毀，將從頭開始。")
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
