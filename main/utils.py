import os
import re
import csv
import threading
from datetime import datetime

import numpy as np
import pandas as pd

import config

try:
    import keyboard
except ImportError:
    keyboard = None

# === Global Stop Events ===
immediate_stop_event = threading.Event()
graceful_stop_event = threading.Event()

# === Global Output Directory ===
_output_dir = None


def set_output_dir(path):
    """Sets the global output directory path."""
    global _output_dir
    _output_dir = path


def get_output_dir():
    """Gets the global output directory path."""
    if _output_dir is None:
        raise ValueError("Output directory has not been set.")
    return _output_dir


def safe_float(value, default=0.0):
    """Safely converts a value to a float, returning a default on failure."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def format_for_log(individual):
    """Formats an individual's genes into a dictionary for logging."""
    return {
        "S2": f"{individual[0]:.4f}",
        "S3": f"{individual[1]:.4f}",
        "A3": f"{individual[2]:.4f}",
    }


def setup_keyboard_hooks():
    """Sets up keyboard hotkeys: 'q' for immediate stop, 'f' for graceful stop."""
    print(
        "\n*** Press 'f' to stop after the current generation, or 'q' to stop immediately. ***\n"
    )
    if keyboard:
        keyboard.add_hotkey(
            "q", lambda: (immediate_stop_event.set(), graceful_stop_event.set())
        )
        keyboard.add_hotkey("f", lambda: graceful_stop_event.set())


def create_log_row(genes, sigmas, eval_result, gen, role, parent_indices):
    """
    創建一個標準化的日誌行 (字典)。
    【修改】: 新增了 Predicted_S2, Predicted_S3, Predicted_A3 欄位。
    """
    # 【修改】解包 eval_result，使其能處理 5, 6, 7 個元素，確保相容性
    if len(eval_result) == 7:
        (
            fitness,
            efficiency,
            process_score,
            angle_efficiencies,
            _,
            prediction_info,
            sim_geometry_info,
        ) = eval_result
    elif len(eval_result) == 6:
        fitness, efficiency, process_score, angle_efficiencies, _, prediction_info = (
            eval_result
        )
        sim_geometry_info = {}
    else:
        fitness, efficiency, process_score, angle_efficiencies, _ = eval_result
        prediction_info = {}
        sim_geometry_info = {}

    row = {
        "generation": gen,
        "role": role,
        "parent1_idx": parent_indices[0],
        "parent2_idx": parent_indices[1],
        "fitness": f"{fitness:.4f}",
        "efficiency": f"{efficiency:.4f}",
        "process_score": f"{process_score:.4f}",
        # 設計值 (演算法決定的基因值)
        "Design_S2": f"{genes[0]:.4f}",
        "Design_S3": f"{genes[1]:.4f}",
        "Design_A3": f"{genes[2]:.4f}",
        # 預測收縮/變化率 (由補償模型預測)
        "Pred_delta_s2": f"{prediction_info.get('pred_delta_s2', 0.0):.6f}",
        "Pred_delta_s3": f"{prediction_info.get('pred_delta_s3', 0.0):.6f}",
        # "Pred_dip_a3": f"{prediction_info.get('pred_dip_a3', 0.0):.4f}",
        # 預測幾何值 (最終用於模擬的值)
        "Predicted_S2": f"{sim_geometry_info.get('predicted_s2', 0.0):.4f}",
        "Predicted_S3": f"{sim_geometry_info.get('predicted_s3', 0.0):.4f}",
        "Predicted_A3": f"{sim_geometry_info.get('predicted_a3', 0.0):.4f}",
        # Sigma 值
        "sigma1": f"{sigmas[0]:.6f}",
        "sigma2": f"{sigmas[1]:.6f}",
        "sigma3": f"{sigmas[2]:.6f}",
    }
    if angle_efficiencies:
        for angle, eff in angle_efficiencies.items():
            row[f"eff_{angle}"] = f"{eff:.4f}"
    return row


def save_generation_log(rows, filepath):
    """Saves all log rows for a generation to a CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except IOError as e:
        print(f"❌ Failed to save log file '{filepath}': {e}")


# In utils.py, replace the 'resume_from_log' function with this new version.
def resume_from_log():
    log_dir = config.LOG_DIR
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_files = [f for f in os.listdir(log_dir) if f.startswith("fitness_gen")]
    except FileNotFoundError:
        return 1, None, None, None, []

    if not log_files:
        return 1, None, None, None, []

    latest_gen_num = 0
    latest_file = ""
    for f in log_files:
        match = re.search(r"fitness_gen(\d+)", f)
        if match and int(match.group(1)) > latest_gen_num:
            latest_gen_num = int(match.group(1))
            latest_file = f

    if not latest_file:
        return 1, None, None, None, []

    latest_filepath = os.path.join(log_dir, latest_file)
    try:
        df = pd.read_csv(latest_filepath)
        is_complete = "parent" in df["role"].values

        # 【邏輯修正】: 定義正確的基因欄位名稱
        gene_columns = ["Design_S2", "Design_S3", "Design_A3"]

        # 處理欄位可能不存在的舊格式日誌
        if "S2" in df.columns and "Design_S2" not in df.columns:
            gene_columns = ["S2", "S3", "A3"]
        elif not all(col in df.columns for col in gene_columns):
            print(f"⚠️ 日誌檔案 '{latest_file}' 缺少必要的基因欄位。將從頭開始。")
            return 1, None, None, None, []

        if is_complete:
            print(
                f"✅ 第 {latest_gen_num} 代已完成。將從新的第 {latest_gen_num + 1} 代開始。"
            )
            start_gen = latest_gen_num + 1
            final_parents = df[df["role"] == "parent"]

            pop_genes = final_parents[gene_columns].to_numpy(dtype=float)
            pop_sigmas = final_parents[["sigma1", "sigma2", "sigma3"]].to_numpy(
                dtype=float
            )

            parent_eval = [
                (
                    safe_float(r["fitness"]),
                    safe_float(r["efficiency"]),
                    safe_float(r["process_score"]),
                    {},
                    0.0,
                )
                for _, r in final_parents.iterrows()
            ]

            return start_gen, pop_genes, pop_sigmas, parent_eval, []
        else:
            # 簡化恢復邏輯：如果世代未完成，則從該世代的初始父代重新開始
            print(f"🔁 偵測到未完成的第 {latest_gen_num} 代，將從此代重新開始評估。")
            start_gen = latest_gen_num

            parent_old_df = df[df["role"] == "parent_old"]
            if parent_old_df.empty:
                print(
                    f"⚠️ 第 {start_gen} 代日誌損毀 (找不到 'parent_old' 資訊)，將從頭開始。"
                )
                return 1, None, None, None, []

            pop_genes = parent_old_df[gene_columns].to_numpy(dtype=float)
            pop_sigmas = parent_old_df[["sigma1", "sigma2", "sigma3"]].to_numpy(
                dtype=float
            )

            # 返回 None，觸發對該世代的重新評估
            return start_gen, pop_genes, pop_sigmas, None, []

    except Exception as e:
        print(f"❌ 讀取日誌檔案 '{latest_filepath}' 時發生錯誤: {e}。將從頭開始。")
        return 1, None, None, None, []


def send_error(subject, body=""):
    """Placeholder function for sending error messages."""
    print(f"\n[ERROR REPORT] {subject}")
    if body:
        print(body.splitlines()[0])
