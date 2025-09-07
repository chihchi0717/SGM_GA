import os
import gc
import csv
import time
import math
import random
import asyncio
import shutil
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# External application modules
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_ES import evaluate_fitness

# Internal modules
import config
import utils
from models import (
    ModelHuber,
    FEATURES,
    create_length_model_from_main_defaults,
    create_angle_model_from_main_defaults,
)

# === Global Variables ===
# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Model objects
model_s2: ModelHuber = None
model_s3: ModelHuber = None
model_ang: ModelHuber = None
# Compensation model switch
USE_COMPENSATION_MODEL = True
# Thread locks
autocad_lock = threading.Lock()
tracepro_lock = threading.Lock()


def copy_scm_to_all_folders(save_root):
    """Copies the TracePro SCM script to each simulation subfolder."""
    print("\n--- Preparing simulation folders and copying SCM file ---")
    macro_dir = os.path.join(BASE_DIR, "Macro")
    scm_file = os.path.join(macro_dir, "Sim.scm")

    if not os.path.exists(scm_file):
        print(f"❌ ERROR: Source SCM file not found at '{scm_file}'.")
        return

    print(f"✅ Source SCM file found. Target root: {save_root}")
    total_individuals = config.POP_SIZE + config.OFFSPRING_SIZE
    try:
        for i in range(1, total_individuals + 1):
            folder = os.path.join(save_root, f"P{i}")
            os.makedirs(folder, exist_ok=True)
            shutil.copy(scm_file, folder)
        print(f"✅ SCM file successfully copied to {total_individuals} folders.")
    except Exception as e:
        print(f"❌ An error occurred while copying SCM files: {e}")


def apply_geometric_constraints(design_dict: dict) -> dict:
    """Calculates dependent variables (s1, a1, a2) from independent variables (s2, s3, a3)."""
    s2 = design_dict["Design_s2(mm)"]
    s3 = design_dict["Design_s3(mm)"]
    a3_deg = design_dict["Design_a3(deg)"]
    a3_rad = math.radians(a3_deg)

    s1_squared = s3**2 + s2**2 - 2 * s3 * s2 * math.cos(a3_rad)
    s1 = math.sqrt(max(0, s1_squared))

    denominator = 2 * s2 * s1 + 1e-9
    cos_a1_arg = (s2**2 + s1**2 - s3**2) / denominator
    cos_a1_arg = max(-1.0, min(1.0, cos_a1_arg))
    a1_rad = math.acos(cos_a1_arg)
    a1_deg = math.degrees(a1_rad)

    a2_deg = 180 - a1_deg - a3_deg

    return {
        "Design_s1(mm)": s1,
        "Design_s2(mm)": s2,
        "Design_s3(mm)": s3,
        "Design_a3(deg)": a3_deg,
        "Design_a1(deg)": a1_deg,
        "Design_a2(deg)": a2_deg,
    }


def predict_shrinkage_and_angle(design_params):
    """Uses the trained models to predict shrinkage rates and angle."""
    if model_s2 is None or model_s3 is None or model_ang is None:
        raise RuntimeError("Models have not been initialized.")
    df_input = pd.DataFrame([design_params])
    pred_delta_s2 = model_s2.predict(df_input)[0]
    pred_delta_s3 = model_s3.predict(df_input)[0]
    pred_angle_a3 = model_ang.predict(df_input)[0]
    return pred_delta_s2, pred_delta_s3, pred_angle_a3


def _calculate_summary_training_errors(
    model: ModelHuber, df: pd.DataFrame, target_name: str
) -> dict:
    """計算模型在訓練集上的摘要誤差指標 (R2, RMSE, MAE)。"""
    y_true = df[target_name].to_numpy()
    y_pred = model.predict(df)
    return {
        "R-squared": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def _calculate_detailed_training_errors(
    model_s2: ModelHuber, model_s3: ModelHuber, model_ang: ModelHuber, df: pd.DataFrame
) -> pd.DataFrame:
    """為每個訓練資料點計算詳細的預測誤差，並加入使用者要求的欄位。"""
    df_details = df.copy()

    pred_delta_s2 = model_s2.predict(df_details)
    pred_delta_s3 = model_s3.predict(df_details)
    pred_dip_a3 = model_ang.predict(df_details)

    results_df = df_details[FEATURES].copy()

    # --- S2 收縮率與長度誤差 ---
    results_df["Actual_delta_s2"] = df_details["delta_s2"]
    results_df["Predicted_delta_s2"] = pred_delta_s2
    results_df["Error_delta_s2"] = (
        results_df["Predicted_delta_s2"] - results_df["Actual_delta_s2"]
    )
    # 2. 長度誤差 Error_s2
    results_df["Error_s2"] = (
        df_details["Design_s2(mm)"] * (1 - results_df["Predicted_delta_s2"])
    ) - (df_details["Design_s2(mm)"] * (1 - results_df["Actual_delta_s2"]))

    # --- S3 收縮率與長度誤差 (新增) ---
    results_df["Actual_delta_s3"] = df_details["delta_s3"]
    results_df["Predicted_delta_s3"] = pred_delta_s3
    # 1. 收縮率誤差 Error_delta_s3
    results_df["Error_delta_s3"] = (
        results_df["Predicted_delta_s3"] - results_df["Actual_delta_s3"]
    )
    # 2. 長度誤差 Error_s3
    results_df["Error_s3"] = (
        df_details["Design_s3(mm)"] * (1 - results_df["Predicted_delta_s3"])
    ) - (df_details["Design_s3(mm)"] * (1 - results_df["Actual_delta_s3"]))

    # --- A3 角度誤差 (新增) ---
    results_df["Actual_delta_a3"] = df_details["DIP_a3(deg)"]
    results_df["Predicted_delta_a3"] = pred_dip_a3
    # 3. 角度誤差 Error_delta_a3
    results_df["Error_delta_a3"] = (
        results_df["Predicted_delta_a3"] - results_df["Actual_delta_a3"]
    )

    return results_df


async def initialize_and_train_models(args: argparse.Namespace):
    """根據命令列參數初始化並訓練三個補償模型。"""
    global model_s2, model_s3, model_ang, USE_COMPENSATION_MODEL

    if args.no_compensation:
        USE_COMPENSATION_MODEL = False
        print("\n🟡 收縮補償模型已禁用。")
        return True, None, None

    USE_COMPENSATION_MODEL = True
    print("\n--- 正在初始化並訓練補償模型 ---")
    try:
        df_train = pd.read_excel(config.TRAIN_DATA_PATH)
    except Exception as e:
        print(f"❌ 讀取訓練資料時發生錯誤: {e}")
        USE_COMPENSATION_MODEL = False
        return False, None, None

    model_kwargs = {
        "add_ratios": args.add_ratios,
        "add_sincos": args.add_sincos,
        "add_interactions": args.add_interactions,
        "add_aa_interact": args.add_aa_interact,
    }
    model_s2 = create_length_model_from_main_defaults(
        scale=args.scale_length, **model_kwargs
    )
    model_s3 = create_length_model_from_main_defaults(
        scale=args.scale_length, **model_kwargs
    )
    model_ang = create_angle_model_from_main_defaults(
        scale=args.scale_angle, **model_kwargs
    )

    try:
        model_s2.fit(df_train, "delta_s2")
        model_s3.fit(df_train, "delta_s3")
        model_ang.fit(df_train, "DIP_a3(deg)")
        print("\n✅ 所有模型訓練成功。")

        summary_reports = {
            "model_s2": _calculate_summary_training_errors(
                model_s2, df_train, "delta_s2"
            ),
            "model_s3": _calculate_summary_training_errors(
                model_s3, df_train, "delta_s3"
            ),
            "model_ang": _calculate_summary_training_errors(
                model_ang, df_train, "DIP_a3(deg)"
            ),
        }

        detailed_df = _calculate_detailed_training_errors(
            model_s2, model_s3, model_ang, df_train
        )

        return True, summary_reports, detailed_df
    except Exception as e:
        print(f"❌ 模型訓練過程中發生錯誤: {traceback.format_exc()}")
        USE_COMPENSATION_MODEL = False
        return False, None, None


import traceback

# In evolution.py, replace your existing 'run_simulation' function with this one.


def run_simulation(s1, s2, s3, a1, a2, a3, loop_num, individual):
    """
    Executes the external simulation programs (AutoCAD, TracePro) in the correct folder.
    This version corrects the evaluate_fitness call to use positional arguments.
    """
    folder_name = f"P{loop_num}"
    folder_path = os.path.join(utils.get_output_dir(), folder_name)
    original_cwd = os.getcwd()

    try:
        os.chdir(folder_path)

        # --- Corrected Build_model call ---
        build_success = False
        with autocad_lock:
            if utils.immediate_stop_event.is_set():
                return None

            result, _ = Build_model(
                [s1, s2, int(round(a1))],
                mode="triangle",
                folder=folder_path,
                fillet=2,
                radius_vertex=0.036,
                radius_inside=0.053,
                light_source_length=1,
                substrate=0.6,
            )
            if result == 1:
                build_success = True

        if not build_success:
            print(
                f"⚠️  Build_model failed for individual P{loop_num}. Aborting its evaluation."
            )
            return None

        # --- Corrected tracepro_fast call ---
        with tracepro_lock:
            if utils.immediate_stop_event.is_set():
                return None
            tracepro_fast(os.path.join(folder_path, "Sim.scm"))

        # --- **FINAL FIX HERE** ---
        # The 'folder' and 'individual' arguments must be POSITIONAL, not keyword.
        return evaluate_fitness(
            folder_path,  # 1st positional argument (was folder=)
            individual,  # 2nd positional argument (was individual=)
            return_uniformity=False,
            eff_weight=1,
            process_weight=1,
            uni_weight=1,
        )

    except Exception as e:
        error_subject = "Critical error during simulation execution"
        error_body = (
            f"Individual: P{loop_num}\n"
            f"Parameters: s1={s1:.4f}, s2={s2:.4f}, a1={a1:.4f}\n"
            f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        )
        print(f"❌ {error_subject}: {e}")
        utils.send_error(error_subject, error_body)
        return None
    finally:
        os.chdir(original_cwd)


def evaluate_individual(individual_data, generation, parent_indices):
    """
    評估單一個體，運行完整的模擬與適應度計算流程。
    【修改】: 此版本會額外回傳模型的預測收縮值與最終模擬幾何值。
    """
    loop_num, individual, _, _ = individual_data
    s2, s3, a3 = individual

    initial_design = apply_geometric_constraints(
        {
            "Design_s2(mm)": s2,
            "Design_s3(mm)": s3,
            "Design_a3(deg)": a3,
        }
    )

    sim_design = initial_design
    prediction_info = {}

    if USE_COMPENSATION_MODEL:
        try:
            d_s2, d_s3, p_a3 = predict_shrinkage_and_angle(initial_design)
            prediction_info = {
                "pred_delta_s2": d_s2,
                "pred_delta_s3": d_s3,
                "pred_dip_a3": p_a3,
            }
            sim_design = apply_geometric_constraints(
                {
                    "Design_s2(mm)": initial_design["Design_s2(mm)"] * (1 - d_s2),
                    "Design_s3(mm)": initial_design["Design_s3(mm)"] * (1 - d_s3),
                    "Design_a3(deg)": p_a3,
                }
            )
        except Exception as e:
            print(f"⚠️ 模型預測失敗: {e}。將使用原始設計進行模擬。")
            prediction_info = {
                "pred_delta_s2": 0,
                "pred_delta_s3": 0,
                "pred_dip_a3": a3,
            }

    # 捕獲將要用於模擬的最終幾何尺寸
    sim_geometry_info = {
        "predicted_s2": sim_design["Design_s2(mm)"],
        "predicted_s3": sim_design["Design_s3(mm)"],
        "predicted_a3": sim_design["Design_a3(deg)"],
    }

    sim_result = run_simulation(
        sim_design["Design_s1(mm)"],
        sim_design["Design_s2(mm)"],
        sim_design["Design_s3(mm)"],
        sim_design["Design_a1(deg)"],
        sim_design["Design_a2(deg)"],
        sim_design["Design_a3(deg)"],
        loop_num,
        individual,
    )

    if sim_result is None:
        return None

    fitness, efficiency, process_score, angle_eff_list = (
        sim_result[0],
        sim_result[1],
        sim_result[2],
        sim_result[3] if len(sim_result) > 3 else [],
    )
    angle_efficiencies_dict = {
        angle: eff for angle, eff in zip(range(10, 91, 10), angle_eff_list)
    }

    # 【修改】回傳的元組現在包含 7 個元素
    return (
        fitness,
        efficiency,
        process_score,
        angle_efficiencies_dict,
        {"s2": s2, "s3": s3, "a3": a3},
        prediction_info,
        sim_geometry_info,  # 新增第 7 個元素
    )


# --- (新增) 純突變函式 ---
def mutate_individual(parent_gene, parent_sigma, adaptation_mode):
    """
    從單一父代透過突變產生一個子代。
    支援適應性或固定的突變強度。
    """
    child_genes = np.copy(parent_gene)
    child_sigmas = np.copy(parent_sigma)

    if adaptation_mode == "adaptive":
        # 適應性突變：同時演化 sigma 值
        N = np.random.normal(0, 1)
        N_i = np.random.normal(0, 1, config.N_VARS)
        child_sigmas *= np.exp(config.TAU_PRIME * N + config.TAU * N_i)

    # 使用新的 (或固定的) sigma 進行突變
    child_genes += child_sigmas * np.random.normal(0, 1, config.N_VARS)

    # 確保基因在邊界範圍內
    child_genes[0] = np.clip(child_genes[0], *config.SIDE_BOUND)
    child_genes[1] = np.clip(child_genes[1], *config.SIDE_BOUND)
    child_genes[2] = np.clip(child_genes[2], *config.ANGLE_BOUND)

    return child_genes, child_sigmas


def apply_diversity_penalty(
    genes_array: np.ndarray, original_fitness_list: list
) -> list:
    """
    根據基因的相似度對 fitness 進行懲罰，以維持族群多樣性。
    """
    num_individuals = len(genes_array)
    if num_individuals <= 1:
        return original_fitness_list

    # 1. 特徵標準化
    normalized_genes = np.copy(genes_array).astype(float)
    min_vals = np.min(normalized_genes, axis=0)
    max_vals = np.max(normalized_genes, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized_genes = (normalized_genes - min_vals) / range_vals

    # 2. 計算距離矩陣
    distances = np.zeros((num_individuals, num_individuals))
    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            dist = np.linalg.norm(normalized_genes[i] - normalized_genes[j])
            distances[i, j] = distances[j, i] = dist

    avg_distances = np.mean(distances, axis=1)

    # 3. 計算懲罰並應用
    adjusted_fitness = []
    max_avg_dist = np.max(avg_distances) if np.max(avg_distances) > 0 else 1.0
    for i in range(num_individuals):
        uniqueness_score = avg_distances[i] / max_avg_dist
        penalty = (1.0 - uniqueness_score) * config.DIVERSITY_PENALTY_FACTOR

        # 只對正值的 fitness 進行懲罰
        if original_fitness_list[i] > 0:
            final_fitness = original_fitness_list[i] * (1.0 - penalty)
        else:
            final_fitness = original_fitness_list[i]  # 負值 fitness 不變

        adjusted_fitness.append(final_fitness)

    return adjusted_fitness


async def main_async(args: argparse.Namespace):
    """The main asynchronous function for the evolutionary strategy optimization."""
    models_ok, summary, details_df = await initialize_and_train_models(args)
    if not models_ok and not args.no_compensation:
        return

    if args.save_report:
        try:
            with pd.ExcelWriter(args.save_report) as writer:
                if summary:
                    pd.DataFrame(summary).T.to_excel(writer, "Training_Error_Summary")
                if details_df is not None:
                    details_df.to_excel(writer, "Training_Error_Details", index=False)
                if model_s2:
                    model_s2.get_coefficients_df().to_excel(writer, "model_s2_coeffs")
                if model_s3:
                    model_s3.get_coefficients_df().to_excel(writer, "model_s3_coeffs")
                if model_ang:
                    model_ang.get_coefficients_df().to_excel(writer, "model_ang_coeffs")
            print(f"✅ Report saved to '{args.save_report}'.")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")

    if args.report_only:
        print("\n--- '--report-only' mode, execution finished. ---")
        return

    print("\n--- Attempting to close any existing TracePro instances... ---")
    try:
        result = os.system("taskkill /F /IM TracePro.exe >nul 2>&1")
        if result == 0:
            print("✅ Successfully closed existing TracePro process(es).")
        else:
            print("🟡 No existing TracePro processes found to close.")
        time.sleep(0.1)
    except Exception as e:
        print(f"⚠️ Could not execute taskkill command: {e}")

    output_dir = os.path.join(BASE_DIR, "GA_population")
    os.makedirs(output_dir, exist_ok=True)
    utils.set_output_dir(output_dir)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # --- **MODIFIED RESUME LOGIC** ---
    start_gen, pop_genes, pop_sigmas, evaluated_results, unevaluated_tasks = (
        utils.resume_from_log()
    )

    if start_gen == 1 and not unevaluated_tasks:
        print("--- Starting a fresh evolution run. ---")
        pop_genes = np.random.rand(config.POP_SIZE, config.N_VARS)
        pop_genes[:, :2] = (
            pop_genes[:, :2] * (config.SIDE_BOUND[1] - config.SIDE_BOUND[0])
            + config.SIDE_BOUND[0]
        )
        pop_genes[:, 2] = (
            pop_genes[:, 2] * (config.ANGLE_BOUND[1] - config.ANGLE_BOUND[0])
            + config.ANGLE_BOUND[0]
        )
        pop_sigmas = np.array(
            [config.VAR_RANGES * config.INITIAL_SIGMA_FACTOR] * config.POP_SIZE
        )

    print(f"\n--- Starting Evolution from Generation {start_gen} ---")
    copy_scm_to_all_folders(output_dir)

    for gen in range(start_gen, config.N_GENERATIONS + 1):
        if utils.graceful_stop_event.is_set():
            break
        print(f"\n--- Generation {gen}/{config.N_GENERATIONS} ---")

        # This will hold all results for the current generation
        all_gen_results = list(evaluated_results)

        # --- **MODIFIED EXECUTION FLOW** ---
        # If there are unevaluated tasks from a previous run, execute them first.
        # 1. 評估父代 (如果需要)
        parent_eval = []
        if not unevaluated_tasks:  # 只有在新的一代才需要評估父代
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        evaluate_individual,
                        (i + 1, pop_genes[i], pop_sigmas[i], "parent_old"),
                        gen,
                        (-1, -1),
                    )
                    for i in range(config.POP_SIZE)
                ]
                parent_eval = [f.result() for f in futures if f.result() is not None]
        else:  # 如果是從中斷處恢復，父代評估結果已從日誌讀取
            parent_eval = list(evaluated_results)
            # 執行未完成的任務
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = [
                    executor.submit(evaluate_individual, task[0], task[1], task[2])
                    for task in unevaluated_tasks
                ]
                newly_evaluated = [
                    f.result() for f in futures if f.result() is not None
                ]
                parent_eval.extend(newly_evaluated)

        # 2. 透過突變產生子代 (統一資料結構)
        children_genes, children_sigmas, children_parent_indices = [], [], []
        for _ in range(config.OFFSPRING_SIZE):
            parent_idx = random.randrange(config.POP_SIZE)
            child_gene, child_sigma = mutate_individual(
                pop_genes[parent_idx], pop_sigmas[parent_idx], args.mutation_adaptation
            )
            children_genes.append(child_gene)
            children_sigmas.append(child_sigma)
            children_parent_indices.append((parent_idx, -1))

        # 3. 評估子代
        offspring_eval = []
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            tasks = {
                executor.submit(
                    evaluate_individual,
                    (
                        i + 1 + config.POP_SIZE,
                        children_genes[i],
                        children_sigmas[i],
                        "child",
                    ),
                    gen,
                    children_parent_indices[i],
                ): i
                for i in range(config.OFFSPRING_SIZE)
            }
            # 確保結果與原始子代對應
            results_with_indices = [
                (tasks[future], future.result()) for future in tasks if future.result()
            ]
            # 排序以保持一致性
            results_with_indices.sort(key=lambda x: x[0])
            offspring_eval = [res for _, res in results_with_indices]
            # 同時過濾掉失敗評估的子代基因與標準差
            successful_indices = [idx for idx, _ in results_with_indices]
            successful_children_genes = [children_genes[i] for i in successful_indices]
            successful_children_sigmas = [
                children_sigmas[i] for i in successful_indices
            ]
            successful_children_parents = [
                children_parent_indices[i] for i in successful_indices
            ]

        # 4. 根據選擇策略合併族群
        print(
            f"策略: ({'μ+λ' if args.selection_strategy == 'plus' else 'μ,λ'}), 突變: {args.mutation_adaptation}"
        )

        if args.selection_strategy == "plus":
            combined_genes = np.vstack([pop_genes] + successful_children_genes)
            combined_sigmas = np.vstack([pop_sigmas] + successful_children_sigmas)
            combined_eval = parent_eval + offspring_eval
        else:
            combined_genes = np.array(successful_children_genes)
            combined_sigmas = np.array(successful_children_sigmas)
            combined_eval = offspring_eval

        if len(combined_eval) < config.POP_SIZE and args.selection_strategy == "comma":
            print(
                f"⚠️ 警告：(μ,λ)策略下，存活的子代數量 ({len(combined_eval)}) 少於下一代要求的父代數量 ({config.POP_SIZE})。"
            )
            if not combined_eval:
                print("❌ (μ,λ)策略下沒有任何子代存活，演化終止。")
                break

        # 1. 從合併後的族群中，提取出「原始的、未經修改的」fitness 列表
        original_fitness_all = [d[0] for d in combined_eval]

        # 2. **立即計算真實的最佳 fitness**，這個值將用於報告和檔名
        true_best_fitness = max(original_fitness_all, default=-9999)

        # 3. 準備一個「用於選擇」的 fitness 列表，它可能會被後續步驟修改
        fitness_for_selection = original_fitness_all

        # 4. 如果啟用多樣性控制，則計算調整後的 fitness，並用它來排序
        if args.diversity_control:
            print("--- 正在應用多樣性懲罰 ---")
            # 準備用於計算距離的基因
            if args.selection_strategy == "plus":
                genes_for_diversity = np.vstack([pop_genes] + successful_children_genes)
            else:  # comma strategy
                genes_for_diversity = np.array(successful_children_genes)

            # 使用調整後的 fitness 來進行選擇
            fitness_for_selection = apply_diversity_penalty(
                genes_for_diversity, original_fitness_all
            )

        # 5. **使用「用於選擇的 fitness」進行排序**，以決定誰能存活
        order = np.argsort(fitness_for_selection)[::-1]

        # 6. 根據排序結果，選出下一代父代
        survivor_indices = order[: config.POP_SIZE]
        next_pop_genes = np.array([combined_genes[i] for i in survivor_indices])
        next_pop_sigmas = np.array([combined_sigmas[i] for i in survivor_indices])
        next_pop_eval = [
            combined_eval[i] for i in survivor_indices
        ]  # 存活者的評估結果仍然是原始的

        # 6. 記錄日誌
        current_rows = []
        # 記錄被評估的父代
        for i in range(len(parent_eval)):
            current_rows.append(
                utils.create_log_row(
                    pop_genes[i],
                    pop_sigmas[i],
                    parent_eval[i],
                    gen,
                    "parent_old",
                    (-1, -1),
                )
            )
        # 記錄被評估的子代
        for i in range(len(offspring_eval)):
            current_rows.append(
                utils.create_log_row(
                    successful_children_genes[i],
                    successful_children_sigmas[i],
                    offspring_eval[i],
                    gen,
                    "child",
                    successful_children_parents[i],
                )
            )
        # 記錄被選為下一代父代的個體
        for i in range(len(next_pop_genes)):
            current_rows.append(
                utils.create_log_row(
                    next_pop_genes[i],
                    next_pop_sigmas[i],
                    next_pop_eval[i],
                    gen,
                    "parent",
                    (-1, -1),
                )
            )

        # 8. **使用第 2 步計算的「真實最佳 fitness」來生成檔名和輸出訊息**
        out_filename = f"fitness_gen{gen}_max{true_best_fitness:.2f}.csv"
        log_filepath = os.path.join(config.LOG_DIR, out_filename)
        utils.save_generation_log(current_rows, log_filepath)

        print(f"★ 第 {gen} 代完成。最佳 Fitness: {true_best_fitness:.4f}")

        # 【邏輯修正結束】

        # 更新族群以進行下一代
        pop_genes = next_pop_genes
        pop_sigmas = next_pop_sigmas
        evaluated_results, unevaluated_tasks = [], []

        gc.collect()
    print("\n--- 演化過程已結束。 ---")
