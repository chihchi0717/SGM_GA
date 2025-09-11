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
evaluation_cache_lock = threading.Lock()
evaluation_cache = {}


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
                print(f"✅ Build_model succeeded for individual P{loop_num}.")
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
            print(f"✅ TracePro simulation completed for individual P{loop_num}.")

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

    gene_key = tuple(round(v, 2) for v in individual)
    with evaluation_cache_lock:
        cached_result = evaluation_cache.get(gene_key)
    if cached_result is not None:
        print(f"參數 {gene_key} 已評估，直接複製紀錄。")
        return cached_result

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

    # --- 模擬執行與重試機制 ---
    sim_result = None
    attempt = 0
    while sim_result is None:
        attempt += 1
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
            if attempt >= config.SIM_MAX_RETRIES:
                print(
                    f"❌ Simulation for individual P{loop_num} failed after {attempt} attempts. Skipping this individual."
                )
                return None
            print(
                f"⚠️ Simulation for individual P{loop_num} failed on attempt {attempt}. Retrying..."
            )
            time.sleep(1)

    fitness, efficiency, process_score, angle_eff_list = (
        sim_result[0],
        sim_result[1],
        sim_result[2],
        sim_result[3] if len(sim_result) > 3 else [],
    )
    angle_efficiencies_dict = {
        angle: eff for angle, eff in zip(range(10, 91, 10), angle_eff_list)
    }

    eval_result = (
        fitness,
        efficiency,
        process_score,
        angle_efficiencies_dict,
        {"s2": s2, "s3": s3, "a3": a3},
        prediction_info,
        sim_geometry_info,
    )
    with evaluation_cache_lock:
        evaluation_cache[gene_key] = eval_result
    return eval_result


def recombine_global(parents_genes, parents_sigmas):
    """
    全域重組：
    - 基因值 x ：全域離散重組 (每一維從隨機父代取值)
    - 步長 σ ：全域中介重組 (取所有父代的平均)
    """
    m, n = parents_genes.shape
    # x: global discrete
    child_gene = np.array([parents_genes[np.random.randint(m), i] for i in range(n)])
    # σ: global intermediary
    child_sigma = np.mean(parents_sigmas, axis=0)
    return child_gene, child_sigma


def mutate_individual(parent_gene, parent_sigma, adaptation_mode):
    """
    Uncorrelated mutation with n σ’s (self-adaptation).
    - σ 先突變再用於 x
    - 有 boundary rule
    """
    n = config.N_VARS
    child_genes = np.copy(parent_gene)
    child_sigmas = np.copy(parent_sigma)

    if adaptation_mode == "adaptive":
        # 自動設定學習率
        tau_prime = 1.0 / np.sqrt(2.0 * n)
        tau = 1.0 / np.sqrt(2.0 * np.sqrt(n))

        # 先突變 σ
        N0 = np.random.normal(0, 1)
        Ni = np.random.normal(0, 1, n)
        child_sigmas = child_sigmas * np.exp(tau_prime * N0 + tau * Ni)

        # Boundary rule
        sigma_min = getattr(config, "SIGMA_MIN", 1e-6)
        sigma_max = getattr(config, "SIGMA_MAX", np.array(config.VAR_RANGES, float))
        child_sigmas = np.clip(child_sigmas, sigma_min, sigma_max)

    # 再突變 x
    child_genes = child_genes + child_sigmas * np.random.normal(0, 1, n)

    # 確保基因在設計範圍內
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

    output_dir = os.path.join(BASE_DIR, "GA_population")
    os.makedirs(output_dir, exist_ok=True)
    utils.set_output_dir(output_dir)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    global evaluation_cache
    start_gen, pop_genes, pop_sigmas, prev_parent_eval, evaluation_cache = (
        utils.resume_from_log()
    )

    if start_gen == 1:
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

    print(f"\n--- 從第 {start_gen} 代開始演化 ---")
    copy_scm_to_all_folders(utils.get_output_dir())

    for gen in range(start_gen, config.N_GENERATIONS + 1):
        if utils.graceful_stop_event.is_set():
            break
        print(f"\n--- 第 {gen}/{config.N_GENERATIONS} 代 ---")

        # 1. 評估父代，並只保留成功的個體
        if prev_parent_eval is not None:
            print("跳過父代評估，直接沿用上一代結果。")
            parent_eval = prev_parent_eval
            successful_parent_genes = pop_genes
            successful_parent_sigmas = pop_sigmas
        else:
            parent_eval = []
            successful_parent_genes = np.array([])
            successful_parent_sigmas = np.array([])

            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                future_to_idx = {
                    executor.submit(
                        evaluate_individual,
                        (i + 1, pop_genes[i], pop_sigmas[i], "parent_old"),
                        gen,
                        (-1, -1),
                    ): i
                    for i in range(len(pop_genes))
                }

                eval_map = {}
                for future in future_to_idx:
                    try:
                        result = future.result()
                        if result:
                            eval_map[future_to_idx[future]] = result
                    except Exception as exc:
                        print(f"個體評估產生例外: {exc}")

                successful_indices = sorted(eval_map.keys())
                if successful_indices:
                    parent_eval = [eval_map[i] for i in successful_indices]
                    successful_parent_genes = pop_genes[successful_indices]
                    successful_parent_sigmas = pop_sigmas[successful_indices]

        prev_parent_eval = None

        if len(successful_parent_genes) == 0:
            print(f"❌ 第 {gen} 代所有父代均評估失敗，無法產生子代。演化終止。")
            break

        # 2. 從成功的父代中產生並評估子代
        children_data = []
        for _ in range(config.OFFSPRING_SIZE):
            parent_idx = random.randrange(len(successful_parent_genes))
            # 例如每次隨機選 3 個父代做重組
            num_parents = min(3, len(successful_parent_genes))
            selected_indices = np.random.choice(
                len(successful_parent_genes), num_parents, replace=False
            )
            parent_subset_genes = successful_parent_genes[selected_indices]
            parent_subset_sigmas = successful_parent_sigmas[selected_indices]

            recombined_gene, recombined_sigma = recombine_global(
                parent_subset_genes, parent_subset_sigmas
            )

            child_gene, child_sigma = mutate_individual(
                recombined_gene,
                recombined_sigma,
                args.mutation_adaptation,
            )

            children_data.append((child_gene, child_sigma, (parent_idx, -1)))

        offspring_results = []
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            future_to_data = {
                executor.submit(
                    evaluate_individual,
                    (i + 1 + config.POP_SIZE, c_data[0], c_data[1], "child"),
                    gen,
                    c_data[2],
                ): c_data
                for i, c_data in enumerate(children_data)
            }
            for future in future_to_data:
                try:
                    result = future.result()
                    if result:
                        offspring_results.append((future_to_data[future], result))
                except Exception as exc:
                    print(f"子代評估產生例外: {exc}")

        successful_children_data = [info for info, _ in offspring_results]
        offspring_eval = [res for _, res in offspring_results]

        # 3. 根據選擇策略合併族群
        if args.selection_strategy == "plus":
            combined_genes_list = list(successful_parent_genes) + [
                data[0] for data in successful_children_data
            ]
            combined_sigmas_list = list(successful_parent_sigmas) + [
                data[1] for data in successful_children_data
            ]
            combined_eval = parent_eval + offspring_eval
        else:  # 'comma'
            combined_genes_list = [data[0] for data in successful_children_data]
            combined_sigmas_list = [data[1] for data in successful_children_data]
            combined_eval = offspring_eval

        if not combined_eval:
            print("❌ 所有個體評估均失敗或沒有子代存活，無法進行選擇。演化終止。")
            break

        combined_genes = np.array(combined_genes_list)
        combined_sigmas = np.array(combined_sigmas_list)

        # 4. 應用多樣性懲罰並選擇
        original_fitness_all = [d[0] for d in combined_eval]
        true_best_fitness = max(original_fitness_all, default=-9999)

        fitness_for_selection = original_fitness_all
        if args.diversity_control and len(combined_genes) > 1:
            fitness_for_selection = apply_diversity_penalty(
                combined_genes, original_fitness_all
            )

        order = np.argsort(fitness_for_selection)[::-1]

        # 5. 選出下一代父代
        survivor_indices = order[: config.POP_SIZE]
        next_pop_genes = combined_genes[survivor_indices]
        next_pop_sigmas = combined_sigmas[survivor_indices]
        next_pop_eval = [combined_eval[i] for i in survivor_indices]

        # 6. 記錄日誌
        current_rows = []
        for i in range(len(parent_eval)):
            current_rows.append(
                utils.create_log_row(
                    successful_parent_genes[i],
                    successful_parent_sigmas[i],
                    parent_eval[i],
                    gen,
                    "parent_old",
                    (-1, -1),
                )
            )
        for i in range(len(offspring_eval)):
            current_rows.append(
                utils.create_log_row(
                    successful_children_data[i][0],
                    successful_children_data[i][1],
                    offspring_eval[i],
                    gen,
                    "child",
                    successful_children_data[i][2],
                )
            )

        # --- 【關鍵修正】 ---
        # 只有在成功評估了子代 (即發生了探索) 的情況下，才記錄 'parent' 角色，
        # 這才能標誌著一個世代的真正完成。
        generation_is_truly_complete = False
        if len(offspring_eval) == (config.OFFSPRING_SIZE):
            generation_is_truly_complete = True
        print(f"本代成功評估子代數: {len(offspring_eval)}")
        print(f"子代數: {config.OFFSPRING_SIZE}")
        if generation_is_truly_complete:
            for i in range(len(next_pop_eval)):
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
        else:
            print(f"⚠️ 第 {gen} 代未完成全部子代評估，在下次啟動時重新執行。")

        out_filename = f"fitness_gen{gen}_max{true_best_fitness:.2f}.csv"
        log_filepath = os.path.join(config.LOG_DIR, out_filename)
        utils.save_generation_log(current_rows, log_filepath)

        print(f"★ 第 {gen} 代完成。真實最佳 Fitness: {true_best_fitness:.4f}")

        # 更新族群以進行下一代
        pop_genes = next_pop_genes
        pop_sigmas = next_pop_sigmas
        prev_parent_eval = next_pop_eval

        gc.collect()

    print("\n--- 演化過程已結束。 ---")
