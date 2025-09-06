import os
import re
import gc
import csv
import time
import math
import random
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import argparse

import numpy as np
import pandas as pd

# 匯入計算誤差所需的函式
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 外部應用程式模組
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_ES import evaluate_fitness

# 內部模組
import config

# 【修改重點】匯入 ModelHuber 以及與 model_main.py 同步設定的工廠函式
from models import (
    ModelHuber,
    FEATURES,
    create_length_model_from_main_defaults,
    create_angle_model_from_main_defaults,
)
import utils

# === 全域變數 ===
# 模型物件
model_s2: ModelHuber = None
model_s3: ModelHuber = None
model_ang: ModelHuber = None
# 補償模型開關
USE_COMPENSATION_MODEL = True
# 執行緒鎖
autocad_lock = threading.Lock()
tracepro_lock = threading.Lock()


# 【保留之原始邏輯】使用已訓練的模型預測收縮率與角度
def predict_shrinkage_and_angle(design_params):
    """使用已訓練的模型預測收縮率與角度"""
    if model_s2 is None or model_s3 is None or model_ang is None:
        raise RuntimeError("模型尚未初始化。")
    df_input = pd.DataFrame([design_params])
    # 將欄位名稱對應到模型所需的格式
    df_features = df_input.rename(
        columns={
            "s1": "Design_s1(mm)",
            "s2": "Design_s2(mm)",
            "s3": "Design_s3(mm)",
            "a1": "Design_a1(deg)",
            "a2": "Design_a2(deg)",
            "a3": "Design_a3(deg)",
        }
    )

    pred_delta_s2 = model_s2.predict(df_features)[0]
    pred_delta_s3 = model_s3.predict(df_features)[0]
    pred_angle_a3 = model_ang.predict(df_features)[0]

    return pred_delta_s2, pred_delta_s3, pred_angle_a3


# 【新增功能】計算總結的訓練誤差
def _calculate_summary_training_errors(
    model: ModelHuber, df: pd.DataFrame, target_name: str
) -> dict:
    """
    計算給定模型在訓練集上的總結誤差指標 (R2, RMSE, MAE)。
    """
    y_true = df[target_name].to_numpy()
    y_pred = model.predict(df)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"R-squared": r2, "RMSE": rmse, "MAE": mae}


# 【新增功能】計算每筆資料的詳細誤差並回傳 DataFrame
def _calculate_detailed_training_errors(
    model_s2: ModelHuber,
    model_s3: ModelHuber,
    model_ang: ModelHuber,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    計算每筆訓練資料的詳細預測誤差，並將結果整理成 DataFrame。
    """
    df_details = df.copy()

    # 1. 進行預測
    pred_delta_s2 = model_s2.predict(df_details)
    pred_delta_s3 = model_s3.predict(df_details)
    pred_dip_a3 = model_ang.predict(df_details)

    # 2. 建立結果 DataFrame，從原始特徵開始
    results_df = df_details[FEATURES].copy()

    # 3. 計算並加入 DIP (mm/deg) 的實際值、預測值和誤差
    # S2
    results_df["Actual_DIP_s2(mm)"] = df_details["Design_s2(mm)"] * (
        1 - df_details["delta_s2"]
    )
    results_df["Predicted_DIP_s2(mm)"] = df_details["Design_s2(mm)"] * (
        1 - pred_delta_s2
    )
    results_df["Error_s2(mm)"] = (
        results_df["Predicted_DIP_s2(mm)"] - results_df["Actual_DIP_s2(mm)"]
    )

    # S3
    results_df["Actual_DIP_s3(mm)"] = df_details["Design_s3(mm)"] * (
        1 - df_details["delta_s3"]
    )
    results_df["Predicted_DIP_s3(mm)"] = df_details["Design_s3(mm)"] * (
        1 - pred_delta_s3
    )
    results_df["Error_s3(mm)"] = (
        results_df["Predicted_DIP_s3(mm)"] - results_df["Actual_DIP_s3(mm)"]
    )

    # A3
    results_df["Actual_DIP_a3(deg)"] = df_details["DIP_a3(deg)"]
    results_df["Predicted_DIP_a3(deg)"] = pred_dip_a3
    results_df["Error_a3(deg)"] = pred_dip_a3 - df_details["DIP_a3(deg)"]

    # 4. 加入 delta (%) 的實際值、預測值和誤差
    results_df["Actual_delta_s2"] = df_details["delta_s2"]
    results_df["Predicted_delta_s2"] = pred_delta_s2
    results_df["Error_delta_s2"] = pred_delta_s2 - df_details["delta_s2"]

    results_df["Actual_delta_s3"] = df_details["delta_s3"]
    results_df["Predicted_delta_s3"] = pred_delta_s3
    results_df["Error_delta_s3"] = pred_delta_s3 - df_details["delta_s3"]

    return results_df


# 【修改重點】模型初始化與訓練函式
async def initialize_and_train_models(args: argparse.Namespace):
    """
    根據命令列參數初始化並訓練三個補償模型。
    並在訓練後計算總結誤差與詳細誤差。
    """
    global model_s2, model_s3, model_ang, USE_COMPENSATION_MODEL

    if args.no_compensation:
        USE_COMPENSATION_MODEL = False
        print("\n🟡 已禁用收縮補償模型。")
        return True, None, None

    USE_COMPENSATION_MODEL = True
    print("\n--- 正在初始化並訓練補償模型 ---")

    try:
        df_train = pd.read_excel(config.TRAIN_DATA_PATH)
        print(f"✅ 成功讀取訓練資料 '{config.TRAIN_DATA_PATH}'。")

        # 【錯誤修復】增加處理缺失值 (NaN) 的步驟
        initial_rows = len(df_train)
        required_cols = FEATURES + ["delta_s2", "delta_s3", "DIP_a3(deg)"]
        missing_cols = [col for col in required_cols if col not in df_train.columns]
        if missing_cols:
            print(f"❌ 訓練資料中缺少必要的欄位: {', '.join(missing_cols)}")
            USE_COMPENSATION_MODEL = False
            return False, None, None

        df_train.dropna(subset=required_cols, inplace=True)
        dropped_rows = initial_rows - len(df_train)
        if dropped_rows > 0:
            print(
                f"⚠️ 已從訓練資料中移除 {dropped_rows} 行，因為它們在必要欄位中包含缺失值 (NaN)。"
            )

        if len(df_train) == 0:
            print("❌ 處理缺失值後，沒有剩餘的有效訓練資料。")
            USE_COMPENSATION_MODEL = False
            return False, None, None

    except FileNotFoundError:
        print(f"❌ 錯誤：找不到訓練資料檔案 '{config.TRAIN_DATA_PATH}'。")
        USE_COMPENSATION_MODEL = False
        return False, None, None
    except Exception as e:
        print(f"❌ 讀取或處理訓練資料時發生錯誤: {e}")
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

    summary_reports = {}
    detailed_error_df = None
    try:
        print("正在訓練 model_s2...")
        model_s2.fit(df_train, "delta_s2")
        print("正在訓練 model_s3...")
        model_s3.fit(df_train, "delta_s3")
        print("正在訓練 model_ang...")
        model_ang.fit(df_train, "DIP_a3(deg)")
        print("\n✅ 所有模型訓練完成。")

        # 計算總結誤差
        summary_reports["model_s2"] = _calculate_summary_training_errors(
            model_s2, df_train, "delta_s2"
        )
        summary_reports["model_s3"] = _calculate_summary_training_errors(
            model_s3, df_train, "delta_s3"
        )
        summary_reports["model_ang"] = _calculate_summary_training_errors(
            model_ang, df_train, "DIP_a3(deg)"
        )

        # 計算詳細的逐筆誤差
        detailed_error_df = _calculate_detailed_training_errors(
            model_s2, model_s3, model_ang, df_train
        )

        # 顯示總結誤差報告
        print("\n--- 模型訓練誤差總結 ---")
        for model_name, errors in summary_reports.items():
            print(f"  - {model_name}:")
            for metric, value in errors.items():
                print(f"    - {metric}: {value:.6f}")
        print("--------------------------")

        return True, summary_reports, detailed_error_df

    except Exception as e:
        print(f"❌ 模型訓練過程中發生錯誤: {e}")
        USE_COMPENSATION_MODEL = False
        return False, None, None


# 【保留之原始邏輯】評估單一個體（包含補償與模擬）
def evaluate_individual(individual_data, generation, parent_indices):
    loop_num, individual, sigma, role = individual_data
    start_time = time.time()
    s1, s2, a1 = individual
    a1_int = int(round(a1))

    design_params_for_prediction = {
        "s1": s1,
        "s2": s2,
        "s3": utils.s3_func(s1, s2),
        "a1": a1_int,
        "a2": utils.a2_func(a1_int),
        "a3": utils.a3_func(a1_int),
    }

    if USE_COMPENSATION_MODEL:
        try:
            d_s2, d_s3, p_a3 = predict_shrinkage_and_angle(design_params_for_prediction)
            compensated_s2 = s2 / (1 - d_s2)
            compensated_s3 = utils.s3_func(s1, compensated_s2)

            final_s1, final_s2, final_a1 = s1, compensated_s2, a1_int

        except Exception as e:
            print(f"⚠️ 補償模型預測失敗: {e}。將使用原始設計參數。")
            final_s1, final_s2, final_a1 = s1, s2, a1_int
    else:
        final_s1, final_s2, final_a1 = s1, s2, a1_int

    simulation_result = run_simulation(final_s1, final_s2, final_a1)

    if simulation_result is None:
        return None

    fitness, efficiency, process_score, angle_efficiencies = simulation_result

    end_time = time.time()
    print(
        f"  (Gen {generation}) 個體 {loop_num} 評估完成 in {end_time - start_time:.2f}s -> Fitness: {fitness:.2f}"
    )

    return (
        fitness,
        efficiency,
        process_score,
        angle_efficiencies,
        {"s1": final_s1, "s2": final_s2, "a1": final_a1},
    )


# 【保留之原始邏輯】執行外部程式模擬
def run_simulation(s1, s2, a1):
    try:
        s3 = utils.s3_func(s1, s2)
        a2 = utils.a2_func(a1)
        a3 = utils.a3_func(a1)

        with autocad_lock:
            if utils.immediate_stop_event.is_set():
                return None
            Build_model(s1, s2, s3, a1, a2, a3)

        with tracepro_lock:
            if utils.immediate_stop_event.is_set():
                return None
            tracepro_fast(a1)

        return evaluate_fitness()

    except Exception as e:
        print(f"❌ 模擬執行中發生嚴重錯誤: {e}")
        utils.send_error(f"模擬執行中發生嚴重錯誤: {e}")
        return None


# 【保留之原始邏輯】交配與變異
def crossover_and_mutate(parents_genes, parents_sigmas):
    parent1_idx, parent2_idx = random.sample(range(len(parents_genes)), 2)
    p1_genes, p2_genes = parents_genes[parent1_idx], parents_genes[parent2_idx]
    p1_sigmas, p2_sigmas = parents_sigmas[parent1_idx], parents_sigmas[parent2_idx]

    child_genes = (p1_genes + p2_genes) / 2
    child_sigmas = (p1_sigmas + p2_sigmas) / 2

    N = np.random.normal(0, 1)
    N_i = np.random.normal(0, 1, config.N_VARS)
    child_sigmas = child_sigmas * np.exp(config.TAU_PRIME * N + config.TAU * N_i)
    child_genes = child_genes + child_sigmas * np.random.normal(0, 1, config.N_VARS)

    child_genes[0] = np.clip(child_genes[0], *config.SIDE_BOUND)
    child_genes[1] = np.clip(child_genes[1], *config.SIDE_BOUND)
    child_genes[2] = np.clip(child_genes[2], *config.ANGLE_BOUND)

    return child_genes, child_sigmas, (parent1_idx, parent2_idx)


# 【修改重點】主函式，整合了報告儲存和僅報告模式
async def main_async(args: argparse.Namespace):
    """演化策略優化的主非同步函式"""

    # --- 1. 初始化模型 ---
    models_ok, summary_reports, detailed_error_df = await initialize_and_train_models(
        args
    )
    if not models_ok and not args.no_compensation:
        print("❌ 模型初始化失敗，無法繼續執行。")
        return

    # --- 2. 儲存報告與僅報告模式 ---
    if args.save_report:
        print(f"\n--- 正在儲存報告至 '{args.save_report}' ---")
        try:
            with pd.ExcelWriter(args.save_report) as writer:
                # 儲存模型係數
                if model_s2:
                    model_s2.get_coefficients_df().to_excel(
                        writer, sheet_name="model_s2_coeffs"
                    )
                if model_s3:
                    model_s3.get_coefficients_df().to_excel(
                        writer, sheet_name="model_s3_coeffs"
                    )
                if model_ang:
                    model_ang.get_coefficients_df().to_excel(
                        writer, sheet_name="model_ang_coeffs"
                    )

                # 儲存訓練誤差總結
                if summary_reports:
                    pd.DataFrame(summary_reports).T.to_excel(
                        writer, sheet_name="Training_Error_Summary"
                    )

                # 【新增】儲存詳細的逐筆誤差
                if detailed_error_df is not None:
                    detailed_error_df.to_excel(
                        writer, sheet_name="Training_Error_Details", index=False
                    )

                # 儲存執行參數
                pd.DataFrame([vars(args)]).to_excel(
                    writer, sheet_name="Run_Parameters", index=False
                )
            print("✅ 報告儲存成功。")
        except Exception as e:
            print(f"❌ 儲存報告時發生錯誤: {e}")

    if args.report_only:
        print("\n--- '--report-only' 模式，執行結束。 ---")
        return

    # --- 3. 【保留之原始邏輯】初始化族群 ---
    start_gen, pop_genes, pop_sigmas, parent_eval, resume_df = utils.resume_from_log()
    if start_gen == 1:
        print("\n--- 初始化初始族群 ---")
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

    # --- 4. 【保留之原始邏輯】執行演化迴圈 ---
    output_dir = utils.create_output_directory()
    print(f"\n--- 開始從第 {start_gen} 代進行演化 ---")

    for gen in range(start_gen, config.N_GENERATIONS + 1):
        if utils.immediate_stop_event.is_set() or utils.graceful_stop_event.is_set():
            print(f"\n🛑 在第 {gen} 代開始前偵測到停止訊號。")
            break

        print(f"\n--- 第 {gen}/{config.N_GENERATIONS} 代 ---")
        current_rows = []

        if gen == start_gen and start_gen > 1:
            print("從日誌中恢復父代評估。")
        else:
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        evaluate_individual,
                        (i, pop_genes[i], pop_sigmas[i], "parent"),
                        gen,
                        (-1, -1),
                    )
                    for i in range(config.POP_SIZE)
                ]
                parent_eval = [f.result() for f in futures if f.result() is not None]

        if not parent_eval:
            print("❌ 父代評估失敗，無法繼續。")
            break

        children_genes, children_sigmas, parent_indices_list = [], [], []
        for _ in range(config.OFFSPRING_SIZE):
            child_g, child_s, p_indices = crossover_and_mutate(pop_genes, pop_sigmas)
            children_genes.append(child_g)
            children_sigmas.append(child_s)
            parent_indices_list.append(p_indices)

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    evaluate_individual,
                    (i, children_genes[i], children_sigmas[i], "child"),
                    gen,
                    parent_indices_list[i],
                )
                for i in range(config.OFFSPRING_SIZE)
            ]
            offspring_eval = [f.result() for f in futures]

        for i, res in enumerate(parent_eval):
            current_rows.append(
                utils.create_log_row(
                    pop_genes[i], pop_sigmas[i], res, gen, "parent_old", (-1, -1)
                )
            )

        for i, res in enumerate(offspring_eval):
            if res:
                current_rows.append(
                    utils.create_log_row(
                        children_genes[i],
                        children_sigmas[i],
                        res,
                        gen,
                        "child",
                        parent_indices_list[i],
                    )
                )

        combined_genes = np.vstack([pop_genes, np.array(children_genes)])
        combined_sigmas = np.vstack([pop_sigmas, np.array(children_sigmas)])
        combined_eval = parent_eval + [
            item for item in offspring_eval if item is not None
        ]

        if not combined_eval:
            print("❌ 所有個體評估均失敗，無法進行選擇。")
            break

        fitness_all = [d[0] for d in combined_eval]
        order = np.argsort(fitness_all)[::-1]

        pop_genes = np.array([combined_genes[i] for i in order[: config.POP_SIZE]])
        pop_sigmas = np.array([combined_sigmas[i] for i in order[: config.POP_SIZE]])
        parent_eval = [combined_eval[i] for i in order[: config.POP_SIZE]]

        for i in range(config.POP_SIZE):
            current_rows.append(
                utils.create_log_row(
                    pop_genes[i], pop_sigmas[i], parent_eval[i], gen, "parent", (-1, -1)
                )
            )

        best_fitness = max((f for f in fitness_all if f > -9990), default=-9999)
        out_filename = f"fitness_gen{gen}_max{best_fitness:.2f}.csv"
        utils.save_generation_log(current_rows, os.path.join(output_dir, out_filename))

    print("\n--- 演化流程結束 ---")
