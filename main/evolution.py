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

import numpy as np
import pandas as pd

# 外部應用程式模組
from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_ES import evaluate_fitness

# 內部模組
import config
from models import ModelHuber, FEATURES
import utils
from utils import (
    immediate_stop_event,
    graceful_stop_event,
    create_log_row,
    save_generation_log,
    is_duplicate_history,
    safe_float,
)

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


def predict_shrinkage_and_angle(design_params):
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


def evaluate_individual(individual: np.ndarray, folder: str, idx: int):
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
            return aborted_fitness, design_params
        print(f"  [Worker {idx}] 取得 AutoCAD 鎖，開始建模...")
        for attempt in range(3):
            if immediate_stop_event.is_set():
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
        return aborted_fitness, design_params

    simulation_success = False
    with tracepro_lock:
        if immediate_stop_event.is_set():
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
    lo = np.array(
        [config.SIDE_BOUND[0], config.SIDE_BOUND[0], config.ANGLE_BOUND[0]], dtype=float
    )
    hi = np.array(
        [config.SIDE_BOUND[1], config.SIDE_BOUND[1], config.ANGLE_BOUND[1]], dtype=float
    )
    y = x.copy()
    for k in range(config.N_VARS):
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


def init_population():
    pop = np.zeros((config.POP_SIZE, config.N_VARS), dtype=float)
    pop[:, 0] = np.random.uniform(
        config.SIDE_BOUND[0], config.SIDE_BOUND[1], size=config.POP_SIZE
    )
    pop[:, 1] = np.random.uniform(
        config.SIDE_BOUND[0], config.SIDE_BOUND[1], size=config.POP_SIZE
    )
    pop[:, 2] = np.random.uniform(
        config.ANGLE_BOUND[0], config.ANGLE_BOUND[1], size=config.POP_SIZE
    )
    for i in range(config.POP_SIZE):
        pop[i] = reflect_bounds(pop[i])
    sigma0 = np.tile(
        (config.VAR_RANGES * config.INITIAL_SIGMA_FACTOR), (config.POP_SIZE, 1)
    )
    return pop, sigma0


def make_offspring(pop_genes, pop_sigmas):
    children_genes, children_sigmas, parent_pairs = [], [], []
    for _ in range(config.OFFSPRING_SIZE):
        p1, p2 = random.sample(range(config.POP_SIZE), 2)
        x1, x2, s1, s2 = pop_genes[p1], pop_genes[p2], pop_sigmas[p1], pop_sigmas[p2]
        alpha = np.random.rand()
        x_bar, s_bar = alpha * x1 + (1 - alpha) * x2, alpha * s1 + (1 - alpha) * s2
        noise = np.random.randn()
        new_sigma = s_bar * np.exp(
            config.TAU_PRIME * noise + config.TAU * np.random.randn(config.N_VARS)
        )
        new_sigma = np.maximum(new_sigma, config.SIGMA_MIN)
        child = reflect_bounds(x_bar + new_sigma * np.random.randn(config.N_VARS))
        children_genes.append(child)
        children_sigmas.append(new_sigma)
        parent_pairs.append((p1, p2))
    return children_genes, children_sigmas, parent_pairs


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
        df_train = pd.read_excel(config.TRAIN_DATA_PATH)
        print(f"✅ 成功讀取訓練資料 '{config.TRAIN_DATA_PATH}'。")
        model_s2.fit(df_train, "delta_s2")
        model_s3.fit(df_train, "delta_s3")
        model_ang.fit(df_train, "DIP_a3(deg)")
        print("✅ 所有模型訓練完成。")
    except Exception as e:
        print(f"❌ 模型訓練失敗: {e}")
        return
    if args.save_report:
        utils.save_model_report(
            {"model_s2": model_s2, "model_s3": model_s3, "model_ang": model_ang},
            args.save_report,
        )
        if args.report_only:
            print("報告已儲存，程式結束。")
            return

    print("---------------------------\n")
    utils.copy_scm_to_all_folders()
    utils.write_run_config(USE_COMPENSATION_MODEL)

    start_gen, pop_genes, pop_sigmas, parent_eval, incomplete_df = (
        utils.load_latest_state()
    )
    executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

    if start_gen == 1 and pop_genes is None:
        gen = 1
        print(
            f"\n{'='*18} GENERATION 1 (使用 {config.MAX_WORKERS} 個並行 workers) {'='*18}"
        )
        pop_genes, pop_sigmas = init_population()
        tasks = []
        initial_rows = []
        parent_eval = [None] * config.POP_SIZE

        if incomplete_df is not None:
            print(f"🌱 恢復未完成的第 1 代...")
            completed_parents = incomplete_df[incomplete_df["role"] == "parent"].copy()
            completed_parents["S1"] = completed_parents["S1"].apply(safe_float)
            completed_parents["S2"] = completed_parents["S2"].apply(safe_float)
            completed_parents["A1"] = completed_parents["A1"].apply(safe_float)

            for i in range(config.POP_SIZE):
                match = completed_parents[
                    np.isclose(completed_parents["S1"], pop_genes[i, 0])
                    & np.isclose(completed_parents["S2"], pop_genes[i, 1])
                    & np.isclose(completed_parents["A1"], pop_genes[i, 2])
                ]
                if (
                    not match.empty
                    and safe_float(match.iloc[0].get("fitness"), -10000) > -9990
                ):
                    row = match.iloc[0]
                    parent_eval[i] = (
                        safe_float(row["fitness"]),
                        safe_float(row["efficiency"]),
                        safe_float(row["process_score"]),
                        [safe_float(row.get(f"eff_{a}")) for a in range(10, 90, 10)],
                        [],
                    )
                    initial_rows.append(row.to_dict())
                else:
                    tasks.append(
                        evaluate_individual_async(
                            executor,
                            pop_genes[i],
                            os.path.join(config.SAVE_ROOT, f"P{i+1}"),
                            i + 1,
                        )
                    )
        else:
            print(f"🌱 全新開始第 1 代...")
            tasks = [
                evaluate_individual_async(
                    executor,
                    pop_genes[i],
                    os.path.join(config.SAVE_ROOT, f"P{i+1}"),
                    i + 1,
                )
                for i in range(config.POP_SIZE)
            ]

        if tasks:
            results = await asyncio.gather(*tasks)
            res_idx = 0
            for i in range(config.POP_SIZE):
                if parent_eval[i] is None:
                    eval_data, design_params = results[res_idx]
                    parent_eval[i] = eval_data
                    if not design_params.get("aborted"):
                        initial_rows.append(
                            create_log_row(
                                pop_genes[i],
                                pop_sigmas[i],
                                eval_data,
                                1,
                                "parent",
                                (-1, -1),
                                design_params,
                            )
                        )
                    res_idx += 1

        best = max((d[0] for d in parent_eval if d[0] > -9990), default=-9999)
        fn = f"fitness_gen1_max{best:.2f}.csv"
        save_generation_log(initial_rows, os.path.join(config.LOG_DIR, fn))
        print(f"★ 第 1 代完成，存為 {fn}")
        start_gen = 2
        incomplete_df = None

    for gen in range(start_gen, config.N_GENERATIONS + 1):
        if graceful_stop_event.is_set() or immediate_stop_event.is_set():
            print(f"在第 {gen} 代開始前偵測到停止訊號，準備結束程式。")
            break

        print(
            f"\n{'='*18} GENERATION {gen} (使用 {config.MAX_WORKERS} 個並行 workers) {'='*18}"
        )

        children_genes, children_sigmas, parent_pairs = make_offspring(
            pop_genes, pop_sigmas
        )
        tasks_to_run, indices_to_run = [], []
        offspring_eval = [None] * config.OFFSPRING_SIZE
        current_rows = []

        if incomplete_df is not None and gen == incomplete_df["generation"].iloc[0]:
            print(f"  -> 恢復第 {gen} 代的未完成工作...")
            offspring_rows = incomplete_df[
                incomplete_df["role"] == "offspring"
            ].reset_index()
            children_genes, children_sigmas, parent_pairs = make_offspring(
                pop_genes, pop_sigmas
            )

            for i in range(config.OFFSPRING_SIZE):
                match = offspring_rows[
                    np.isclose(offspring_rows["S1"].astype(float), children_genes[i][0])
                    & np.isclose(
                        offspring_rows["S2"].astype(float), children_genes[i][1]
                    )
                    & np.isclose(
                        offspring_rows["A1"].astype(float), children_genes[i][2]
                    )
                ]

                if (
                    not match.empty
                    and safe_float(match.iloc[0].get("fitness"), -10000) > -9990
                ):
                    row = match.iloc[0]
                    offspring_eval[i] = (
                        safe_float(row["fitness"]),
                        safe_float(row["efficiency"]),
                        safe_float(row["process_score"]),
                        [safe_float(row.get(f"eff_{a}")) for a in range(10, 90, 10)],
                        [],
                    )
                    current_rows.append(row.to_dict())
                else:
                    folder = os.path.join(
                        config.SAVE_ROOT, f"P{config.POP_SIZE + i + 1}"
                    )
                    tasks_to_run.append(
                        evaluate_individual_async(
                            executor, children_genes[i], folder, config.POP_SIZE + i + 1
                        )
                    )
                    indices_to_run.append(i)
            incomplete_df = None
        else:
            history_rows = []
            try:
                files = sorted(
                    [
                        f
                        for f in os.listdir(config.LOG_DIR)
                        if f.startswith("fitness_gen")
                    ],
                    key=lambda x: int(re.search(r"gen(\d+)", x).group(1)),
                )
                for f in files:
                    history_rows.extend(
                        list(
                            csv.DictReader(
                                open(
                                    os.path.join(config.LOG_DIR, f),
                                    "r",
                                    encoding="utf-8",
                                )
                            )
                        )
                    )
            except Exception as e:
                print(f"⚠️ 讀歷史日誌失敗：{e}")

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
                    folder = os.path.join(
                        config.SAVE_ROOT, f"P{config.POP_SIZE + i + 1}"
                    )
                    tasks_to_run.append(
                        evaluate_individual_async(
                            executor, child, folder, config.POP_SIZE + i + 1
                        )
                    )
                    indices_to_run.append(i)

        if tasks_to_run:
            print(f"  -> 正在並行評估 {len(tasks_to_run)} 個新子代...")
            results = await asyncio.gather(*tasks_to_run)
            res_idx = 0
            for i in range(config.OFFSPRING_SIZE):
                if offspring_eval[i] is None:
                    eval_data, design_params = results[res_idx]
                    offspring_eval[i] = eval_data
                    if not design_params.get("aborted"):
                        current_rows.append(
                            create_log_row(
                                children_genes[i],
                                children_sigmas[i],
                                eval_data,
                                gen,
                                "offspring",
                                parent_pairs[i],
                                design_params,
                            )
                        )
                    res_idx += 1

        if any(v is None for v in offspring_eval):
            raise RuntimeError(f"第 {gen} 代有子代未評估")

        for i in range(config.POP_SIZE):
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

        pop_genes = np.array([combined_genes[i] for i in order[: config.POP_SIZE]])
        pop_sigmas = np.array([combined_sigmas[i] for i in order[: config.POP_SIZE]])
        parent_eval = [combined_eval[i] for i in order[: config.POP_SIZE]]

        for i in range(config.POP_SIZE):
            current_rows.append(
                create_log_row(
                    pop_genes[i], pop_sigmas[i], parent_eval[i], gen, "parent", (-1, -1)
                )
            )

        best_fitness = max((f for f in fitness_all if f > -9990), default=-9999)
        out = f"fitness_gen{gen}_max{best_fitness:.2f}.csv"
        save_generation_log(current_rows, os.path.join(config.LOG_DIR, out))
        print(
            f"★ Generation {gen} 完成，最佳 fitness = {best_fitness:.4f}，日誌：{out}"
        )
        gc.collect()

    executor.shutdown()
    if graceful_stop_event.is_set() or immediate_stop_event.is_set():
        print("\n🛑 程式已由使用者安全停止。")
    else:
        print("\n🎉 所有世代執行完成！")
