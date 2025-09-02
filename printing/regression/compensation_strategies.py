# -*- coding: utf-8 -*-
"""
compensation_strategies.py
- 包含不同的迭代補償演算法。
  - compensate_with_jacobian: 使用 Jacobian 矩陣進行多變數優化。
  - compensate_without_jacobian: 使用簡化的直接誤差回饋法。
"""
import numpy as np
import pandas as pd
import random
from .compensation_utils import (
    FEATURES,
    apply_geometric_constraints,
    precomp_in_deviation_space,
)


def compensate_with_jacobian(
    model_len, model_ang, target_design, num_steps=1000, step_size=0.0001
):
    """使用 Jacobian 矩陣的迭代補償策略。"""
    print("\n=== Running Compensation Strategy: with Jacobian ===")
    current_design = target_design.copy()

    for i in range(num_steps):
        if (i + 1) % 50 == 0:
            print(f"  Iteration {i+1}/{num_steps}...")

        current_df = pd.DataFrame([current_design])
        current_vec = current_df[FEATURES].to_numpy().flatten()

        # 預測當前設計的偏差
        pred_deltas = model_len.predict_df(current_df)
        # 【邏輯更新】預測的是最終角度，而不是偏差
        pred_angle = model_ang.predict_df(current_df)
        pred_ds2_frac = pred_deltas["delta_s2"].iloc[0]
        pred_ds3_frac = pred_deltas["delta_s3"].iloc[0]
        pred_ma3 = pred_angle["DIP_a3(deg)"].iloc[0]

        # 計算目標偏差與預測偏差的誤差
        eps = 1e-9
        target_ds2_frac = 1 - (
            target_design["Design_s2(mm)"] / (current_design["Design_s2(mm)"] + eps)
        )
        target_ds3_frac = 1 - (
            target_design["Design_s3(mm)"] / (current_design["Design_s3(mm)"] + eps)
        )

        error_ds2 = target_ds2_frac - pred_ds2_frac
        error_ds3 = target_ds3_frac - pred_ds3_frac
        # 【邏輯更新】誤差是目標角度與預測角度的直接差距
        error_a3 = target_design["Design_a3(deg)"] - pred_ma3
        b_prime = [error_ds2, error_ds3, error_a3]

        # 計算 Jacobian 並求解最佳修正方向 dx
        J_ang_row = model_ang.local_jacobian_numeric(current_vec)
        J_len_rows = model_len.local_jacobian()
        J_current = np.vstack([J_len_rows, J_ang_row[None, :]])
        dx_full = precomp_in_deviation_space(J_current, b_prime, ridge=1e-5)

        # 更新設計
        step_to_take = step_size * dx_full
        temp_design = {
            k: v + step for (k, v), step in zip(current_design.items(), step_to_take)
        }
        current_design = apply_geometric_constraints(temp_design)

    return current_design


def compensate_without_jacobian(
    model_len, model_ang, target_design, num_steps=1000, step_size=0.0001
):
    """不使用 Jacobian 矩陣的直接回饋補償策略。"""
    print("\n=== Running Compensation Strategy: Direct Feedback (No Jacobian) ===")
    current_design = target_design.copy()

    for i in range(num_steps):
        if (i + 1) % 50 == 0:
            print(f"  Iteration {i+1}/{num_steps}...")

        current_df = pd.DataFrame([current_design])

        # 預測當前設計的偏差
        pred_deltas = model_len.predict_df(current_df)
        # 【邏輯更新】預測的是最終角度，而不是偏差
        pred_angle = model_ang.predict_df(current_df)
        pred_ds2_frac = pred_deltas["delta_s2"].iloc[0]
        pred_ds3_frac = pred_deltas["delta_s3"].iloc[0]
        pred_ma3 = pred_angle["DIP_a3(deg)"].iloc[0]

        # 計算預測的最終成品尺寸
        predicted_s2 = current_design["Design_s2(mm)"] * (1 - pred_ds2_frac)
        predicted_s3 = current_design["Design_s3(mm)"] * (1 - pred_ds3_frac)
        # 【邏輯更新】預測的最終角度就是模型輸出
        predicted_a3 = pred_ma3

        # 計算成品尺寸與目標尺寸的直接誤差
        error_s2 = target_design["Design_s2(mm)"] - predicted_s2
        error_s3 = target_design["Design_s3(mm)"] - predicted_s3
        error_a3 = target_design["Design_a3(deg)"] - predicted_a3

        # 直接將誤差按比例加回獨立設計變數
        temp_design = current_design.copy()
        temp_design["Design_s2(mm)"] += step_size * error_s2
        temp_design["Design_s3(mm)"] += step_size * error_s3
        temp_design["Design_a3(deg)"] += step_size * error_a3

        # 套用幾何約束，更新相依變數
        current_design = apply_geometric_constraints(temp_design)

    return current_design


def compensate_with_random_search(
    model_len, model_ang, target_design, df_train, n_trials=10000
):
    """
    【新增】使用隨機搜尋策略在訓練資料範圍內尋找最佳補償解。
    """
    print("\n=== Running Compensation Strategy: Random Search ===")

    # 1. 從訓練資料中確定搜尋邊界
    bounds = {
        "Design_s2(mm)": (
            df_train["Design_s2(mm)"].min(),
            df_train["Design_s2(mm)"].max(),
        ),
        "Design_s3(mm)": (
            df_train["Design_s3(mm)"].min(),
            df_train["Design_s3(mm)"].max(),
        ),
        "Design_a3(deg)": (
            df_train["Design_a3(deg)"].min(),
            df_train["Design_a3(deg)"].max(),
        ),
    }
    print("  Search bounds determined from training data:")
    for key, (min_val, max_val) in bounds.items():
        print(f"    - {key}: {min_val:.4f} to {max_val:.4f}")

    best_design = None
    min_error = float("inf")

    target_s2 = target_design["Design_s2(mm)"]
    target_s3 = target_design["Design_s3(mm)"]
    target_a3 = target_design["Design_a3(deg)"]

    # 2. 執行隨機搜尋
    print(f"  Running {n_trials} random search trials...")
    for i in range(n_trials):
        # 隨機生成一組候選獨立變數
        candidate_independent = {
            "Design_s2(mm)": np.random.uniform(*bounds["Design_s2(mm)"]),
            "Design_s3(mm)": np.random.uniform(*bounds["Design_s3(mm)"]),
            "Design_a3(deg)": np.random.uniform(*bounds["Design_a3(deg)"]),
        }

        # 應用幾何約束得到完整的候選設計
        candidate_design = apply_geometric_constraints(candidate_independent)
        candidate_df = pd.DataFrame([candidate_design])

        # 預測此候選設計的成品尺寸
        pred_len = model_len.predict_df(candidate_df)
        pred_ang = model_ang.predict_df(candidate_df)

        final_s2 = candidate_design["Design_s2(mm)"] * (
            1 - pred_len["delta_s2"].iloc[0]
        )
        final_s3 = candidate_design["Design_s3(mm)"] * (
            1 - pred_len["delta_s3"].iloc[0]
        )
        final_a3 = pred_ang["DIP_a3(deg)"].iloc[0]

        # 計算正規化的均方根誤差 (RMSE)，以便比較不同單位的誤差
        s2_range = bounds["Design_s2(mm)"][1] - bounds["Design_s2(mm)"][0]
        s3_range = bounds["Design_s3(mm)"][1] - bounds["Design_s3(mm)"][0]
        a3_range = bounds["Design_a3(deg)"][1] - bounds["Design_a3(deg)"][0]

        error_s2_norm = (
            ((final_s2 - target_s2) / s2_range) ** 2 if s2_range > 1e-9 else 0
        )
        error_s3_norm = (
            ((final_s3 - target_s3) / s3_range) ** 2 if s3_range > 1e-9 else 0
        )
        error_a3_norm = (
            ((final_a3 - target_a3) / a3_range) ** 2 if a3_range > 1e-9 else 0
        )

        total_error = np.sqrt(error_s2_norm + error_s3_norm + error_a3_norm)

        # 如果找到更好的解，就更新
        if total_error < min_error:
            min_error = total_error
            best_design = candidate_design
            if i % 500 == 0:  # 每隔一段時間印出進度
                print(f"    Trial {i}: New best error = {min_error:.6f}")

    if best_design is None:
        print(
            "  Warning: Random search did not find a valid solution. Returning target design."
        )
        return target_design

    print(f"  Search finished. Best error found: {min_error:.6f}")
    return best_design


def compensate_with_genetic_algorithm(
    model_len,
    model_ang,
    target_design,
    df_train,
    pop_size=500,
    generations=10,
    mutation_rate=0.2,
    crossover_rate=0.7,
    tournament_size=10,
):
    """
    【新增】使用遺傳演算法在訓練資料範圍內尋找最佳補償解。
    """
    print("\n=== Running Compensation Strategy: Genetic Algorithm ===")

    # 1. 初始化
    bounds = {
        "s2": (df_train["Design_s2(mm)"].min(), df_train["Design_s2(mm)"].max()),
        "s3": (df_train["Design_s3(mm)"].min(), df_train["Design_s3(mm)"].max()),
        "a3": (df_train["Design_a3(deg)"].min(), df_train["Design_a3(deg)"].max()),
    }
    print("  Search bounds determined from training data.")

    target_dims = {
        "s2": target_design["Design_s2(mm)"],
        "s3": target_design["Design_s3(mm)"],
        "a3": target_design["Design_a3(deg)"],
    }

    # 適應度函式 (計算誤差)
    def calculate_fitness(individual):
        design = apply_geometric_constraints(
            {
                "Design_s2(mm)": individual[0],
                "Design_s3(mm)": individual[1],
                "Design_a3(deg)": individual[2],
            }
        )
        df = pd.DataFrame([design])
        pred_len = model_len.predict_df(df)
        pred_ang = model_ang.predict_df(df)

        final_s2 = design["Design_s2(mm)"] * (1 - pred_len["delta_s2"].iloc[0])
        final_s3 = design["Design_s3(mm)"] * (1 - pred_len["delta_s3"].iloc[0])
        final_a3 = pred_ang["DIP_a3(deg)"].iloc[0]

        s2_range = bounds["s2"][1] - bounds["s2"][0]
        s3_range = bounds["s3"][1] - bounds["s3"][0]
        a3_range = bounds["a3"][1] - bounds["a3"][0]

        e_s2 = (
            ((final_s2 - target_dims["s2"]) / s2_range) ** 2 if s2_range > 1e-9 else 0
        )
        e_s3 = (
            ((final_s3 - target_dims["s3"]) / s3_range) ** 2 if s3_range > 1e-9 else 0
        )
        e_a3 = (
            ((final_a3 - target_dims["a3"]) / a3_range) ** 2 if a3_range > 1e-9 else 0
        )

        error = np.sqrt(e_s2 + e_s3 + e_a3)
        return 1 / (1 + error)  # 誤差越小，適應度越高

    # 2. 建立初始族群
    population = []
    for _ in range(pop_size):
        individual = [
            random.uniform(*bounds["s2"]),
            random.uniform(*bounds["s3"]),
            random.uniform(*bounds["a3"]),
        ]
        population.append(individual)

    # 3. 演化迴圈
    print(
        f"  Running GA for {generations} generations with population size {pop_size}..."
    )
    best_individual = None
    best_fitness = -1

    for gen in range(generations):
        # 評估族群
        fitness_scores = [calculate_fitness(ind) for ind in population]

        new_population = []

        # 保留最佳個體 (精英策略)
        elite_idx = np.argmax(fitness_scores)
        if fitness_scores[elite_idx] > best_fitness:
            best_fitness = fitness_scores[elite_idx]
            best_individual = population[elite_idx]
        new_population.append(best_individual)

        # 錦標賽選擇
        def tournament_selection():
            tournament = random.sample(
                list(zip(population, fitness_scores)), tournament_size
            )
            return max(tournament, key=lambda x: x[1])[0]

        # 生成新一代
        while len(new_population) < pop_size:
            parent1 = tournament_selection()
            parent2 = tournament_selection()

            # 交配
            if random.random() < crossover_rate:
                child1, child2 = list(parent1), list(parent2)
                crossover_point = random.randint(1, len(child1) - 1)
                child1[crossover_point:], child2[crossover_point:] = (
                    child2[crossover_point:],
                    child1[crossover_point:],
                )
            else:
                child1, child2 = list(parent1), list(parent2)

            # 突變
            for child in [child1, child2]:
                if random.random() < mutation_rate:
                    gene_to_mutate = random.randint(0, len(child) - 1)
                    if gene_to_mutate == 0:
                        child[gene_to_mutate] = random.uniform(*bounds["s2"])
                    elif gene_to_mutate == 1:
                        child[gene_to_mutate] = random.uniform(*bounds["s3"])
                    else:
                        child[gene_to_mutate] = random.uniform(*bounds["a3"])

            if len(new_population) < pop_size:
                new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

        if (gen + 1) % 10 == 0:
            print(f"    Generation {gen+1}: Best Fitness = {best_fitness:.6f}")

    print(f"  GA finished. Best Fitness found: {best_fitness:.6f}")

    # 4. 回傳最佳解
    final_design = apply_geometric_constraints(
        {
            "Design_s2(mm)": best_individual[0],
            "Design_s3(mm)": best_individual[1],
            "Design_a3(deg)": best_individual[2],
        }
    )
    return final_design
