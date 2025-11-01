# -*- coding: utf-8 -*-
"""
compensation_strategies.py
- 包含不同的迭代補償演算法。
  - compensate_with_jacobian: 使用 Jacobian 矩陣進行多變數優化。
  - compensate_without_jacobian: 使用簡化的直接誤差回饋法。
  - compensate_with_nelder_mead: 【新】使用改良的循序優化法 (Coordinate Descent)。
"""
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize, minimize_scalar

try:
    # 優先使用相對匯入 (適用於套件)
    from .compensation_utils import (
        FEATURES,
        apply_geometric_constraints,
        precomp_in_deviation_space,
    )
except ImportError:
    # 當作為獨立腳本執行時，改用直接匯入
    from compensation_utils import (
        FEATURES,
        apply_geometric_constraints,
        precomp_in_deviation_space,
    )


def compensate_with_nelder_mead(
    model_len,
    model_ang,
    target_design,
    df_train,
    max_cycles=100,
    bounds_expansion_factor=0.2,
):
    """
    【新版策略】使用改良的循序優化法 (Coordinate Descent)。
    此方法為每個目標（s2, s3, a3）使用獨立的目標函式，並在擴展後的
    邊界內進行搜尋，直到所有目標的成品誤差均滿足容忍度為止。
    這解決了先前版本中搜尋範圍狹窄和收斂判斷不佳的問題。
    """
    print(
        "\n=== Running Compensation Strategy: Sequential Optimization (Coordinate Descent) ==="
    )

    # 1. 從訓練資料中確定搜尋邊界，並向外擴展
    print(f"  Expanding search bounds by {bounds_expansion_factor*100}%.")
    bounds = {}
    for key_short, key_long in [
        ("s2", "Design_s2(mm)"),
        ("s3", "Design_s3(mm)"),
        ("a3", "Design_a3(deg)"),
    ]:
        min_val = df_train[key_long].min()
        max_val = df_train[key_long].max()
        range_val = max_val - min_val if max_val > min_val else 1.0

        expanded_min = min_val - range_val * bounds_expansion_factor
        expanded_max = max_val + range_val * bounds_expansion_factor
        bounds[key_short] = (expanded_min, expanded_max)

    print("  New expanded search bounds:")
    for key, (min_val, max_val) in bounds.items():
        print(f"    - Design_{key}(...): {min_val:.4f} to {max_val:.4f}")

    target_dims = {
        "s2": target_design["Design_s2(mm)"],
        "s3": target_design["Design_s3(mm)"],
        "a3": target_design["Design_a3(deg)"],
    }

    # 2. 為每個目標定義誤差收斂容忍度
    error_tolerances = {
        "s2": 0.001,  # s2 成品誤差需小於 0.001 mm
        "s3": 0.001,  # s3 成品誤差需小於 0.001 mm
        "a3": 0.05,  # a3 成品誤差需小於 0.05 度
    }
    print("\n  Convergence Error Tolerances (for final product):")
    for k, v in error_tolerances.items():
        print(f"    - {k}_error: < {v}")

    # 3. 建立一個輔助函式，用於計算給定設計向量下所有目標的最終成品尺寸和誤差
    def evaluate_design(design_vector):
        candidate_independent = {
            "Design_s2(mm)": design_vector[0],
            "Design_s3(mm)": design_vector[1],
            "Design_a3(deg)": design_vector[2],
        }
        design = apply_geometric_constraints(candidate_independent)
        df = pd.DataFrame([design])

        pred_len = model_len.predict_df(df)
        pred_ang = model_ang.predict_df(df)

        final_s2 = design["Design_s2(mm)"] * (1 - pred_len["delta_s2"].iloc[0])
        final_s3 = design["Design_s3(mm)"] * (1 - pred_len["delta_s3"].iloc[0])
        final_a3 = pred_ang["DIP_a3(deg)"].iloc[0]

        errors = {
            "s2": abs(final_s2 - target_dims["s2"]),
            "s3": abs(final_s3 - target_dims["s3"]),
            "a3": abs(final_a3 - target_dims["a3"]),
        }
        return errors

    # 4. 設定初始猜測值，並確保在擴展後的邊界內
    current_solution = np.array(
        [
            target_design["Design_s2(mm)"],
            target_design["Design_s3(mm)"],
            target_design["Design_a3(deg)"],
        ]
    )
    current_solution[0] = np.clip(current_solution[0], bounds["s2"][0], bounds["s2"][1])
    current_solution[1] = np.clip(current_solution[1], bounds["s3"][0], bounds["s3"][1])
    current_solution[2] = np.clip(current_solution[2], bounds["a3"][0], bounds["a3"][1])

    # 5. 執行循序優化的迭代循環
    for cycle in range(max_cycles):
        print(f"\n--- Cycle {cycle + 1}/{max_cycles} ---")

        # 依序優化每個維度 (s2, s3, a3)
        for i, key in enumerate(["s2", "s3", "a3"]):

            # 【獨立目標函式】: 建立一個只針對當前維度的目標函式
            def objective_for_dim(value_to_optimize):
                temp_solution = current_solution.copy()
                temp_solution[i] = value_to_optimize
                # 此函式只回傳當前目標的誤差
                return evaluate_design(temp_solution)[key]

            # 使用高效的 1D 優化器 `minimize_scalar` 在擴展邊界內尋找最佳解
            res = minimize_scalar(
                objective_for_dim,
                bounds=bounds[key],
                method="bounded",
                options={"xatol": 1e-9},
            )

            if res.success:
                current_solution[i] = res.x
                print(
                    f"  Optimized {key}: value = {res.x:.6f}, error for this dim = {res.fun:.6f}"
                )
            else:
                print(f"  Warning: Optimization for {key} failed in this cycle.")

        # 6. 【嚴格收斂標準】: 檢查所有目標的成品誤差是否都已達標
        final_errors = evaluate_design(current_solution)

        print(f"--- End of Cycle {cycle + 1} ---")
        print(
            f"  Current Errors: s2={final_errors['s2']:.6f}, s3={final_errors['s3']:.6f}, a3={final_errors['a3']:.6f}"
        )

        is_converged = (
            final_errors["s2"] < error_tolerances["s2"]
            and final_errors["s3"] < error_tolerances["s3"]
            and final_errors["a3"] < error_tolerances["a3"]
        )

        if is_converged:
            print(
                f"\nSuccess: All target errors are within tolerance. Convergence achieved after {cycle + 1} cycles."
            )
            break

    else:  # for-else 結構，若 for 迴圈正常跑完 (沒被 break) 則執行
        print("\nWarning: Reached max cycles without meeting all error tolerances.")

    # 7. 回傳最終找到的最佳設計方案
    best_design_independent = {
        "Design_s2(mm)": current_solution[0],
        "Design_s3(mm)": current_solution[1],
        "Design_a3(deg)": current_solution[2],
    }

    final_design = apply_geometric_constraints(best_design_independent)
    return final_design


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

        pred_deltas = model_len.predict_df(current_df)
        pred_angle = model_ang.predict_df(current_df)
        pred_ds2_frac = pred_deltas["delta_s2"].iloc[0]
        pred_ds3_frac = pred_deltas["delta_s3"].iloc[0]
        pred_ma3 = pred_angle["DIP_a3(deg)"].iloc[0]

        eps = 1e-9
        target_ds2_frac = 1 - (
            target_design["Design_s2(mm)"] / (current_design["Design_s2(mm)"] + eps)
        )
        target_ds3_frac = 1 - (
            target_design["Design_s3(mm)"] / (current_design["Design_s3(mm)"] + eps)
        )

        error_ds2 = target_ds2_frac - pred_ds2_frac
        error_ds3 = target_ds3_frac - pred_ds3_frac
        error_a3 = target_design["Design_a3(deg)"] - pred_ma3
        b_prime = [error_ds2, error_ds3, error_a3]

        J_ang_row = model_ang.local_jacobian_numeric(current_vec)
        J_len_rows = model_len.local_jacobian()
        J_current = np.vstack([J_len_rows, J_ang_row[None, :]])
        dx_full = precomp_in_deviation_space(J_current, b_prime, ridge=1e-5)

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

        pred_deltas = model_len.predict_df(current_df)
        pred_angle = model_ang.predict_df(current_df)
        pred_ds2_frac = pred_deltas["delta_s2"].iloc[0]
        pred_ds3_frac = pred_deltas["delta_s3"].iloc[0]
        pred_ma3 = pred_angle["DIP_a3(deg)"].iloc[0]

        predicted_s2 = current_design["Design_s2(mm)"] * (1 - pred_ds2_frac)
        predicted_s3 = current_design["Design_s3(mm)"] * (1 - pred_ds3_frac)
        predicted_a3 = pred_ma3

        error_s2 = target_design["Design_s2(mm)"] - predicted_s2
        error_s3 = target_design["Design_s3(mm)"] - predicted_s3
        error_a3 = target_design["Design_a3(deg)"] - predicted_a3

        temp_design = current_design.copy()
        temp_design["Design_s2(mm)"] += step_size * error_s2
        temp_design["Design_s3(mm)"] += step_size * error_s3
        temp_design["Design_a3(deg)"] += step_size * error_a3

        current_design = apply_geometric_constraints(temp_design)

    return current_design


def compensate_with_random_search(
    model_len, model_ang, target_design, df_train, n_trials=10000
):
    """使用隨機搜尋策略在訓練資料範圍內尋找最佳補償解。"""
    print("\n=== Running Compensation Strategy: Random Search ===")

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

    print(f"  Running {n_trials} random search trials...")
    for i in range(n_trials):
        candidate_independent = {
            "Design_s2(mm)": np.random.uniform(*bounds["Design_s2(mm)"]),
            "Design_s3(mm)": np.random.uniform(*bounds["Design_s3(mm)"]),
            "Design_a3(deg)": np.random.uniform(*bounds["Design_a3(deg)"]),
        }

        candidate_design = apply_geometric_constraints(candidate_independent)
        candidate_df = pd.DataFrame([candidate_design])

        pred_len = model_len.predict_df(candidate_df)
        pred_ang = model_ang.predict_df(candidate_df)

        final_s2 = candidate_design["Design_s2(mm)"] * (
            1 - pred_len["delta_s2"].iloc[0]
        )
        final_s3 = candidate_design["Design_s3(mm)"] * (
            1 - pred_len["delta_s3"].iloc[0]
        )
        final_a3 = pred_ang["DIP_a3(deg)"].iloc[0]

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

        if total_error < min_error:
            min_error = total_error
            best_design = candidate_design
            if i % 500 == 0:
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
    """使用遺傳演算法在訓練資料範圍內尋找最佳補償解。"""
    print("\n=== Running Compensation Strategy: Genetic Algorithm ===")

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
        return 1 / (1 + error)

    population = []
    for _ in range(pop_size):
        individual = [
            random.uniform(*bounds["s2"]),
            random.uniform(*bounds["s3"]),
            random.uniform(*bounds["a3"]),
        ]
        population.append(individual)

    print(
        f"  Running GA for {generations} generations with population size {pop_size}..."
    )
    best_individual = None
    best_fitness = -1

    for gen in range(generations):
        fitness_scores = [calculate_fitness(ind) for ind in population]

        new_population = []

        elite_idx = np.argmax(fitness_scores)
        if fitness_scores[elite_idx] > best_fitness:
            best_fitness = fitness_scores[elite_idx]
            best_individual = population[elite_idx]
        new_population.append(best_individual)

        def tournament_selection():
            tournament = random.sample(
                list(zip(population, fitness_scores)), tournament_size
            )
            return max(tournament, key=lambda x: x[1])[0]

        while len(new_population) < pop_size:
            parent1 = tournament_selection()
            parent2 = tournament_selection()

            if random.random() < crossover_rate:
                child1, child2 = list(parent1), list(parent2)
                crossover_point = random.randint(1, len(child1) - 1)
                child1[crossover_point:], child2[crossover_point:] = (
                    child2[crossover_point:],
                    child1[crossover_point:],
                )
            else:
                child1, child2 = list(parent1), list(parent2)

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

    final_design = apply_geometric_constraints(
        {
            "Design_s2(mm)": best_individual[0],
            "Design_s3(mm)": best_individual[1],
            "Design_a3(deg)": best_individual[2],
        }
    )
    return final_design
