# -*- coding: utf-8 -*-
"""
compensation_strategies.py
- 包含不同的迭代補償演算法。
  - compensate_with_jacobian: 使用 Jacobian 矩陣進行多變數優化。
  - compensate_without_jacobian: 使用簡化的直接誤差回饋法。
"""
import numpy as np
import pandas as pd
from compensation_utils import (
    FEATURES,
    apply_geometric_constraints,
    precomp_in_deviation_space,
)


def compensate_with_jacobian(
    model_len, model_ang, target_design, num_steps=90, step_size=0.01
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
        dx_full = precomp_in_deviation_space(J_current, b_prime)

        # 更新設計
        step_to_take = step_size * dx_full
        temp_design = {
            k: v + step for (k, v), step in zip(current_design.items(), step_to_take)
        }
        current_design = apply_geometric_constraints(temp_design)

    return current_design


def compensate_without_jacobian(
    model_len, model_ang, target_design, num_steps=100, step_size=0.01
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
