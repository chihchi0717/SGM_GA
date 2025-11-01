# -*- coding: utf-8 -*-
"""
compensation_utils.py
- 包含所有共享的輔助函式，例如：
  - 幾何約束計算
  - 資料平均
  - 特徵增強
  - 模型評估指標
  - 線性求解器
"""
import numpy as np
import pandas as pd
import math
from typing import Dict, List

# --- 全域變數 ---
FEATURES = [
    "Design_s1(mm)",
    "Design_s2(mm)",
    "Design_s3(mm)",
    "Design_a3(deg)",
    "Design_a1(deg)",
    "Design_a2(deg)",
]
TARGETS = ["delta_s2", "delta_s3", "DIP_a3(deg)"]

# --- 幾何與資料處理 ---


def apply_geometric_constraints(design_dict: dict) -> dict:
    """
    根據獨立變數 (s2, s3, a3) 計算相依變數 (s1, a1, a2)。
    """
    s2 = design_dict["Design_s2(mm)"]
    s3 = design_dict["Design_s3(mm)"]
    a3_deg = design_dict["Design_a3(deg)"]
    a3_rad = math.radians(a3_deg)

    s1_squared = s3**2 + s2**2 - 2 * s3 * s2 * math.cos(a3_rad)
    s1 = math.sqrt(max(0, s1_squared))

    cos_a1_arg = (s2**2 + s1**2 - s3**2) / (2 * s2 * s1 + 1e-9)
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


def average_by_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    依據唯一的結構設計，對目標值進行平均。
    """
    df_ = df[FEATURES + TARGETS].dropna().copy()
    gb = df_.groupby(FEATURES, dropna=False, as_index=False)
    df_avg = gb[TARGETS].mean()
    counts = gb.size().rename(columns={"size": "n_samples"})
    df_avg = df_avg.merge(counts, on=FEATURES, how="left")
    return df_avg


def augment_feats_for_lengths(
    X,
    add_ratios: bool = True,
    add_sincos: bool = False,
    return_names: bool = False,
    add_interactions: bool = False,
):
    """
    為模型增強特徵。
    """
    s1, s2, s3, a3, a1, a2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    feats = [s1, s2, s3, a3, a1, a2]
    names = ["s1", "s2", "s3", "a3", "a1", "a2"]
    if add_ratios:
        eps = 1e-9
        feats += [
            s1 / np.clip(s2, eps, None),
            s1 / np.clip(s3, eps, None),
            s2 / np.clip(s3, eps, None),
        ]
        names += ["r12", "r13", "r23"]
    if add_sincos:
        rad1, rad2, rad3 = np.deg2rad(a1), np.deg2rad(a2), np.deg2rad(a3)
        feats += [
            np.sin(rad1),
            np.cos(rad1),
            np.sin(rad2),
            np.cos(rad2),
            np.sin(rad3),
            np.cos(rad3),
        ]
        names += ["sin_a1", "cos_a1", "sin_a2", "cos_a2", "sin_a3", "cos_a3"]
    if add_interactions:
        feats += [s1 * s2, s1 * s3, s2 * s3, a1 * a2, a1 * a3, a2 * a3]
        names += ["s1*s2", "s1*s3", "s2*s3", "a1*a2", "a1*a3", "a2*a3"]
    X_aug = np.column_stack(feats)
    return (X_aug, names) if return_names else X_aug


# --- 求解器 ---


def precomp_in_deviation_space(J, b_prime, ridge=1e-6):
    """
    在模型輸出空間 (deviation space) 進行線性求解，用於 Jacobian 策略。
    """
    A = J
    b = np.asarray(b_prime, dtype=float).reshape(-1, 1)
    ATA = A.T @ A + ridge * np.eye(A.shape[1])
    ATb = A.T @ b
    dx = np.linalg.solve(ATA, ATb).flatten()
    return dx


# --- 模型評估 ---


def _metrics(y_true, y_pred):
    e = y_true - y_pred
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    p95 = float(np.percentile(np.abs(e), 95))
    return mae, rmse, r2, p95


def evaluate_overall(df_used, model_len, model_ang) -> pd.DataFrame:
    y_true = df_used[TARGETS].to_numpy(dtype=float)
    y_len = model_len.predict_df(df_used)
    y_ang = model_ang.predict_df(df_used)
    y_pred = pd.concat([y_len, y_ang], axis=1)[TARGETS].to_numpy(dtype=float)
    rows = []
    for j, name in enumerate(TARGETS):
        mae, rmse, r2, p95 = _metrics(y_true[:, j], y_pred[:, j])
        rows.append(
            {"Target": name, "MAE": mae, "RMSE": rmse, "R2": r2, "P95_abs_err": p95}
        )
    return pd.DataFrame(rows)


def evaluate_per_structure(df_raw, model_len, model_ang):
    d = df_raw[FEATURES + TARGETS].dropna().copy()
    d["__key__"] = d[FEATURES].astype(str).agg("|".join, axis=1)
    y_len = model_len.predict_df(d)
    y_ang = model_ang.predict_df(d)
    d_pred = pd.concat([y_len, y_ang], axis=1)
    rows = []
    for key, g in d.groupby("__key__"):
        res = {"structure_key": key, "n_samples": len(g)}
        for t in TARGETS:
            mae, rmse, _, _ = _metrics(
                g[t].to_numpy(), d_pred.loc[g.index, t].to_numpy()
            )
            res[f"{t}_MAE"] = mae
            res[f"{t}_RMSE"] = rmse
        rows.append(res)
    return pd.DataFrame(rows)


def evaluate_kfold_cv(df_used, k: int, model_builder, seed: int = 42):
    if k is None or k <= 1:
        return None
    d = df_used[FEATURES + TARGETS].dropna().copy()
    n = d.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(idx), k)
    Y = d[TARGETS].to_numpy(dtype=float)
    Yhat = np.zeros_like(Y)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        df_tr, df_te = d.iloc[tr], d.iloc[f]
        mdl_len, mdl_ang = model_builder(df_tr)
        pred = pd.concat(
            [mdl_len.predict_df(df_te), mdl_ang.predict_df(df_te)], axis=1
        )[TARGETS].to_numpy(dtype=float)
        Yhat[f] = pred
    rows = []
    for j, name in enumerate(TARGETS):
        mae, rmse, r2, p95 = _metrics(Y[:, j], Yhat[:, j])
        rows.append(
            {"Target": name, "MAE": mae, "RMSE": rmse, "R2": r2, "P95_abs_err": p95}
        )
    return pd.DataFrame(rows)


def evaluate_loso_cv(df_used, model_builder):
    d = df_used[FEATURES + TARGETS].dropna().copy()
    d["__key__"] = d[FEATURES].astype(str).agg("|".join, axis=1)
    Y_true = d[TARGETS].to_numpy(dtype=float)
    Y_pred = np.zeros_like(Y_true)
    per_struct_rows = []
    for key, g in d.groupby("__key__"):
        te_idx = g.index
        tr_idx = d.index.difference(te_idx)
        df_tr, df_te = d.loc[tr_idx, :], d.loc[te_idx, :]
        mdl_len, mdl_ang = model_builder(df_tr)
        pred_df = pd.concat(
            [mdl_len.predict_df(df_te), mdl_ang.predict_df(df_te)], axis=1
        )
        pred = pred_df[TARGETS].to_numpy(dtype=float)
        Y_pred[d.index.get_indexer(te_idx)] = pred
        for t in TARGETS:
            mae, rmse, r2, p95 = _metrics(df_te[t].to_numpy(), pred_df[t].to_numpy())
            per_struct_rows.append(
                {
                    "structure_key": key,
                    "n_samples": len(df_te),
                    "Target": t,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "P95_abs_err": p95,
                }
            )
    overall_rows = []
    for j, name in enumerate(TARGETS):
        mae, rmse, r2, p95 = _metrics(Y_true[:, j], Y_pred[:, j])
        overall_rows.append(
            {"Target": name, "MAE": mae, "RMSE": rmse, "R2": r2, "P95_abs_err": p95}
        )
    return pd.DataFrame(overall_rows), pd.DataFrame(per_struct_rows)
