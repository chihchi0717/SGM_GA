# -*- coding: utf-8 -*-
"""
run_regression_jacobian.py (Geometrically Constrained Version)
- This version incorporates user-defined geometric constraints to ensure
  that the compensated design is physically valid.
- The iterative compensation now adjusts the 3 independent variables (s2, s3, a3)
  and re-computes the 3 dependent variables (s1, a1, a2) at each step.
"""

import argparse
import numpy as np
import pandas as pd
import math
from typing import Optional, Dict, List

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "Design_s1(mm)",
    "Design_s2(mm)",
    "Design_s3(mm)",
    "Design_a3(deg)",
    "Design_a1(deg)",
    "Design_a2(deg)",
]
TARGETS = ["delta_s2", "delta_s3", "delta_a3"]


# ---------------- NEW: Geometric Constraint Function ----------------
def apply_geometric_constraints(design_dict: dict) -> dict:
    """
    Applies the user-defined geometric rules to calculate dependent variables
    (s1, a1, a2) from the independent variables (s2, s3, a3).
    """
    s2 = design_dict["Design_s2(mm)"]
    s3 = design_dict["Design_s3(mm)"]
    a3_deg = design_dict["Design_a3(deg)"]
    a3_rad = math.radians(a3_deg)

    # Rule 1: Law of Cosines to find s1
    # s1^2 = s2^2 + s3^2 - 2*s2*s3*cos(a1_deg) <- This assumes a1 is between s2 and s3
    # The user provided s1 = sqrt(s3^2 + s2^2 - 2*s3*s2*cos(a3))
    # This implies a3 is the angle between s2 and s3. We will follow this rule.
    s1_squared = s3**2 + s2**2 - 2 * s3 * s2 * math.cos(a3_rad)
    s1 = math.sqrt(max(0, s1_squared))  # Ensure non-negative before sqrt

    # Rule 2: Law of Cosines to find a1
    # Clamp the argument for acos to [-1, 1] to avoid math domain errors
    cos_a1_arg = (s2**2 + s1**2 - s3**2) / (2 * s2 * s1 + 1e-9)
    cos_a1_arg = max(-1.0, min(1.0, cos_a1_arg))
    a1_rad = math.acos(cos_a1_arg)
    a1_deg = math.degrees(a1_rad)

    # Rule 3: Sum of angles is 180 degrees
    a2_deg = 180 - a1_deg - a3_deg

    # Create a new dictionary with the updated, constrained values
    constrained_design = {
        "Design_s1(mm)": s1,
        "Design_s2(mm)": s2,
        "Design_s3(mm)": s3,
        "Design_a3(deg)": a3_deg,
        "Design_a1(deg)": a1_deg,
        "Design_a2(deg)": a2_deg,
    }
    return constrained_design


# ---------------- Averaging ----------------
def average_by_structure(
    df: pd.DataFrame, feature_cols=FEATURES, target_cols=TARGETS
) -> pd.DataFrame:
    df_ = df[feature_cols + target_cols].dropna().copy()
    gb = df_.groupby(feature_cols, dropna=False, as_index=False)
    df_avg = gb[target_cols].mean()
    counts = gb.size().rename(columns={"size": "n_samples"})
    df_avg = df_avg.merge(counts, on=feature_cols, how="left")
    return df_avg


# ---------------- Feature augmentation ----------------
def augment_feats_for_lengths(
    X,
    add_ratios: bool = True,
    add_sincos: bool = False,
    return_names: bool = False,
    add_interactions: bool = False,
):
    import numpy as np

    s1, s2, s3, a3, a1, a2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    feats = [s1, s2, s3, a3, a1, a2]
    names = ["s1", "s2", "s3", "a3", "a1", "a2"]
    if add_ratios:
        eps = 1e-9
        r12 = s1 / np.clip(s2, eps, None)
        r13 = s1 / np.clip(s3, eps, None)
        r23 = s2 / np.clip(s3, eps, None)
        feats += [r12, r13, r23]
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
        inter_terms = [
            s1 * s2,
            s1 * s3,
            s2 * s3,
            a1 * a2,
            a1 * a3,
            a2 * a3,
        ]
        inter_names = ["s1*s2", "s1*s3", "s2*s3", "a1*a2", "a1*a3", "a2*a3"]
        feats += inter_terms
        names += inter_names
    X_aug = np.column_stack(feats)
    return (X_aug, names) if return_names else X_aug


# ---------------- Linear OLS (for lengths) ----------------
class LinearOLS:
    def __init__(self, ridge: float = 1e-9):
        self.ridge = ridge
        self.beta: Dict[str, np.ndarray] = {}  # y -> (1+6,)

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X = d[FEATURES].to_numpy(dtype=float)
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        K = X_.T @ X_ + self.ridge * np.eye(X_.shape[1])
        Kinv = np.linalg.inv(K)
        for yname in ["delta_s2", "delta_s3"]:
            y = d[yname].to_numpy(dtype=float).reshape(-1, 1)
            b = Kinv @ (X_.T @ y)
            self.beta[yname] = b.ravel()

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        out = {}
        for yname, b in self.beta.items():
            out[yname] = (X_ @ b).ravel()
        return pd.DataFrame(out, index=df_design.index)

    def local_jacobian(self) -> np.ndarray:
        # 兩列（delta_s2, delta_s3）的解析斜率（去掉截距）
        J = np.zeros((2, 6), dtype=float)
        for i, y in enumerate(["delta_s2", "delta_s3"]):
            J[i, :] = self.beta[y][1:]
        return J


class LengthModelRF:
    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        add_ratios: bool = False,
        add_sincos: bool = False,
        add_interactions: bool = False,
        max_features: float = 1.0,
        criterion: str = "squared_error",
    ):
        self.add_ratios = bool(add_ratios)
        self.add_sincos = bool(add_sincos)
        self.add_interactions = bool(add_interactions)

        self.model_s2 = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            max_features=max_features,
            criterion=criterion,
        )
        self.model_s3 = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            max_features=max_features,
            criterion=criterion,
        )

    def _X(self, df):
        X_raw = df[FEATURES].to_numpy(dtype=float)
        return augment_feats_for_lengths(
            X_raw,
            add_ratios=self.add_ratios,
            add_sincos=self.add_sincos,
            add_interactions=self.add_interactions,
        )

    def fit(self, df):
        X = self._X(df)
        y2 = df["delta_s2"].to_numpy(dtype=float)
        y3 = df["delta_s3"].to_numpy(dtype=float)
        self.model_s2.fit(X, y2)
        self.model_s3.fit(X, y3)
        return self

    def predict_df(self, df_design):
        X = self._X(df_design)
        return pd.DataFrame(
            {
                "delta_s2": self.model_s2.predict(X),
                "delta_s3": self.model_s3.predict(X),
            },
            index=df_design.index,
        )

    def local_jacobian(self):
        # RF 無解析 Jacobian
        return np.zeros((2, 6))


class LengthModelHuber:
    """
    以 Huber 擬合 delta_s2、delta_s3
    - 可選：特徵標準化(scale)、比例(add_ratios)、角度sincos(add_sincos)
    - local_jacobian(): 對原始6設計量 [s1,s2,s3,a3,a1,a2] 的 2x6 Jacobian
      * 若 add_ratios/sincos=True，使用鏈式法則（在訓練集均值處評估）
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        epsilon: float = 1.35,
        max_iter: int = 2000,
        scale: bool = True,
        add_ratios: bool = False,
        add_sincos: bool = False,
        add_interactions: bool = False,
    ):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.scale = bool(scale)
        self.add_ratios = bool(add_ratios)
        self.add_sincos = bool(add_sincos)
        self.add_interactions = bool(add_interactions)
        self.model_s2 = HuberRegressor(
            alpha=self.alpha, epsilon=self.epsilon, max_iter=self.max_iter
        )
        self.model_s3 = HuberRegressor(
            alpha=self.alpha, epsilon=self.epsilon, max_iter=self.max_iter
        )

        self.scaler: StandardScaler | None = None
        self._aug_names: list[str] | None = None
        self._ref_point_: np.ndarray | None = None  # 訓練集均值(原始6維)

    def _augment(self, X_raw: np.ndarray, fit_names: bool = False) -> np.ndarray:
        if fit_names or (self._aug_names is None):
            X_aug, names = augment_feats_for_lengths(
                X_raw,
                add_ratios=self.add_ratios,
                add_sincos=self.add_sincos,
                add_interactions=self.add_interactions,
                return_names=True,
            )
            self._aug_names = list(names)
        else:
            X_aug = augment_feats_for_lengths(
                X_raw,
                add_ratios=self.add_ratios,
                add_sincos=self.add_sincos,
                add_interactions=self.add_interactions,
                return_names=False,
            )
        return X_aug

    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_raw = df[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw, fit_names=True if fit else False)
        if self.scale:
            if fit or (self.scaler is None):
                self.scaler = StandardScaler().fit(X_aug)
            X_aug = self.scaler.transform(X_aug)
        return X_aug

    def fit(self, df: pd.DataFrame):
        X_aug = self._prepare_X(df, fit=True)
        self._ref_point_ = df[FEATURES].to_numpy(dtype=float).mean(axis=0)

        y2 = df["delta_s2"].to_numpy(dtype=float)
        y3 = df["delta_s3"].to_numpy(dtype=float)

        self.model_s2.fit(X_aug, y2)
        self.model_s3.fit(X_aug, y3)
        return self

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X_aug = self._prepare_X(df_design, fit=False)
        pred_s2 = self.model_s2.predict(X_aug)
        pred_s3 = self.model_s3.predict(X_aug)
        return pd.DataFrame(
            {"delta_s2": pred_s2, "delta_s3": pred_s3}, index=df_design.index
        )

    def _aug_jac_wrt_base(self, x_raw: np.ndarray) -> np.ndarray:
        assert self._aug_names is not None, "Aug names not set; call fit() first."
        s1, s2, s3, a3, a1, a2 = x_raw.tolist()
        eps = 1e-9
        rows = []
        deg2rad = np.pi / 180.0

        for name in self._aug_names:
            v = np.zeros(6, dtype=float)
            if name == "s1":
                v[0] = 1.0
            elif name == "s2":
                v[1] = 1.0
            elif name == "s3":
                v[2] = 1.0
            elif name == "a3":
                v[3] = 1.0
            elif name == "a1":
                v[4] = 1.0
            elif name == "a2":
                v[5] = 1.0
            elif name == "r12":
                denom = s2 if abs(s2) > eps else (np.sign(s2) * eps if s2 != 0 else eps)
                v[0] = 1.0 / denom
                v[1] = -s1 / (denom**2)
            elif name == "r13":
                denom = s3 if abs(s3) > eps else (np.sign(s3) * eps if s3 != 0 else eps)
                v[0] = 1.0 / denom
                v[2] = -s1 / (denom**2)
            elif name == "r23":
                denom = s3 if abs(s3) > eps else (np.sign(s3) * eps if s3 != 0 else eps)
                v[1] = 1.0 / denom
                v[2] = -s2 / (denom**2)
            elif name == "sin_a1":
                v[4] = np.cos(a1 * deg2rad) * deg2rad
            elif name == "cos_a1":
                v[4] = -np.sin(a1 * deg2rad) * deg2rad
            elif name == "sin_a2":
                v[5] = np.cos(a2 * deg2rad) * deg2rad
            elif name == "cos_a2":
                v[5] = -np.sin(a2 * deg2rad) * deg2rad
            elif name == "sin_a3":
                v[3] = np.cos(a3 * deg2rad) * deg2rad
            elif name == "cos_a3":
                v[3] = -np.sin(a3 * deg2rad) * deg2rad
            rows.append(v)
        return np.vstack(rows)

    def _jac_at_point(self, x_raw: np.ndarray) -> np.ndarray:
        def _coef_on_raw(model):
            if self.scale and (self.scaler is not None):
                std = self.scaler.scale_.astype(float)
                std = np.where(std == 0, 1.0, std)
                return model.coef_.astype(float) / std
            return model.coef_.astype(float)

        g2_aug = _coef_on_raw(self.model_s2).ravel()
        g3_aug = _coef_on_raw(self.model_s3).ravel()

        D = self._aug_jac_wrt_base(x_raw)
        dy2_dx = D.T @ g2_aug
        dy3_dx = D.T @ g3_aug
        J = np.vstack([dy2_dx, dy3_dx])
        return J

    def local_jacobian(self) -> np.ndarray:
        if self.add_ratios or self.add_sincos:
            if self._ref_point_ is None:
                raise RuntimeError("Model must be fit() before computing Jacobian.")
            return self._jac_at_point(self._ref_point_)
        if self.scale and (self.scaler is not None):
            std = self.scaler.scale_.astype(float)
            std = np.where(std == 0, 1.0, std)
            g2 = (self.model_s2.coef_.astype(float) / std).ravel()
            g3 = (self.model_s3.coef_.astype(float) / std).ravel()
        else:
            g2 = self.model_s2.coef_.astype(float).ravel()
            g3 = self.model_s3.coef_.astype(float).ravel()
        g2 = g2[: len(FEATURES)]
        g3 = g3[: len(FEATURES)]
        return np.vstack([g2, g3])

    def local_jacobian_at(self, x_raw: np.ndarray) -> np.ndarray:
        x_raw = np.asarray(x_raw, dtype=float).ravel()
        if x_raw.shape[0] != len(FEATURES):
            raise ValueError(f"x_raw must have length {len(FEATURES)}")
        return self._jac_at_point(x_raw)


# ---------------- Angle model base (feature augmentation) ----------------
class AngleModelBase:
    def __init__(self, add_sincos: bool = True, add_ratios: bool = False):
        self.add_sincos = bool(add_sincos)
        self.add_ratios = bool(add_ratios)
        self.feat_names_: List[str] = []

    def _augment(self, X: np.ndarray) -> np.ndarray:
        s1, s2, s3 = X[:, 0], X[:, 1], X[:, 2]
        a3, a1, a2 = X[:, 3], X[:, 4], X[:, 5]
        feats = [s1, s2, s3, a3, a1, a2]
        names = ["s1", "s2", "s3", "a3", "a1", "a2"]

        if self.add_ratios:
            eps = 1e-9
            r12 = s1 / np.clip(s2, eps, None)
            r13 = s1 / np.clip(s3, eps, None)
            r23 = s2 / np.clip(s3, eps, None)
            feats += [r12, r13, r23]
            names += ["s1/s2", "s1/s3", "s2/s3"]

        if self.add_sincos:
            r1, r2, r3 = np.deg2rad(a1), np.deg2rad(a2), np.deg2rad(a3)
            feats += [
                np.sin(r1),
                np.cos(r1),
                np.sin(r2),
                np.cos(r2),
                np.sin(r3),
                np.cos(r3),
            ]
            names += ["sin(a1)", "cos(a1)", "sin(a2)", "cos(a2)", "sin(a3)", "cos(a3)"]

        X_aug = np.column_stack(feats)
        self.feat_names_ = names
        return X_aug

    def local_jacobian_numeric(self, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        def f(xx: np.ndarray) -> float:
            df_tmp = pd.DataFrame([dict(zip(FEATURES, xx))])
            return float(self.predict_df(df_tmp)["delta_a3"].iloc[0])

        J = np.zeros(6, dtype=float)
        for j in range(6):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            J[j] = (f(xp) - f(xm)) / (2 * h)
        return J


# ---------------- Angle model variants ----------------
class AngleModelOLS(AngleModelBase):
    def __init__(
        self,
        degree: int = 2,
        ridge: float = 1e-2,
        add_sincos: bool = True,
        add_ratios: bool = False,
    ):
        super().__init__(add_sincos, add_ratios)
        self.degree = int(degree)
        self.ridge = float(ridge)
        self.poly: Optional[PolynomialFeatures] = None
        self.beta: Optional[np.ndarray] = None
        self.poly_feat_names: List[str] = []

    def _design(self, X: np.ndarray) -> np.ndarray:
        X_aug = self._augment(X)
        if self.degree > 1:
            if self.poly is None:
                self.poly = PolynomialFeatures(self.degree, include_bias=True)
                Phi = self.poly.fit_transform(X_aug)
                self.poly_feat_names = list(
                    self.poly.get_feature_names_out(self.feat_names_)
                )
            else:
                Phi = self.poly.transform(X_aug)
        else:
            Phi = np.hstack([np.ones((X_aug.shape[0], 1)), X_aug])
            self.poly_feat_names = ["Intercept"] + self.feat_names_
        return Phi

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X = d[FEATURES].to_numpy(dtype=float)
        Phi = self._design(X)
        y = d["delta_a3"].to_numpy(dtype=float).reshape(-1, 1)
        K = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        self.beta = (np.linalg.inv(K) @ (Phi.T @ y)).ravel()

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        Phi = self._design(X)
        yhat = (Phi @ self.beta).ravel()
        return pd.DataFrame({"delta_a3": yhat}, index=df_design.index)


class AngleModelHuber(AngleModelBase):
    def __init__(
        self,
        alpha: float = 1e-3,
        epsilon: float = 1.35,
        max_iter: int = 2000,
        scale: bool = True,
        add_sincos: bool = True,
        add_ratios: bool = False,
    ):
        super().__init__(add_sincos, add_ratios)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.scale = bool(scale)
        self.scaler: Optional[StandardScaler] = None
        self.model = HuberRegressor(
            alpha=self.alpha, epsilon=self.epsilon, max_iter=self.max_iter
        )

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X = d[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X)
        if self.scale:
            self.scaler = StandardScaler().fit(X_aug)
            X_aug = self.scaler.transform(X_aug)
        y = d["delta_a3"].to_numpy(dtype=float)
        self.model.fit(X_aug, y)

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X)
        if self.scale and self.scaler is not None:
            X_aug = self.scaler.transform(X_aug)
        yhat = self.model.predict(X_aug)
        return pd.DataFrame({"delta_a3": yhat}, index=df_design.index)


class AngleModelRF(AngleModelBase):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        add_sincos: bool = True,
        add_ratios: bool = False,
    ):
        super().__init__(add_sincos, add_ratios)
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth if max_depth is None else int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X = d[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X)
        y = d["delta_a3"].to_numpy(dtype=float)
        self.model.fit(X_aug, y)

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X)
        yhat = self.model.predict(X_aug)
        return pd.DataFrame({"delta_a3": yhat}, index=df_design.index)


# ---------------- Metrics & evaluation ----------------
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
        df_tr = d.iloc[tr]
        df_te = d.iloc[f]
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
    need = FEATURES + TARGETS
    d = df_used[need].dropna().copy()
    d["__key__"] = d[FEATURES].astype(str).agg("|".join, axis=1)

    Y_true = d[TARGETS].to_numpy(dtype=float)
    Y_pred = np.zeros_like(Y_true)

    per_struct_rows = []
    for key, g in d.groupby("__key__"):
        te_idx = g.index
        tr_idx = d.index.difference(te_idx)
        df_tr = d.loc[tr_idx, :]
        df_te = d.loc[te_idx, :]

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

    overall_df = pd.DataFrame(overall_rows)
    per_struct_df = pd.DataFrame(per_struct_rows)
    return overall_df, per_struct_df


# ---------------- Pre-compensation Solvers ----------------
def precomp_shrink_into_original(
    J, dm2, dm3, dma3, weights=None, allow_mask=None, ridge=1e-6
):
    """求解器: 在成品尺寸空間 (measurement space) 進行線性求解"""
    E_s2 = np.array([0, 1, 0, 0, 0, 0], float)
    E_s3 = np.array([0, 0, 1, 0, 0, 0], float)
    A = np.vstack([E_s2 - J[0, :], E_s3 - J[1, :], J[2, :]])
    b = np.array([dm2, dm3, dma3], float).reshape(-1, 1)
    if weights is not None:
        W = np.diag(
            np.asarray(weights, dtype=float).reshape(
                3,
            )
        )
        A = W @ A
        b = W @ b
    masked = False
    if allow_mask is not None:
        allow_mask = np.asarray(allow_mask, dtype=bool).reshape(
            6,
        )
        A = A[:, allow_mask]
        masked = True
    ATA = A.T @ A + ridge * np.eye(A.shape[1])
    ATb = A.T @ b
    dx_red = np.linalg.solve(ATA, ATb).flatten()
    if masked:
        dx = np.zeros(6, dtype=float)
        dx[allow_mask] = dx_red
        return dx
    return dx_red


def precomp_in_deviation_space(J, b_prime, ridge=1e-6):
    """求解器: 在模型輸出空間 (deviation space) 進行線性求解"""
    A = J
    b = np.asarray(b_prime, dtype=float).reshape(-1, 1)
    ATA = A.T @ A + ridge * np.eye(A.shape[1])
    ATb = A.T @ b
    dx = np.linalg.solve(ATA, ATb).flatten()
    return dx


# ---------------- CLI ----------------
def parse_vec6(s: str) -> np.ndarray:
    vals = [float(v.strip()) for v in s.split(",")]
    if len(vals) != 6:
        raise ValueError("--jac-at 需要 6 個數字，順序為 s1,s2,s3,a3,a1,a2")
    return np.array(vals, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="analysis_results0814.xlsx")
    ap.add_argument("--sheet", type=str, default="Sheet1")
    ap.add_argument("--average", action="store_true")
    ap.add_argument("--save-avg", type=str, default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument(
        "--cv",
        type=str,
        default="0",
        help='交叉驗證：整數 K 代表 K-fold；或 "loso" 啟用 Leave-One-Structure-Out',
    )
    ap.add_argument("--save-report", type=str, default=None)
    ap.add_argument("--jac-at", type=str, default=None)
    ap.add_argument(
        "--add-interactions",
        action="store_true",
        help="為長度與角度模型加入交互作用特徵",
    )

    # ---- 長度模型選項 ----
    ap.add_argument(
        "--length-model",
        type=str,
        default="ols",
        choices=["ols", "huber", "rf"],
        help="長度回歸器：ols/huber/rf",
    )
    ap.add_argument("--len-huber-alpha", type=float, default=1e-3)
    ap.add_argument("--len-huber-eps", type=float, default=1.35)
    ap.add_argument("--len-huber-max-iter", type=int, default=2000)
    ap.add_argument(
        "--scale-length", action="store_true", help="標準化長度模型輸入特徵"
    )
    ap.add_argument("--len-rf-n-est", type=int, default=300)
    ap.add_argument("--len-rf-max-depth", type=int, default=None)
    ap.add_argument("--len-rf-min-leaf", type=int, default=1)
    ap.add_argument("--len-rf-max-features", type=float, default=1.0)
    ap.add_argument(
        "--len-rf-criterion",
        type=str,
        default="squared_error",
        choices=["squared_error", "absolute_error"],
    )
    ap.add_argument("--len-add-ratios", action="store_true")
    ap.add_argument("--len-add-sincos", action="store_true")

    # 角度模型選擇與參數
    ap.add_argument(
        "--angle-model",
        type=str,
        default="rf",
        choices=["ols", "rf", "huber"],
        help="角度回歸器：ols(多項式+ridge)/rf(RandomForest)/huber",
    )
    ap.add_argument("--angle-poly", type=int, default=2)
    ap.add_argument("--angle-ridge", type=float, default=1e-2)
    ap.add_argument("--rf-n-est", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-leaf", type=int, default=1)
    ap.add_argument("--huber-max-iter", type=int, default=2000)
    ap.add_argument("--huber-eps", type=float, default=1.35)
    ap.add_argument(
        "--scale-angle", action="store_true", help="標準化角度模型的輸入特徵"
    )
    ap.add_argument("--add-angle-sincos", action="store_true")
    ap.add_argument("--add-ratios", action="store_true")

    args = ap.parse_args()

    # 1) 讀資料 +（可選）平均
    df_raw = pd.read_excel(args.file, sheet_name=args.sheet)
    if args.average:
        df_use = average_by_structure(df_raw, FEATURES, TARGETS)
        print(
            f"\n[Average] 合併後共有 {len(df_use)} 種結構；樣本數總和={int(df_use['n_samples'].sum())}（平均前）"
        )
        if args.save_avg:
            df_use.to_excel(args.save_avg, index=False)
            print(f"[Average] 已輸出平均後表格：{args.save_avg}")
    else:
        df_use = df_raw.copy()

    # 2) 訓練模型
    if args.length_model == "ols":
        model_len = LinearOLS(ridge=1e-9)
    elif args.length_model == "huber":
        model_len = LengthModelHuber(
            alpha=args.len_huber_alpha,
            epsilon=args.len_huber_eps,
            max_iter=args.len_huber_max_iter,
            scale=args.scale_length,
            add_ratios=getattr(args, "len_add_ratios", False),
            add_sincos=getattr(args, "len_add_sincos", False),
            add_interactions=getattr(args, "add_interactions", False),
        )
    elif args.length_model == "rf":
        model_len = LengthModelRF(
            n_estimators=args.len_rf_n_est,
            max_depth=args.len_rf_max_depth,
            min_samples_leaf=args.len_rf_min_leaf,
            add_ratios=getattr(args, "len_add_ratios", False),
            add_sincos=getattr(args, "len_add_sincos", False),
            max_features=getattr(args, "len_rf_max_features", 1.0),
            criterion=getattr(args, "len_rf_criterion", "squared_error"),
            add_interactions=getattr(args, "add_interactions", False),
        )
    else:
        raise ValueError(f"Unknown --length-model: {args.length_model}")
    model_len.fit(df_use)

    if args.angle_model == "ols":
        model_ang = AngleModelOLS(
            degree=args.angle_poly,
            ridge=args.angle_ridge,
            add_sincos=args.add_angle_sincos,
            add_ratios=args.add_ratios,
        )
    elif args.angle_model == "huber":
        model_ang = AngleModelHuber(
            alpha=args.angle_ridge,
            epsilon=args.huber_eps,
            max_iter=args.huber_max_iter,
            scale=args.scale_angle,
            add_sincos=args.add_angle_sincos,
            add_ratios=args.add_ratios,
        )
    else:  # "rf"
        model_ang = AngleModelRF(
            n_estimators=args.rf_n_est,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_leaf,
            add_sincos=args.add_angle_sincos,
            add_ratios=args.add_ratios,
        )
    model_ang.fit(df_use)

    # 3) ===== 迭代式事前預測補償 (Iterative Proactive Pre-compensation) =====
    print("\n\n=== Iterative Proactive Pre-compensation (Deviation Space Strategy) ===")
    print(
        "[LOGIC UPDATE] Angle compensation is now based on delta_a3 = design_a3 - dip_a3."
    )

    # --- 步驟 1: 定義您的「目標設計 (target_design)」---
    # --- 手動設定區塊 ---
    target_design_independent = {
        "Design_s2(mm)": 0.87,
        "Design_s3(mm)": 0.88,
        "Design_a3(deg)": 31.0,
    }
    target_design = apply_geometric_constraints(target_design_independent)

    print("\n--- Initial Target Design (Geometrically Constrained) ---")
    for k, v in target_design.items():
        print(f"  - {k}: {v:.6f}")

    # --- 迭代控制器 ---
    num_steps = 1000
    step_size = 0.001
    current_design = target_design.copy()
    print(
        f"\n[Info] Starting iterative compensation with {num_steps} steps (step_size={step_size})..."
    )

    # --- 迭代迴圈 ---
    for i in range(num_steps):
        if (i + 1) % 500 == 0:  # 每 50 次迭代印一次 log
            print(f"\n--- Iteration {i+1}/{num_steps} ---")
            print(
                "  Current Design:", {k: f"{v:.4f}" for k, v in current_design.items()}
            )

        current_df = pd.DataFrame([current_design])
        current_vec = current_df[FEATURES].to_numpy().flatten()

        # 步驟 2: 從「當前設計」預測其收縮率與角度偏差
        predicted_deltas = model_len.predict_df(current_df)
        predicted_angle_delta = model_ang.predict_df(current_df)

        pred_ds2_frac = predicted_deltas["delta_s2"].iloc[0]
        pred_ds3_frac = predicted_deltas["delta_s3"].iloc[0]
        pred_delta_a3 = predicted_angle_delta["delta_a3"].iloc[
            0
        ]  # 模型預測的偏差 (design - dip)

        # 步驟 3: 計算「目標偏差」與「預測偏差」之間的差距
        eps = 1e-9
        target_ds2_frac = 1 - (
            target_design["Design_s2(mm)"] / (current_design["Design_s2(mm)"] + eps)
        )
        target_ds3_frac = 1 - (
            target_design["Design_s3(mm)"] / (current_design["Design_s3(mm)"] + eps)
        )
        error_ds2 = target_ds2_frac - pred_ds2_frac
        error_ds3 = target_ds3_frac - pred_ds3_frac

        # 【邏輯更新】計算角度的目標偏差與誤差
        # 目標：最終的 dip_a3 應該等於 target_design["Design_a3(deg)"]
        # 對於目前的 current_design，我們期望的偏差值應該是：
        target_delta_a3 = (
            current_design["Design_a3(deg)"] - target_design["Design_a3(deg)"]
        )
        # 誤差 = 期望的偏差 - 模型預測的偏差
        error_a3 = target_delta_a3 - pred_delta_a3

        b_prime = [error_ds2, error_ds3, error_a3]

        # 步驟 4: 在「當前點」計算雅可比矩陣 J，並求解 Δx
        J_ang_row_current = model_ang.local_jacobian_numeric(current_vec)
        J_current = np.vstack([model_len.local_jacobian(), J_ang_row_current[None, :]])
        dx_full = precomp_in_deviation_space(J_current, b_prime)

        # 步驟 5: 只往前走一小步
        step_to_take = step_size * dx_full

        # 步驟 6: 更新設計，並強制執行幾何約束
        temp_design = {
            k: v + step for (k, v), step in zip(current_design.items(), step_to_take)
        }
        current_design = apply_geometric_constraints(temp_design)

    # --- 最終結果 ---
    compensated_design = current_design
    print("\n\n--- Final Compensated Design (after iterations) ---")
    for k, v in compensated_design.items():
        print(f"  - {k}: {v:.6f}")

    # --- 步驟 7: (驗證) 將補償後的設計代入模型，檢視最終預測結果 ---
    print("\n\n--- Step 7: Verification of the Final Design ---")
    compensated_df = pd.DataFrame([compensated_design])
    final_predicted_deltas = model_len.predict_df(compensated_df)
    final_predicted_angle_delta = model_ang.predict_df(compensated_df)

    final_pred_ds2_frac = final_predicted_deltas["delta_s2"].iloc[0]
    final_pred_ds3_frac = final_predicted_deltas["delta_s3"].iloc[0]
    final_pred_delta_a3 = final_predicted_angle_delta["delta_a3"].iloc[0]  # 預測的偏差

    print("\n  - Predicted Shrinkage & Deviation for Compensated Design:")
    print(
        f"    - s2 shrinkage: {final_pred_ds2_frac:.4f} ({final_pred_ds2_frac*100:.2f}%)"
    )
    print(
        f"    - s3 shrinkage: {final_pred_ds3_frac:.4f} ({final_pred_ds3_frac*100:.2f}%)"
    )
    print(f"    - a3 deviation (design-dip): {final_pred_delta_a3:.4f} degrees")

    # 【邏輯更新】計算最終成品尺寸
    final_s2 = compensated_design["Design_s2(mm)"] * (1 - final_pred_ds2_frac)
    final_s3 = compensated_design["Design_s3(mm)"] * (1 - final_pred_ds3_frac)
    # 最終成品角度 (dip_a3) = 設計角度 (design_a3) - 預測偏差 (predicted_delta_a3)
    final_a3 = compensated_design["Design_a3(deg)"] - final_pred_delta_a3

    print(
        "\n  - Final Predicted Dimensions (Compensated Design + Predicted Shrinkage/Deviation):"
    )
    final_s1 = compensated_design["Design_s1(mm)"]
    final_a1 = compensated_design["Design_a1(deg)"]
    final_a2 = compensated_design["Design_a2(deg)"]
    print(f"    - s1: {final_s1:.6f} mm")
    print(f"    - s2: {final_s2:.6f} mm")
    print(f"    - s3: {final_s3:.6f} mm")
    print(f"    - a3: {final_a3:.6f} degrees")
    print(f"    - a1: {final_a1:.6f} degrees")
    print(f"    - a2: {final_a2:.6f} degrees")

    print("\n  - Comparison with Original Target:")
    print(
        f"    - Target s1: {target_design['Design_s1(mm)']:.6f}  |  Achieved s1: {final_s1:.6f}  (Constrained)"
    )
    print(
        f"    - Target s2: {target_design['Design_s2(mm)']:.6f}  |  Achieved s2: {final_s2:.6f}  (Predicted by model)"
    )
    print(
        f"    - Target s3: {target_design['Design_s3(mm)']:.6f}  |  Achieved s3: {final_s3:.6f}  (Predicted by model)"
    )
    print(
        f"    - Target a3: {target_design['Design_a3(deg)']:.6f}  |  Achieved a3: {final_a3:.6f}  (Predicted by model)"
    )
    print(
        f"    - Target a1: {target_design['Design_a1(deg)']:.6f}  |  Achieved a1: {final_a1:.6f}  (Constrained)"
    )
    print(
        f"    - Target a2: {target_design['Design_a2(deg)']:.6f}  |  Achieved a2: {final_a2:.6f}  (Constrained)"
    )

    # 構造便捷 builder，供 K-fold/LOSO 重新訓練
    def builder(df_tr):
        if args.length_model == "ols":
            mdl_len = LinearOLS(ridge=1e-9)
        elif args.length_model == "huber":
            mdl_len = LengthModelHuber(
                alpha=args.len_huber_alpha,
                epsilon=args.len_huber_eps,
                max_iter=args.len_huber_max_iter,
                scale=args.scale_length,
                add_ratios=getattr(args, "len_add_ratios", False),
                add_sincos=getattr(args, "len_add_sincos", False),
                add_interactions=getattr(args, "add_interactions", False),
            )
        elif args.length_model == "rf":
            mdl_len = LengthModelRF(
                n_estimators=args.len_rf_n_est,
                max_depth=args.len_rf_max_depth,
                min_samples_leaf=args.len_rf_min_leaf,
                add_ratios=getattr(args, "len_add_ratios", False),
                add_sincos=getattr(args, "len_add_sincos", False),
                max_features=getattr(args, "len_rf_max_features", 1.0),
                criterion=getattr(args, "len_rf_criterion", "squared_error"),
                add_interactions=getattr(args, "add_interactions", False),
            )
        else:
            raise ValueError(f"Unknown --length-model: {args.length_model}")
        mdl_len.fit(df_tr)

        if args.angle_model == "ols":
            mdl_ang = AngleModelOLS(
                degree=args.angle_poly,
                ridge=args.angle_ridge,
                add_sincos=args.add_angle_sincos,
                add_ratios=args.add_ratios,
            )
        elif args.angle_model == "huber":
            mdl_ang = AngleModelHuber(
                alpha=args.angle_ridge,
                epsilon=args.huber_eps,
                max_iter=args.huber_max_iter,
                scale=args.scale_angle,
                add_sincos=args.add_angle_sincos,
                add_ratios=args.add_ratios,
            )
        else:  # "rf"
            mdl_ang = AngleModelRF(
                n_estimators=args.rf_n_est,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_leaf,
                add_sincos=args.add_angle_sincos,
                add_ratios=args.add_ratios,
            )
        mdl_ang.fit(df_tr)

        return mdl_len, mdl_ang

    # 7) 評估 & 報告
    if args.eval or (args.cv and args.cv != "0") or args.save_report:
        print("\n\n=== Model evaluation ===")
        overall = evaluate_overall(df_use, model_len, model_ang)
        print("\n-- Overall (train set) --")
        print(overall.round(4))

        per_struct = evaluate_per_structure(df_raw, model_len, model_ang)
        print("\n-- Per-structure (first 10 rows) --")
        print(per_struct.head(10).round(4))

        cv_df = None
        loso_overall = None
        loso_detail = None

        if args.cv.isdigit() and int(args.cv) > 1:
            k = int(args.cv)
            cv_df = evaluate_kfold_cv(df_use, k=k, model_builder=builder)
            print(f"\n-- {k}-fold CV --")
            print(cv_df.round(4))

        elif args.cv.strip().lower() == "loso":
            print("\n-- Leave-One-Structure-Out (LOSO) CV --")
            loso_overall, loso_detail = evaluate_loso_cv(df_use, model_builder=builder)
            print("\n[LOSO overall]")
            print(loso_overall.round(4))
            print("\n[LOSO per-structure sample]")
            print(loso_detail.head(12).round(4))

        if args.save_report:
            with pd.ExcelWriter(args.save_report) as xl:
                overall.to_excel(xl, index=False, sheet_name="overall")
                per_struct.to_excel(xl, index=False, sheet_name="per_structure")
                if cv_df is not None:
                    cv_df.to_excel(xl, index=False, sheet_name="kfold_cv")
                if loso_overall is not None:
                    loso_overall.to_excel(xl, index=False, sheet_name="loso_cv_overall")
                if loso_detail is not None:
                    loso_detail.to_excel(xl, index=False, sheet_name="loso_cv_detail")
                info = pd.DataFrame(
                    {
                        "length_model": [args.length_model],
                        "len_huber_alpha": [args.len_huber_alpha],
                        "len_huber_eps": [args.len_huber_eps],
                        "len_huber_max_iter": [args.len_huber_max_iter],
                        "scale_length": [args.scale_length],
                        "len_rf_n_estimators": [args.len_rf_n_est],
                        "len_rf_max_depth": [args.len_rf_max_depth],
                        "len_rf_min_samples_leaf": [args.len_rf_min_leaf],
                        "len_rf_max_features": [args.len_rf_max_features],
                        "len_rf_criterion": [args.len_rf_criterion],
                        "len_jacobian_mode": [
                            (
                                "chain"
                                if (
                                    getattr(args, "len_add_ratios", False)
                                    or getattr(args, "len_add_sincos", False)
                                )
                                else "linear"
                            )
                        ],
                        "len_add_ratios": [args.len_add_ratios],
                        "len_add_sincos": [args.len_add_sincos],
                        "add_interactions": [args.add_interactions],
                        "angle_model": [args.angle_model],
                        "angle_poly": [args.angle_poly],
                        "angle_ridge_or_alpha": [args.angle_ridge],
                        "rf_n_estimators": [args.rf_n_est],
                        "rf_max_depth": [args.rf_max_depth],
                        "rf_min_samples_leaf": [args.rf_min_leaf],
                        "add_angle_sincos": [args.add_angle_sincos],
                        "add_ratios": [args.add_ratios],
                        "cv": [args.cv],
                    }
                )
                info.to_excel(xl, index=False, sheet_name="model_info")
            print(f"\n[Saved] report -> {args.save_report}")


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
