# -*- coding: utf-8 -*-
"""
compensation_models.py
- 包含所有用於預測長度與角度偏差的機器學習模型類別。
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor

# 從工具模組導入共享的函式與變數
try:  # 優先使用套件內的相對匯入
    from .compensation_utils import FEATURES, TARGETS, augment_feats_for_lengths
except ImportError:  # pragma: no cover - 允許作為獨立腳本執行
    from compensation_utils import FEATURES, TARGETS, augment_feats_for_lengths


def augment_feats_for_lengths_custom_interactions(
    X_raw: np.ndarray,
    add_ratios: bool,
    add_sincos: bool,
    return_names: bool = False,
    add_interactions: bool = False,
    add_aa_interact: bool = False,
):
    """
    對長度模型的特徵進行增強，交互作用項被拆分為獨立開關。
    """
    s1, s2, s3, a3, a1, a2 = (X_raw[:, i] for i in range(6))
    final_feats = [s1, s2, s3, a3, a1, a2]
    final_names = FEATURES.copy()

    if add_ratios:
        eps = 1e-9
        final_feats += [
            s1 / np.clip(s2, eps, None),
            s1 / np.clip(s3, eps, None),
            s2 / np.clip(s3, eps, None),
        ]
        final_names += ["r12", "r13", "r23"]

    if add_sincos:
        r1, r2, r3 = np.deg2rad(a1), np.deg2rad(a2), np.deg2rad(a3)
        final_feats += [
            np.sin(r1),
            np.cos(r1),
            np.sin(r2),
            np.cos(r2),
            np.sin(r3),
            np.cos(r3),
        ]
        final_names += ["sin_a1", "cos_a1", "sin_a2", "cos_a2", "sin_a3", "cos_a3"]

    s_features = {"s1": X_raw[:, 0], "s2": X_raw[:, 1], "s3": X_raw[:, 2]}
    a_features = {"a3": X_raw[:, 3], "a1": X_raw[:, 4], "a2": X_raw[:, 5]}

    if add_interactions:
        # 邊長 * 邊長
        for s_i, s_j in combinations(s_features.keys(), 2):
            final_feats.append(s_features[s_i] * s_features[s_j])
            final_names.append(f"{s_i}*{s_j}")
        # 邊長 * 角度
        for s_key in s_features:
            for a_key in a_features:
                final_feats.append(s_features[s_key] * a_features[a_key])
                final_names.append(f"{s_key}*{a_key}")

    if add_aa_interact:
        for a_i, a_j in combinations(a_features.keys(), 2):
            final_feats.append(a_features[a_i] * a_features[a_j])
            final_names.append(f"{a_i}*{a_j}")

    X_aug = np.column_stack(final_feats)

    if return_names:
        return X_aug, final_names
    return X_aug


# ---------------- 長度模型 ----------------


class LinearOLS:
    def __init__(self, ridge: float = 1e-9):
        self.ridge = ridge
        self.beta: Dict[str, np.ndarray] = {}
        self.feature_names_ = ["intercept"] + FEATURES

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

    def get_coefficients_df(self) -> Optional[pd.DataFrame]:
        if not self.beta:
            return None
        return pd.DataFrame(self.beta, index=self.feature_names_)

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        pred = {}
        for yname, b_vec in self.beta.items():
            pred[yname] = (X_ @ b_vec.reshape(-1, 1)).flatten()
        return pd.DataFrame(pred, index=df_design.index)

    def local_jacobian(self) -> np.ndarray:
        J = np.zeros((2, 6), dtype=float)
        for i, y in enumerate(["delta_s2", "delta_s3"]):
            if y in self.beta:
                J[i, :] = self.beta[y][1:]
        return J


class LengthModelRF:
    def __init__(self, **kwargs):
        self.add_ratios = bool(kwargs.get("add_ratios", False))
        self.add_sincos = bool(kwargs.get("add_sincos", False))
        self.add_interactions = bool(kwargs.get("add_interactions", False))
        self.add_aa_interact = bool(kwargs.get("add_aa_interact", False))

        rf_params = {
            k: v
            for k, v in kwargs.items()
            if k
            not in ["add_ratios", "add_sincos", "add_interactions", "add_aa_interact"]
        }
        self.model_s2 = RandomForestRegressor(**rf_params)
        self.model_s3 = RandomForestRegressor(**rf_params)
        self._ref_point_: Optional[np.ndarray] = None

    def _X(self, df):
        X_raw = df[FEATURES].to_numpy(dtype=float)
        return augment_feats_for_lengths_custom_interactions(
            X_raw,
            self.add_ratios,
            self.add_sincos,
            add_interactions=self.add_interactions,
            add_aa_interact=self.add_aa_interact,
        )

    def fit(self, df):
        X = self._X(df)
        self.model_s2.fit(X, df["delta_s2"].to_numpy(dtype=float))
        self.model_s3.fit(X, df["delta_s3"].to_numpy(dtype=float))
        self._ref_point_ = df[FEATURES].to_numpy(dtype=float).mean(axis=0)
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

    def local_jacobian_numeric(self, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        def f(xx: np.ndarray) -> np.ndarray:
            df_tmp = pd.DataFrame([dict(zip(FEATURES, xx))])
            preds = self.predict_df(df_tmp)
            return preds[["delta_s2", "delta_s3"]].to_numpy().flatten()

        J = np.zeros((2, 6), dtype=float)
        for j in range(6):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (f(xp) - f(xm)) / (2 * h)
        return J

    def local_jacobian(self) -> np.ndarray:
        if self._ref_point_ is None:
            raise RuntimeError("Model must be fit() before computing Jacobian.")
        return self.local_jacobian_numeric(self._ref_point_)


class LengthModelHuber:
    def __init__(
        self,
        alpha=1e-3,
        epsilon=1.35,
        max_iter=2000,
        scale=True,
        add_ratios=False,
        add_sincos=False,
        add_interactions=False,
        add_aa_interact=False,
    ):
        self.scale = bool(scale)
        self.add_ratios = bool(add_ratios)
        self.add_sincos = bool(add_sincos)
        self.add_interactions = bool(add_interactions)
        self.add_aa_interact = bool(add_aa_interact)

        huber_params = {
            "alpha": float(alpha),
            "epsilon": float(epsilon),
            "max_iter": int(max_iter),
        }
        self.model_s2 = HuberRegressor(**huber_params)
        self.model_s3 = HuberRegressor(**huber_params)
        self.scaler: Optional[StandardScaler] = None
        self._aug_names: Optional[List[str]] = None
        self._ref_point_: Optional[np.ndarray] = None

    def _augment(self, X_raw: np.ndarray, fit_names: bool = False) -> np.ndarray:
        if fit_names or self._aug_names is None:
            X_aug, names = augment_feats_for_lengths_custom_interactions(
                X_raw,
                self.add_ratios,
                self.add_sincos,
                True,
                self.add_interactions,
                add_aa_interact=self.add_aa_interact,
            )
            self._aug_names = list(names)
        else:
            X_aug = augment_feats_for_lengths_custom_interactions(
                X_raw,
                self.add_ratios,
                self.add_sincos,
                False,
                self.add_interactions,
                add_aa_interact=self.add_aa_interact,
            )
        return X_aug

    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_raw = df[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw, fit_names=fit)
        if self.scale:
            if fit or self.scaler is None:
                self.scaler = StandardScaler().fit(X_aug)
            X_aug = self.scaler.transform(X_aug)
        return X_aug

    def fit(self, df: pd.DataFrame):
        X_aug = self._prepare_X(df, fit=True)
        self._ref_point_ = df[FEATURES].to_numpy(dtype=float).mean(axis=0)
        self.model_s2.fit(X_aug, df["delta_s2"].to_numpy(dtype=float))
        self.model_s3.fit(X_aug, df["delta_s3"].to_numpy(dtype=float))
        return self

    def get_coefficients_df(self) -> Optional[pd.DataFrame]:
        if not self._aug_names:
            return None
        coef_data = {}
        for model, name in [(self.model_s2, "delta_s2"), (self.model_s3, "delta_s3")]:
            series = pd.Series(model.coef_, index=self._aug_names)
            series["intercept"] = model.intercept_
            coef_data[name] = series
        return pd.DataFrame(coef_data).fillna(0)

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X_aug = self._prepare_X(df_design, fit=False)
        return pd.DataFrame(
            {
                "delta_s2": self.model_s2.predict(X_aug),
                "delta_s3": self.model_s3.predict(X_aug),
            },
            index=df_design.index,
        )

    def _aug_jac_wrt_base(self, x_raw: np.ndarray) -> np.ndarray:
        assert self._aug_names is not None
        feat_vals = {name: val for name, val in zip(FEATURES, x_raw)}
        rows, deg2rad, eps = [], np.pi / 180.0, 1e-9
        for aug_name in self._aug_names:
            v = np.zeros(6, dtype=float)
            if aug_name in FEATURES:
                v[FEATURES.index(aug_name)] = 1.0
            elif aug_name == "r12":
                v[0] = 1.0 / (feat_vals["Design_s2(mm)"] or eps)
                v[1] = -feat_vals["Design_s1(mm)"] / (feat_vals["Design_s2(mm)"] ** 2)
            elif aug_name == "r13":
                v[0] = 1.0 / (feat_vals["Design_s3(mm)"] or eps)
                v[2] = -feat_vals["Design_s1(mm)"] / (feat_vals["Design_s3(mm)"] ** 2)
            elif aug_name == "r23":
                v[1] = 1.0 / (feat_vals["Design_s3(mm)"] or eps)
                v[2] = -feat_vals["Design_s2(mm)"] / (feat_vals["Design_s3(mm)"] ** 2)
            elif aug_name == "sin_a1":
                v[4] = np.cos(np.deg2rad(feat_vals["Design_a1(deg)"])) * deg2rad
            elif aug_name == "cos_a1":
                v[4] = -np.sin(np.deg2rad(feat_vals["Design_a1(deg)"])) * deg2rad
            elif aug_name == "sin_a2":
                v[5] = np.cos(np.deg2rad(feat_vals["Design_a2(deg)"])) * deg2rad
            elif aug_name == "cos_a2":
                v[5] = -np.sin(np.deg2rad(feat_vals["Design_a2(deg)"])) * deg2rad
            elif aug_name == "sin_a3":
                v[3] = np.cos(np.deg2rad(feat_vals["Design_a3(deg)"])) * deg2rad
            elif aug_name == "cos_a3":
                v[3] = -np.sin(np.deg2rad(feat_vals["Design_a3(deg)"])) * deg2rad
            elif "*" in aug_name:
                parts = aug_name.split("*")
                var1_short, var2_short = parts
                var1_full = (
                    f"Design_{var1_short}(mm)"
                    if "s" in var1_short
                    else f"Design_{var1_short}(deg)"
                )
                var2_full = (
                    f"Design_{var2_short}(mm)"
                    if "s" in var2_short
                    else f"Design_{var2_short}(deg)"
                )
                if var1_full in FEATURES and var2_full in FEATURES:
                    v[FEATURES.index(var1_full)] = feat_vals[var2_full]
                    v[FEATURES.index(var2_full)] = feat_vals[var1_full]
            rows.append(v)
        return np.vstack(rows)

    def _jac_at_point(self, x_raw: np.ndarray) -> np.ndarray:
        def _coef(model):
            if self.scale and self.scaler:
                std = np.where(self.scaler.scale_ < 1e-9, 1.0, self.scaler.scale_)
                return model.coef_.astype(float) / std
            return model.coef_.astype(float)

        g2_aug, g3_aug = _coef(self.model_s2).ravel(), _coef(self.model_s3).ravel()
        D = self._aug_jac_wrt_base(x_raw)
        return np.vstack([D.T @ g2_aug, D.T @ g3_aug])

    def local_jacobian(self) -> np.ndarray:
        if self._ref_point_ is None:
            raise RuntimeError("Fit model first.")
        if (
            self.add_ratios
            or self.add_sincos
            or self.add_interactions
            or self.add_aa_interact
        ):
            return self._jac_at_point(self._ref_point_)

        def _coef(model):
            if self.scale and self.scaler:
                std = np.where(self.scaler.scale_ < 1e-9, 1.0, self.scaler.scale_)
                return (model.coef_.astype(float) / std).ravel()
            return model.coef_.astype(float).ravel()

        g2, g3 = (_coef(self.model_s2), _coef(self.model_s3))
        return np.vstack([g2, g3])


# ---------------- 角度模型 ----------------


class AngleModelBase:
    def __init__(
        self,
        add_sincos: bool = True,
        add_ratios: bool = False,
        add_interactions: bool = False,
        add_aa_interact: bool = False,
    ):
        self.add_sincos = bool(add_sincos)
        self.add_ratios = bool(add_ratios)
        self.add_interactions = bool(add_interactions)
        self.add_aa_interact = bool(add_aa_interact)
        self.feat_names_: List[str] = []

    def _augment(self, X: np.ndarray, fit_transformer: bool = False) -> np.ndarray:
        """
        [修正] 使用與長度模型完全相同的特徵增強邏輯，以確保一致性。
        """
        # 呼叫與長度模型相同的函式來產生特徵
        X_aug, names = augment_feats_for_lengths_custom_interactions(
            X,
            add_ratios=self.add_ratios,
            add_sincos=self.add_sincos,
            return_names=True,
            add_interactions=self.add_interactions,
            add_aa_interact=self.add_aa_interact,
        )
        self.feat_names_ = names
        return X_aug

    def local_jacobian_numeric(self, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        def f(xx: np.ndarray) -> float:
            df_tmp = pd.DataFrame([dict(zip(FEATURES, xx))])
            return float(self.predict_df(df_tmp)["DIP_a3(deg)"].iloc[0])

        J = np.zeros(6, dtype=float)
        for j in range(6):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            J[j] = (f(xp) - f(xm)) / (2 * h)
        return J


class AngleModelOLS(AngleModelBase):
    def __init__(
        self,
        degree=2,
        ridge=1e-2,
        add_sincos=True,
        add_ratios=False,
        add_interactions=False,
        add_aa_interact=False,
    ):
        super().__init__(add_sincos, add_ratios, add_interactions, add_aa_interact)
        self.ridge = float(ridge)
        self.beta: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X_aug = self._augment(d[FEATURES].to_numpy(dtype=float))
        Phi = np.hstack([np.ones((X_aug.shape[0], 1)), X_aug])
        y = d["DIP_a3(deg)"].to_numpy(dtype=float).reshape(-1, 1)
        K = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        self.beta = (np.linalg.inv(K) @ (Phi.T @ y)).ravel()

    def get_coefficients_df(self) -> Optional[pd.DataFrame]:
        if self.beta is None:
            return None
        s = pd.Series(self.beta, index=["intercept"] + self.feat_names_)
        return s.to_frame(name="DIP_a3(deg)")

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X_aug = self._augment(df_design[FEATURES].to_numpy(dtype=float))
        Phi = np.hstack([np.ones((X_aug.shape[0], 1)), X_aug])
        yhat = (Phi @ self.beta) if self.beta is not None else np.zeros(len(df_design))
        return pd.DataFrame({"DIP_a3(deg)": yhat}, index=df_design.index)


class AngleModelHuber(AngleModelBase):
    def __init__(
        self,
        alpha=1e-3,
        epsilon=1.35,
        max_iter=2000,
        scale=True,
        add_sincos=True,
        add_ratios=False,
        add_interactions=False,
        add_aa_interact=False,
    ):
        super().__init__(add_sincos, add_ratios, add_interactions, add_aa_interact)
        self.scale = bool(scale)
        self.scaler: Optional[StandardScaler] = None
        self.model = HuberRegressor(
            alpha=float(alpha), epsilon=float(epsilon), max_iter=int(max_iter)
        )

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X_aug = self._augment(d[FEATURES].to_numpy(dtype=float))
        if self.scale:
            self.scaler = StandardScaler().fit(X_aug)
            X_aug = self.scaler.transform(X_aug)
        self.model.fit(X_aug, d["DIP_a3(deg)"].to_numpy(dtype=float))

    def get_coefficients_df(self) -> Optional[pd.DataFrame]:
        if not hasattr(self.model, "coef_"):
            return None
        s = pd.Series(self.model.coef_, index=self.feat_names_)
        s["intercept"] = self.model.intercept_
        return s.to_frame(name="DIP_a3(deg)")

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X_aug = self._augment(df_design[FEATURES].to_numpy(dtype=float))
        if self.scale and self.scaler:
            X_aug = self.scaler.transform(X_aug)
        return pd.DataFrame(
            {"DIP_a3(deg)": self.model.predict(X_aug)}, index=df_design.index
        )


class AngleModelRF(AngleModelBase):
    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        add_sincos=True,
        add_ratios=False,
        add_interactions=False,
        add_aa_interact=False,
    ):
        super().__init__(add_sincos, add_ratios, add_interactions, add_aa_interact)
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X_aug = self._augment(d[FEATURES].to_numpy(dtype=float))
        self.model.fit(X_aug, d["DIP_a3(deg)"].to_numpy(dtype=float))

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X_aug = self._augment(df_design[FEATURES].to_numpy(dtype=float))
        return pd.DataFrame(
            {"DIP_a3(deg)": self.model.predict(X_aug)}, index=df_design.index
        )
