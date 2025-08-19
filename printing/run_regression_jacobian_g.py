# -*- coding: utf-8 -*-
"""
run_regression_jacobian.py
- 長度：線性 OLS
- 角度：可選 ols(多項式+ridge) / rf(RandomForest) / huber
- 特徵：可選加入 sin/cos(a1,a2,a3) 與 邊長比例特徵 (s1/s2, s1/s3, s2/s3)
- Jacobian：長度用解析、角度用數值微分（中央差分）
- 驗證：--cv {K|loso}
- 報告：--save-report，角度會重點列出 MAE / P95（同時仍提供 R2 以便對照）
- 預補償：量測域線性化 AΔx≈b

建議用法
  # 角度改用 RF + sin/cos + ratios + LOSO（最貼近新幾何情境）
  python run_regression_jacobian.py --file "analysis_results0814.xlsx" --average --eval --cv loso --angle-model rf --add-angle-sincos --add-ratios --save-report "model_report.xlsx"

  # 5-fold，RF + sin/cos
  python run_regression_jacobian.py --file "analysis_results0814.xlsx" --average --eval --cv 5 --angle-model rf --add-angle-sincos
"""

import argparse
import numpy as np
import pandas as pd
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
TARGETS = ["delta_s2", "delta_s3", "DIP_a3(deg)"]


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
    """
    X: shape (n,6) 對應 FEATURES = [s1,s2,s3,a3,a1,a2]（角度為度）
    回傳: X_aug 或 (X_aug, names)
    """
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
    if add_interactions:  # <--- 新增交互作用
        inter_terms = [
            s1 * s2,
            s1 * s3,
            s2 * s3,  # 邊長交互
            a1 * a2,
            a1 * a3,
            a2 * a3,  # 角度交互
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


from sklearn.ensemble import RandomForestRegressor


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

        # 注意：這裡才把 max_features / criterion 傳進 RF
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


import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

# 需與主程式一致
FEATURES = [
    "Design_s1(mm)",
    "Design_s2(mm)",
    "Design_s3(mm)",
    "Design_a3(deg)",
    "Design_a1(deg)",
    "Design_a2(deg)",
]

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from itertools import combinations


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

    # ---- 內部：增強 + （可選）標準化 ----
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
        # For Jacobian chain rule, we need the mean of the original features
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

    # ---- 內部：aug特徵對原始6維的雅可比 D (n_aug x 6) 於某點 x_raw ----
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
        return np.vstack(rows)  # (n_aug x 6)

    # 核心：在某參考點，將「aug域係數」轉成「原始6維」的 Jacobian
    def _jac_at_point(self, x_raw: np.ndarray) -> np.ndarray:
        # 取出 Huber 線性係數（若有標準化，除以 std）
        def _coef_on_raw(model):
            if self.scale and (self.scaler is not None):
                std = self.scaler.scale_.astype(float)
                std = np.where(std == 0, 1.0, std)
                return model.coef_.astype(float) / std  # shape (n_aug,)
            return model.coef_.astype(float)

        g2_aug = _coef_on_raw(self.model_s2).ravel()  # dy2/d(aug)
        g3_aug = _coef_on_raw(self.model_s3).ravel()  # dy3/d(aug)

        D = self._aug_jac_wrt_base(x_raw)  # d(aug)/d(base)  (n_aug x 6)
        dy2_dx = D.T @ g2_aug  # shape (6,)
        dy3_dx = D.T @ g3_aug  # shape (6,)
        J = np.vstack([dy2_dx, dy3_dx])  # (2 x 6)
        return J

    def local_jacobian(self) -> np.ndarray:
        """
        若 add_ratios/sincos=False：退化為線性(標準化)係數對應的常數梯度。
        若 add_* = True：在訓練集均值 self._ref_point_ 評估鏈式 Jacobian。
        """
        if self.add_ratios or self.add_sincos:
            if self._ref_point_ is None:
                raise RuntimeError("Model must be fit() before computing Jacobian.")
            return self._jac_at_point(self._ref_point_)
        # 無增強：與舊版一致
        if self.scale and (self.scaler is not None):
            std = self.scaler.scale_.astype(float)
            std = np.where(std == 0, 1.0, std)
            g2 = (self.model_s2.coef_.astype(float) / std).ravel()
            g3 = (self.model_s3.coef_.astype(float) / std).ravel()
        else:
            g2 = self.model_s2.coef_.astype(float).ravel()
            g3 = self.model_s3.coef_.astype(float).ravel()
        # 只取前6個(原始6維)，避免萬一增強名單不同步
        g2 = g2[: len(FEATURES)]
        g3 = g3[: len(FEATURES)]
        return np.vstack([g2, g3])

    # (選用) 若你想在特定設計點算 Jacobian，可呼叫這個
    def local_jacobian_at(self, x_raw: np.ndarray) -> np.ndarray:
        """
        x_raw: 長度6的一維 ndarray/list，順序對應 FEATURES
        """
        x_raw = np.asarray(x_raw, dtype=float).ravel()
        if x_raw.shape[0] != len(FEATURES):
            raise ValueError(f"x_raw must have length {len(FEATURES)}")
        return self._jac_at_point(x_raw)


# ---------------- Angle model base (feature augmentation) ----------------
class AngleModelBase:
    """
    提供統一的特徵增維：
      - add_sincos: 針對 a1,a2,a3 產生 sin/cos
      - add_ratios: 產生 s1/s2, s1/s3, s2/s3（避免除零）
    子類別需實作：
      - fit(df)
      - predict_df(df_design) -> DataFrame({"DIP_a3(deg)": ...})
    """

    def __init__(self, add_sincos: bool = True, add_ratios: bool = False):
        self.add_sincos = bool(add_sincos)
        self.add_ratios = bool(add_ratios)
        self.feat_names_: List[str] = []  # 訓練時最後使用的特徵名稱（for調試）

    def _augment(self, X: np.ndarray) -> np.ndarray:
        # X: [s1,s2,s3,a3,a1,a2]
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
        """
        針對原始6個設計參數 (s1,s2,s3,a3,a1,a2) 做中央差分近似：
          ∂f/∂x_j ≈ [f(x+hej) - f(x-hej)]/(2h)
        """

        def f(xx: np.ndarray) -> float:
            df_tmp = pd.DataFrame([dict(zip(FEATURES, xx))])
            return float(self.predict_df(df_tmp)["DIP_a3(deg)"].iloc[0])

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
    """原本 OLS：多項式 + ridge"""

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
        y = d["DIP_a3(deg)"].to_numpy(dtype=float).reshape(-1, 1)
        K = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        self.beta = (np.linalg.inv(K) @ (Phi.T @ y)).ravel()

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        Phi = self._design(X)
        yhat = (Phi @ self.beta).ravel()
        return pd.DataFrame({"DIP_a3(deg)": yhat}, index=df_design.index)


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
        y = d["DIP_a3(deg)"].to_numpy(dtype=float)
        self.model.fit(X_aug, y)

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X)
        if self.scale and self.scaler is not None:
            X_aug = self.scaler.transform(X_aug)
        yhat = self.model.predict(X_aug)
        return pd.DataFrame({"DIP_a3(deg)": yhat}, index=df_design.index)


class AngleModelRF(AngleModelBase):
    """RandomForest：擬合非線性且對特徵尺度不敏感。"""

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
        y = d["DIP_a3(deg)"].to_numpy(dtype=float)
        self.model.fit(X_aug, y)

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X)
        yhat = self.model.predict(X_aug)
        return pd.DataFrame({"DIP_a3(deg)": yhat}, index=df_design.index)


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


# ---------------- Pre-compensation ----------------
def precomp_shrink_into_original(
    J, dm2, dm3, dma3, weights=None, allow_mask=None, ridge=1e-6
):
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
        help="長度模型加入交互作用特徵 (s1*s2, s1*s3, s2*s3, a1*a2, ...)",
    )

    # ---- 長度模型選項 ----
    ap.add_argument(
        "--length-model",
        type=str,
        default="ols",
        choices=["ols", "huber", "rf"],
        help="長度回歸器：ols/huber/rf",
    )

    ap.add_argument(
        "--len-huber-alpha",
        type=float,
        default=1e-3,
        help="(length huber) alpha，預設 1e-3",
    )
    ap.add_argument(
        "--len-huber-eps",
        type=float,
        default=1.35,
        help="(length huber) epsilon，預設 1.35",
    )
    ap.add_argument(
        "--len-huber-max-iter",
        type=int,
        default=2000,
        help="(length huber) max_iter，預設 2000",
    )
    ap.add_argument(
        "--scale-length", action="store_true", help="標準化長度模型輸入特徵"
    )
    # ---- 長度 RF 專用參數 ----
    ap.add_argument(
        "--len-rf-n-est", type=int, default=300, help="(length rf) n_estimators"
    )
    ap.add_argument(
        "--len-rf-max-depth",
        type=int,
        default=None,
        help="(length rf) max_depth (None=不限)",
    )
    ap.add_argument(
        "--len-rf-min-leaf", type=int, default=1, help="(length rf) min_samples_leaf"
    )
    ap.add_argument(
        "--len-rf-max-features",
        type=float,
        default=1.0,
        help="(length rf) max_features，建議 0.6~0.9 以抑制過擬合",
    )
    ap.add_argument(
        "--len-rf-criterion",
        type=str,
        default="squared_error",
        choices=["squared_error", "absolute_error"],
        help="(length rf) 分裂準則；absolute_error 對 MAE/P95 較友善",
    )
    # ---- 長度特徵增強 ----
    ap.add_argument(
        "--len-add-ratios",
        action="store_true",
        help="長度模型加入比例特徵：s1/s2, s1/s3, s2/s3",
    )
    ap.add_argument(
        "--len-add-sincos",
        action="store_true",
        help="長度模型加入角度正弦/餘弦特徵：sin,cos(a1,a2,a3)",
    )

    # 角度模型選擇與參數
    ap.add_argument(
        "--angle-model",
        type=str,
        default="rf",
        choices=["ols", "rf", "huber"],
        help="角度回歸器：ols(多項式+ridge)/rf(RandomForest)/huber",
    )
    ap.add_argument(
        "--angle-poly", type=int, default=2, help="(ols) polynomial degree (default 2)"
    )
    ap.add_argument(
        "--angle-ridge",
        type=float,
        default=1e-2,
        help="(ols) ridge; (huber) alpha (default 1e-2)",
    )
    ap.add_argument("--rf-n-est", type=int, default=300, help="(rf) n_estimators")
    ap.add_argument(
        "--rf-max-depth", type=int, default=None, help="(rf) max_depth (None=不限)"
    )
    ap.add_argument("--rf-min-leaf", type=int, default=1, help="(rf) min_samples_leaf")
    ap.add_argument(
        "--huber-max-iter",
        type=int,
        default=2000,
        help="(huber) max_iter, default 2000",
    )
    ap.add_argument(
        "--huber-eps", type=float, default=1.35, help="(huber) epsilon, default 1.35"
    )
    ap.add_argument(
        "--scale-angle", action="store_true", help="標準化角度模型的輸入特徵"
    )

    ap.add_argument(
        "--add-angle-sincos", action="store_true", help="增加 sin/cos(a1,a2,a3) 特徵"
    )
    ap.add_argument(
        "--add-ratios", action="store_true", help="增加比例特徵 s1/s2, s1/s3, s2/s3"
    )

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
    # === main() 內建立長度模型 ===
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

    # 3) Jacobian at operating point（角度用數值微分）
    if args.jac_at:
        xop = parse_vec6(args.jac_at)
        print("\n[Jacobian] 使用 --jac-at 指定的操作點")
    else:
        xop = df_use[FEATURES].mean().to_numpy()
        print("\n[Jacobian] 使用訓練數據的平均值作為操作點")

    J_len = model_len.local_jacobian()
    J_ang_row = model_ang.local_jacobian_numeric(xop)
    J = np.vstack([J_len, J_ang_row[None, :]])
    J_df = pd.DataFrame(J, index=TARGETS, columns=FEATURES)

    print("\n=== Local Jacobian (dy/dx) at operating point ===")
    print(J_df.round(6))

    # 4) ===== 全新：事前預測補償 (Proactive Pre-compensation) =====
    print("\n\n=== Proactive Pre-compensation Workflow ===")

    # 步驟 1: 定義您的「目標設計 (target_design)」, 也就是您最終想要的完美成品尺寸

    # --- 手動設定區塊 ---
    # 如果您有特定的目標設計，請取消以下區塊的註解 (#)，並填入您想要的6個數值。
    target_design = {
        "Design_s1(mm)": 0.42,
        "Design_s2(mm)": 0.85,
        "Design_s3(mm)": 0.9,
        "Design_a3(deg)": 28,
        "Design_a1(deg)": 83,
        "Design_a2(deg)": 69.0,
    }

    # --- 自動設定區塊 (預設使用訓練數據的平均值) ---
    # 如果上面手動設定區塊被註解掉，程式將會使用此預設值。
    try:
        target_design
    except NameError:
        print("\n[Info] 未手動設定 target_design，將使用訓練數據的平均值作為範例。")
        target_design = dict(df_use[FEATURES].mean())

    print("\n--- Step 1: Target Design ---")
    for k, v in target_design.items():
        print(f"  - {k}: {v:.6f}")

    # 步驟 2: 使用模型「預測」將會發生的誤差
    target_df = pd.DataFrame([target_design])
    predicted_deltas = model_len.predict_df(target_df)
    predicted_angle = model_ang.predict_df(target_df)

    pred_ds2_frac = predicted_deltas["delta_s2"].iloc[0]
    pred_ds3_frac = predicted_deltas["delta_s3"].iloc[0]
    pred_ma3 = predicted_angle["DIP_a3(deg)"].iloc[0]

    print("\n--- Step 2: Predicted Deviations (if using target as design) ---")
    print(f"  - Predicted s2 shrinkage: {pred_ds2_frac:.4f} ({pred_ds2_frac*100:.2f}%)")
    print(f"  - Predicted s3 shrinkage: {pred_ds3_frac:.4f} ({pred_ds3_frac*100:.2f}%)")
    print(f"  - Predicted final angle: {pred_ma3:.4f} degrees")

    # 步驟 3: 計算需要補償的目標誤差 `b`
    m2_predicted_actual = target_design["Design_s2(mm)"] * (1 - pred_ds2_frac)
    m3_predicted_actual = target_design["Design_s3(mm)"] * (1 - pred_ds3_frac)

    dm2 = target_design["Design_s2(mm)"] - m2_predicted_actual
    dm3 = target_design["Design_s3(mm)"] - m3_predicted_actual
    dma3 = target_design["Design_a3(deg)"] - pred_ma3

    print("\n--- Step 3: Calculated Error to Compensate ---")
    print(f"  - Required change in m2 (dm2): {dm2:.6f} mm")
    print(f"  - Required change in m3 (dm3): {dm3:.6f} mm")
    print(f"  - Required change in ma3 (dma3): {dma3:.6f} degrees")

    # 步驟 4: 求解並應用補償量 Δx
    # (注意: Jacobian 應該在目標點計算，以獲得最準確的線性近似)
    J_ang_row_target = model_ang.local_jacobian_numeric(
        np.array(list(target_design.values()))
    )
    J_target = np.vstack([model_len.local_jacobian(), J_ang_row_target[None, :]])

    dx = precomp_shrink_into_original(J_target, dm2, dm3, dma3)

    print("\n--- Step 4: Calculated Compensation Vector (Δx) ---")
    print("Order:", FEATURES)
    print(np.round(dx, 6))

    # 步驟 5: 產出最終的「補償後設計」
    compensated_design = {
        k: (v + dx_val) for (k, v), dx_val in zip(target_design.items(), dx)
    }

    print("\n--- Step 5: Final Compensated Design to Manufacture ---")
    for k, v in compensated_design.items():
        print(f"  - {k}: {v:.6f}")

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
                J_df.to_excel(xl, sheet_name="jacobian_at_op")
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

# python run_regression_jacobian_g.py --file "C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\DOE_RB\0.6_0.9\results_analyzed_final\analysis_results -copy.xlsx" --average --eval --cv 5 --length-model huber --len-huber-alpha 1 --len-huber-eps 1 --len-huber-max-iter 1000 --scale-length  --len-add-sincos --angle-model huber --add-ratios --add-angle-sincos --scale-angle --angle-ridge  0.000001 --huber-max-iter 1000 --save-report "C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\DOE_RB\0.6_0.9\model_report.xlsx"
