# -*- coding: utf-8 -*-
"""
compensation_models.py
- 包含所有用於預測長度與角度偏差的機器學習模型類別。
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor

# 從工具模組導入共享的函式與變數
try:  # 優先使用套件內的相對匯入
    from .compensation_utils import FEATURES, TARGETS, augment_feats_for_lengths
except ImportError:  # pragma: no cover - 允許作為獨立腳本執行
    from compensation_utils import FEATURES, TARGETS, augment_feats_for_lengths

# ---------------- 長度模型 ----------------


class LinearOLS:
    def __init__(self, ridge: float = 1e-9):
        self.ridge = ridge
        self.beta: Dict[str, np.ndarray] = {}
        # [新增] 儲存特徵名稱以便與係數對應
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
        """[新增] 將模型係數轉換為 DataFrame。"""
        if not self.beta:
            return None
        return pd.DataFrame(self.beta, index=self.feature_names_)

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X = df_design[FEATURES].to_numpy(dtype=float)
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        # [修正] 確保矩陣維度正確以進行預測
        pred = {}
        for yname, b_vec in self.beta.items():
            pred[yname] = (X_ @ b_vec.reshape(-1, 1)).flatten()
        return pd.DataFrame(pred, index=df_design.index)

    def local_jacobian(self) -> np.ndarray:
        J = np.zeros((2, 6), dtype=float)
        for i, y in enumerate(["delta_s2", "delta_s3"]):
            if y in self.beta:
                # 係數從 index 1 開始，因為 index 0 是截距
                J[i, :] = self.beta[y][1:]
        return J


class LengthModelRF:
    def __init__(self, **kwargs):
        self.add_ratios = bool(kwargs.get("add_ratios", False))
        self.add_sincos = bool(kwargs.get("add_sincos", False))
        self.add_interactions = bool(kwargs.get("add_interactions", False))
        rf_params = {
            k: v
            for k, v in kwargs.items()
            if k not in ["add_ratios", "add_sincos", "add_interactions"]
        }
        self.model_s2 = RandomForestRegressor(**rf_params)
        self.model_s3 = RandomForestRegressor(**rf_params)
        self._ref_point_: Optional[np.ndarray] = None

    def _X(self, df):
        X_raw = df[FEATURES].to_numpy(dtype=float)
        return augment_feats_for_lengths(
            X_raw,
            self.add_ratios,
            self.add_sincos,
            add_interactions=self.add_interactions,
        )

    def fit(self, df):
        X = self._X(df)
        self.model_s2.fit(X, df["delta_s2"].to_numpy(dtype=float))
        self.model_s3.fit(X, df["delta_s3"].to_numpy(dtype=float))
        # 儲存參考點以供 Jacobian 計算
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
        """
        【新增】針對 RF 模型，用數值方法計算 Jacobian。
        """

        def f(xx: np.ndarray) -> np.ndarray:
            df_tmp = pd.DataFrame([dict(zip(FEATURES, xx))])
            preds = self.predict_df(df_tmp)
            return preds[["delta_s2", "delta_s3"]].to_numpy().flatten()

        J = np.zeros((2, 6), dtype=float)
        for j in range(6):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            grad_col = (f(xp) - f(xm)) / (2 * h)
            J[:, j] = grad_col
        return J

    def local_jacobian(self) -> np.ndarray:
        """
        【更新】讓 RF 模型也能提供有效的 Jacobian。
        """
        if self._ref_point_ is None:
            raise RuntimeError("Model must be fit() before computing Jacobian for RF.")
        # 對於 RF，永遠使用數值 Jacobian，因為沒有解析解
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
    ):
        self.scale = bool(scale)
        self.add_ratios = bool(add_ratios)
        self.add_sincos = bool(add_sincos)
        self.add_interactions = bool(add_interactions)
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
            X_aug, names = augment_feats_for_lengths(
                X_raw, self.add_ratios, self.add_sincos, True, self.add_interactions
            )
            self._aug_names = list(names)
        else:
            X_aug = augment_feats_for_lengths(
                X_raw, self.add_ratios, self.add_sincos, False, self.add_interactions
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
        """[新增] 將模型係數轉換為 DataFrame。"""
        if not self._aug_names:
            return None

        coef_data = {}
        # 處理 delta_s2
        s2_coefs = self.model_s2.coef_
        s2_intercept = self.model_s2.intercept_
        s2_series = pd.Series(s2_coefs, index=self._aug_names)
        s2_series["intercept"] = s2_intercept
        coef_data["delta_s2"] = s2_series

        # 處理 delta_s3
        s3_coefs = self.model_s3.coef_
        s3_intercept = self.model_s3.intercept_
        s3_series = pd.Series(s3_coefs, index=self._aug_names)
        s3_series["intercept"] = s3_intercept
        coef_data["delta_s3"] = s3_series

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
                v[0] = 1.0 / (
                    s2 if abs(s2) > eps else np.sign(s2) * eps if s2 != 0 else eps
                )
                v[1] = -s1 / (s2**2)
            elif name == "r13":
                v[0] = 1.0 / (
                    s3 if abs(s3) > eps else np.sign(s3) * eps if s3 != 0 else eps
                )
                v[2] = -s1 / (s3**2)
            elif name == "r23":
                v[1] = 1.0 / (
                    s3 if abs(s3) > eps else np.sign(s3) * eps if s3 != 0 else eps
                )
                v[2] = -s2 / (s3**2)
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
        def _coef(model):
            if self.scale and self.scaler:
                std = np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)
                return model.coef_.astype(float) / std
            return model.coef_.astype(float)

        g2_aug, g3_aug = _coef(self.model_s2).ravel(), _coef(self.model_s3).ravel()
        D = self._aug_jac_wrt_base(x_raw)
        return np.vstack([D.T @ g2_aug, D.T @ g3_aug])

    def local_jacobian(self) -> np.ndarray:
        if self.add_ratios or self.add_sincos:
            if self._ref_point_ is None:
                raise RuntimeError("Fit model first.")
            return self._jac_at_point(self._ref_point_)

        def _coef(model):
            if self.scale and self.scaler:
                std = np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)
                return (model.coef_.astype(float) / std).ravel()
            return model.coef_.astype(float).ravel()

        g2, g3 = (
            _coef(self.model_s2)[: len(FEATURES)],
            _coef(self.model_s3)[: len(FEATURES)],
        )
        return np.vstack([g2, g3])


# ---------------- 角度模型 ----------------


class AngleModelBase:
    def __init__(
        self,
        add_sincos: bool = True,
        add_ratios: bool = False,
        add_interactions: bool = False,
    ):
        self.add_sincos = bool(add_sincos)
        self.add_ratios = bool(add_ratios)
        self.add_interactions = bool(add_interactions)
        self.feat_names_: List[str] = []
        self.poly_transformer: Optional[PolynomialFeatures] = None

    def _augment(self, X: np.ndarray, fit_transformer: bool = False) -> np.ndarray:
        # [修改] 邏輯重構以分離交互作用
        s1, s2, s3, a3, a1, a2 = (X[:, i] for i in range(6))

        # 永遠從原始特徵開始
        final_feats = [s1, s2, s3, a3, a1, a2]
        final_names = ["s1", "s2", "s3", "a3", "a1", "a2"]

        # 1. 處理僅限原始特徵的交互作用
        if self.add_interactions:
            if fit_transformer:
                self.poly_transformer = PolynomialFeatures(
                    degree=2, interaction_only=True, include_bias=False
                )
                # 只對原始 6 個特徵做 fit_transform
                interaction_features = self.poly_transformer.fit_transform(X)
                # 取得交互作用項的名稱 (例如 's1 s2')
                interaction_names = self.poly_transformer.get_feature_names_out(
                    FEATURES
                )

                # 將原始特徵和交互作用項合併
                final_feats = list(interaction_features.T)
                final_names = list(interaction_names)
            else:
                if self.poly_transformer is None:
                    raise RuntimeError("Polynomial transformer has not been fitted.")
                interaction_features = self.poly_transformer.transform(X)
                final_feats = list(interaction_features.T)
                final_names = list(
                    self.poly_transformer.get_feature_names_out(FEATURES)
                )

        # 2. 獨立加入 Ratio 特徵
        if self.add_ratios:
            eps = 1e-9
            final_feats += [
                s1 / np.clip(s2, eps, None),
                s1 / np.clip(s3, eps, None),
                s2 / np.clip(s3, eps, None),
            ]
            final_names += ["s1/s2", "s1/s3", "s2/s3"]

        # 3. 獨立加入 Sin/Cos 特徵
        if self.add_sincos:
            r1, r2, r3 = np.deg2rad(a1), np.deg2rad(a2), np.deg2rad(a3)
            final_feats += [
                np.sin(r1),
                np.cos(r1),
                np.sin(r2),
                np.cos(r2),
                np.sin(r3),
                np.cos(r3),
            ]
            final_names += [
                "sin(a1)",
                "cos(a1)",
                "sin(a2)",
                "cos(a2)",
                "sin(a3)",
                "cos(a3)",
            ]

        # 最終組合
        self.feat_names_ = final_names
        return np.column_stack(final_feats)

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
    ):
        super().__init__(add_sincos, add_ratios, add_interactions)
        self.degree = int(degree)
        self.ridge = float(ridge)
        self.poly_deg_n: Optional[PolynomialFeatures] = (
            None  # 用於處理 > 2 次方的多項式
        )
        self.beta: Optional[np.ndarray] = None
        self.feature_names_out_: List[str] = []

    def _design(self, X: np.ndarray) -> np.ndarray:
        # 使用父類別的方法產生所有特徵
        X_aug = super()._augment(X, fit_transformer=(self.beta is None))

        # 如果已經有交互作用，或者 degree <= 1，就不再做多項式展開
        if self.add_interactions or self.degree <= 1:
            self.feature_names_out_ = ["intercept"] + self.feat_names_
            return np.hstack([np.ones((X_aug.shape[0], 1)), X_aug])

        # 處理 > 2 次方的多項式
        if self.poly_deg_n is None:
            self.poly_deg_n = PolynomialFeatures(self.degree, include_bias=True)

        if self.beta is None:  # fit
            Phi = self.poly_deg_n.fit_transform(X_aug)
            self.feature_names_out_ = self.poly_deg_n.get_feature_names_out(
                self.feat_names_
            )
        else:  # predict
            Phi = self.poly_deg_n.transform(X_aug)

        return Phi

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        Phi = self._design(d[FEATURES].to_numpy(dtype=float))

        y = d["DIP_a3(deg)"].to_numpy(dtype=float).reshape(-1, 1)
        K = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        self.beta = (np.linalg.inv(K) @ (Phi.T @ y)).ravel()

    def get_coefficients_df(self) -> Optional[pd.DataFrame]:
        if self.beta is None or len(self.feature_names_out_) == 0:
            return None
        s = pd.Series(self.beta, index=self.feature_names_out_)
        return s.to_frame(name="DIP_a3(deg)")

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        Phi = self._design(df_design[FEATURES].to_numpy(dtype=float))
        yhat = (
            (Phi @ self.beta).ravel()
            if self.beta is not None
            else np.zeros(len(df_design))
        )
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
    ):
        super().__init__(add_sincos, add_ratios, add_interactions)
        self.scale = bool(scale)
        self.scaler: Optional[StandardScaler] = None
        self.model = HuberRegressor(
            alpha=float(alpha), epsilon=float(epsilon), max_iter=int(max_iter)
        )

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X_aug = self._augment(d[FEATURES].to_numpy(dtype=float), fit_transformer=True)
        if self.scale:
            self.scaler = StandardScaler().fit(X_aug)
            X_aug = self.scaler.transform(X_aug)
        self.model.fit(X_aug, d["DIP_a3(deg)"].to_numpy(dtype=float))

    def get_coefficients_df(self) -> Optional[pd.DataFrame]:
        if not hasattr(self.model, "coef_") or len(self.feat_names_) == 0:
            return None

        coefs = self.model.coef_
        intercept = self.model.intercept_
        s = pd.Series(coefs, index=self.feat_names_)
        s["intercept"] = intercept
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
    ):
        super().__init__(add_sincos, add_ratios, add_interactions)
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame):
        d = df[FEATURES + TARGETS].dropna().copy()
        X_aug = self._augment(d[FEATURES].to_numpy(dtype=float), fit_transformer=True)
        self.model.fit(X_aug, d["DIP_a3(deg)"].to_numpy(dtype=float))

    def predict_df(self, df_design: pd.DataFrame) -> pd.DataFrame:
        X_aug = self._augment(df_design[FEATURES].to_numpy(dtype=float))
        return pd.DataFrame(
            {"DIP_a3(deg)": self.model.predict(X_aug)}, index=df_design.index
        )
