from typing import Optional, Dict, Tuple, List
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

# === 特徵定義 ===
FEATURES = [
    "Design_s1(mm)",
    "Design_s2(mm)",
    "Design_s3(mm)",
    "Design_a3(deg)",
    "Design_a1(deg)",
    "Design_a2(deg)",
]


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
        for s_i, s_j in combinations(s_features.keys(), 2):
            final_feats.append(s_features[s_i] * s_features[s_j])
            final_names.append(f"{s_i}*{s_j}")
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


class ModelHuber:
    """一個通用的 Huber 迴歸模型類別，封裝了特徵工程和縮放。"""

    def __init__(
        self,
        scale: bool,
        alpha: float,
        epsilon: float,
        max_iter: int,
        add_ratios: bool = False,
        add_sincos: bool = False,
        add_interactions: bool = False,
        add_aa_interact: bool = False,
    ):
        self.scaler: Optional[StandardScaler] = None
        self.scale = scale
        self.model = HuberRegressor(
            alpha=alpha, epsilon=epsilon, max_iter=int(max_iter)
        )
        self.add_ratios = add_ratios
        self.add_sincos = add_sincos
        self.add_interactions = add_interactions
        self.add_aa_interact = add_aa_interact
        self.feature_names_: List[str] = []

    def _augment(self, X_raw: np.ndarray, fit_mode: bool = False) -> np.ndarray:
        if fit_mode:
            X_aug, names = augment_feats_for_lengths_custom_interactions(
                X_raw,
                add_ratios=self.add_ratios,
                add_sincos=self.add_sincos,
                return_names=True,
                add_interactions=self.add_interactions,
                add_aa_interact=self.add_aa_interact,
            )
            self.feature_names_ = names
            return X_aug
        else:
            return augment_feats_for_lengths_custom_interactions(
                X_raw,
                add_ratios=self.add_ratios,
                add_sincos=self.add_sincos,
                add_interactions=self.add_interactions,
                add_aa_interact=self.add_aa_interact,
            )

    def fit(self, df: pd.DataFrame, target: str):
        d = df[FEATURES + [target]].dropna().copy()
        X_raw = d[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw, fit_mode=True)

        if self.scale:
            self.scaler = StandardScaler()
            X_aug = self.scaler.fit_transform(X_aug)

        self.model.fit(X_aug, d[target].to_numpy(dtype=float))

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        X_raw = df_features[FEATURES].to_numpy(dtype=float)
        X_aug = self._augment(X_raw, fit_mode=False)
        if self.scale and self.scaler:
            X_aug = self.scaler.transform(X_aug)
        return self.model.predict(X_aug)

    def get_coefficients_df(self) -> pd.DataFrame:
        if not self.feature_names_ or not hasattr(self.model, "coef_"):
            return pd.DataFrame()
        s = pd.Series(self.model.coef_, index=self.feature_names_, name="coefficient")
        s["_intercept"] = self.model.intercept_
        return s.to_frame()
