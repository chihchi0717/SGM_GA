# -*- coding: utf-8 -*-
"""
run_regression_jacobian.py
- 長度：線性 OLS
- 角度：可選 sin/cos 特徵 + 多項式 (預設 angle-poly=2, angle-ridge=1e-2)
- Jacobian：長度用解析、角度用數值微分（中央差分）
- 支援：--average / --eval / --cv {K|loso} / --save-report
- 預補償：量測域線性化 AΔx≈b

用法範例
  # LOSO 驗證 + 報告
  python run_regression_jacobian.py --file "analysis_results0814.xlsx" --average --eval --cv loso --add-angle-sincos --angle-ridge 0.3 --save-report "model_report.xlsx"

  # 5-fold + 預設 angle-poly=2 + sin/cos
  python run_regression_jacobian.py --file "analysis_results0814.xlsx" --average --eval --cv 5 --add-angle-sincos
"""

import argparse
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from sklearn.preprocessing import PolynomialFeatures

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


# ---------------- Angle model (sin/cos + polynomial) ----------------
class AngleModel:
    """
    DIP_a3(deg) 專用：
      - 可加入 sin/cos(angle) 特徵（a1,a2,a3）
      - 多項式展開 + ridge OLS
      - Jacobian：對原始6設計參數做中央差分
    """

    def __init__(self, degree: int = 2, ridge: float = 1e-2, add_sincos: bool = True):
        self.degree = int(degree)
        self.ridge = float(ridge)
        self.add_sincos = bool(add_sincos)
        self.poly: Optional[PolynomialFeatures] = None
        self.beta: Optional[np.ndarray] = None  # (n_phi,)
        self.aug_feat_names: List[str] = []
        self.poly_feat_names: List[str] = []

    def _augment(self, X: np.ndarray) -> np.ndarray:
        if not self.add_sincos:
            self.aug_feat_names = FEATURES[:]
            return X
        s1, s2, s3 = X[:, 0], X[:, 1], X[:, 2]
        a3, a1, a2 = X[:, 3], X[:, 4], X[:, 5]
        r1, r2, r3 = np.deg2rad(a1), np.deg2rad(a2), np.deg2rad(a3)
        self.aug_feat_names = FEATURES[:] + [
            "sin(a1)",
            "cos(a1)",
            "sin(a2)",
            "cos(a2)",
            "sin(a3)",
            "cos(a3)",
        ]
        return np.column_stack(
            [
                s1,
                s2,
                s3,
                a3,
                a1,
                a2,
                np.sin(r1),
                np.cos(r1),
                np.sin(r2),
                np.cos(r2),
                np.sin(r3),
                np.cos(r3),
            ]
        )

    def _design(self, X: np.ndarray) -> np.ndarray:
        X_aug = self._augment(X)
        if self.degree > 1:
            if self.poly is None:
                self.poly = PolynomialFeatures(self.degree, include_bias=True)
                Phi = self.poly.fit_transform(X_aug)
                self.poly_feat_names = list(
                    self.poly.get_feature_names_out(self.aug_feat_names)
                )
            else:
                Phi = self.poly.transform(X_aug)
        else:
            Phi = np.hstack([np.ones((X_aug.shape[0], 1)), X_aug])
            self.poly_feat_names = ["Intercept"] + self.aug_feat_names
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

    def local_jacobian_numeric(self, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        def f(xx: np.ndarray) -> float:
            Phi = self._design(xx.reshape(1, -1))
            return float(Phi @ self.beta)

        J = np.zeros(6, dtype=float)
        for j in range(6):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            J[j] = (f(xp) - f(xm)) / (2 * h)
        return J


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


def evaluate_overall(df_used, model_len: LinearOLS, model_ang: AngleModel):
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


def evaluate_per_structure(df_raw, model_len: LinearOLS, model_ang: AngleModel):
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


def evaluate_kfold_cv(
    df_used,
    k: int,
    angle_degree: int,
    angle_ridge: float,
    add_sincos: bool,
    seed: int = 42,
):
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
        mdl_len = LinearOLS(ridge=1e-9)
        mdl_len.fit(df_tr)
        mdl_ang = AngleModel(
            degree=angle_degree, ridge=angle_ridge, add_sincos=add_sincos
        )
        mdl_ang.fit(df_tr)
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


# ---- helper for LOSO ----
def _train_pair_models(df_train, angle_degree, angle_ridge, add_sincos):
    mdl_len = LinearOLS(ridge=1e-9)
    mdl_len.fit(df_train)
    mdl_ang = AngleModel(degree=angle_degree, ridge=angle_ridge, add_sincos=add_sincos)
    mdl_ang.fit(df_train)
    return mdl_len, mdl_ang


def evaluate_loso_cv(df_used, angle_degree: int, angle_ridge: float, add_sincos: bool):
    """
    Leave-One-Structure-Out：
      - 以 FEATURES 完全相同為一個「結構」
      - 每次整組結構當測試，其餘當訓練
      - 回傳 overall 指標，以及每個結構×目標的誤差表
    若 df_used 是經 --average 的資料（一列一結構），等同 LOO。
    """
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

        mdl_len, mdl_ang = _train_pair_models(
            df_tr, angle_degree, angle_ridge, add_sincos
        )
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
        help='交叉驗證：整數 K 代表 K-fold；或用 "loso" 啟用 Leave-One-Structure-Out',
    )
    ap.add_argument("--save-report", type=str, default=None)
    ap.add_argument("--jac-at", type=str, default=None)
    # 角度模型設定（預設：poly=2, ridge=1e-2）
    ap.add_argument(
        "--angle-poly",
        type=int,
        default=2,
        help="polynomial degree for angle model (default: 2)",
    )
    ap.add_argument(
        "--angle-ridge",
        type=float,
        default=1e-2,
        help="ridge for angle model (default: 1e-2)",
    )
    ap.add_argument(
        "--add-angle-sincos",
        action="store_true",
        help="augment a1/a2/a3 with sin/cos features",
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
    model_len = LinearOLS(ridge=1e-9)
    model_len.fit(df_use)
    model_ang = AngleModel(
        degree=args.angle_poly, ridge=args.angle_ridge, add_sincos=args.add_angle_sincos
    )
    model_ang.fit(df_use)

    # 3) Jacobian at operating point
    if args.jac_at:
        xop = parse_vec6(args.jac_at)
        print("\n[Jacobian] 使用 --jac-at 指定的操作點")
    else:
        xop = np.array([0.76, 0.90, 0.826455, 52.021592, 59.0, 68.978408], dtype=float)
        print("\n[Jacobian] 使用預設示範設計作為操作點")

    J_len = model_len.local_jacobian()  # 2x6
    J_ang_row = model_ang.local_jacobian_numeric(xop)  # (6,)
    J = np.vstack([J_len, J_ang_row[None, :]])  # 3x6
    J_df = pd.DataFrame(J, index=TARGETS, columns=FEATURES)

    print("\n=== Local Jacobian (dy/dx) at operating point ===")
    print(J_df.round(6))

    # 4) 示範預測
    row0 = df_use[FEATURES].dropna().iloc[0]
    demo = pd.concat(
        [
            model_len.predict_df(row0.to_frame().T),
            model_ang.predict_df(row0.to_frame().T),
        ],
        axis=1,
    )
    print("\n=== Predict on first row of data used ===")
    print({k: float(demo.iloc[0][k]) for k in TARGETS})

    # 5) 預補償示範（與之前相同）
    s2_design = 0.900000
    s3_design = 0.826455
    a3_design = 52.021592
    delta_s2_meas = 0.16
    delta_s3_meas = 0.16
    angle_meas = 58.6
    m2_cur = s2_design - delta_s2_meas
    m3_cur = s3_design - delta_s3_meas
    ma3_cur = angle_meas
    dm2 = s2_design - m2_cur
    dm3 = s3_design - m3_cur
    dma3 = a3_design - ma3_cur
    dx = precomp_shrink_into_original(J, dm2, dm3, dma3, weights=None, allow_mask=None)
    print("\n=== Pre-compensation Δx (Design variables order) ===")
    print("Order:", FEATURES)
    print(np.round(dx, 6))
    orig = dict(zip(FEATURES, xop))
    comp = {k: (orig[k] + v) for k, v in zip(FEATURES, dx)}
    print("\n=== Compensated design (orig + Δx) ===")
    for k in FEATURES:
        print(f"{k}: {comp[k]:.6f}")

    # 6) 驗證（線性化）
    E_s2 = np.array([0, 1, 0, 0, 0, 0], float)
    E_s3 = np.array([0, 0, 1, 0, 0, 0], float)
    A = np.vstack([E_s2 - J[0, :], E_s3 - J[1, :], J[2, :]])
    b = np.array([dm2, dm3, dma3], float).reshape(-1, 1)
    dm_lin = (A @ dx.reshape(-1, 1)).flatten()
    print("\n=== Linearized verification (Δm2, Δm3, Δma3) ===")
    print("achieved ≈", np.round(dm_lin, 6))
    print("desired  =", [round(dm2, 6), round(dm3, 6), round(dma3, 6)])

    # 7) 評估 & 報告
    if args.eval or (args.cv and args.cv != "0") or args.save_report:
        print("\n=== Model evaluation ===")
        overall = evaluate_overall(df_use, model_len, model_ang)
        print("\n-- Overall (train set) --")
        print(overall.round(4))

        per_struct = evaluate_per_structure(df_raw, model_len, model_ang)
        print("\n-- Per-structure (first 10 rows) --")
        print(per_struct.head(10).round(4))

        cv_df = None
        loso_overall = None
        loso_detail = None

        # K-fold
        if args.cv.isdigit() and int(args.cv) > 1:
            k = int(args.cv)
            cv_df = evaluate_kfold_cv(
                df_use,
                k=k,
                angle_degree=args.angle_poly,
                angle_ridge=args.angle_ridge,
                add_sincos=args.add_angle_sincos,
            )
            print(f"\n-- {k}-fold CV --")
            print(cv_df.round(4))

        # LOSO
        elif args.cv.strip().lower() == "loso":
            print("\n-- Leave-One-Structure-Out (LOSO) CV --")
            loso_overall, loso_detail = evaluate_loso_cv(
                df_use,
                angle_degree=args.angle_poly,
                angle_ridge=args.angle_ridge,
                add_sincos=args.add_angle_sincos,
            )
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
                        "angle_poly": [args.angle_poly],
                        "angle_ridge": [args.angle_ridge],
                        "add_angle_sincos": [args.add_angle_sincos],
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
