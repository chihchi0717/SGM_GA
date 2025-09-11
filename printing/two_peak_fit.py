# two_peak_fit_wide_multi_keepEXP.py
# 讀取 EXP_SIM_shrinkage_fillet.xlsx 的所有工作表 → 雙高斯 + Lambertian 擬合
# 會把原表中 **所有以 'EXP_' 開頭的欄位** 一併保留到對應的輸出工作表
# 可用 σ 下限 + 輸出端模糊 σ_out 調整「分布寬一點」

import numpy as np
import pandas as pd
from pathlib import Path
import sys, re

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

INPUT_XLSX = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202509\0910\EXP_SIM_shrinkage_fillet.xlsx"
OUTPUT_XLSX = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202509\0910\two_peak_fit_wide_multi.xlsx"

# 欄位辨識
ANGLE_COL_HINTS = ["SIM_angle", "angle", "theta"]
TARGET_KEYWORD = "sim_shrinkage"  # 目標欄名稱內含這段字（不分大小寫）
EXP_PREFIX = "EXP"  # 需要保留的欄位前綴

# 峰位置搜尋範圍（度）
MU1_RANGE = (44.0, 56.0)  # ≈ 第一峰
MU2_RANGE = (96.0, 114.0)  # ≈ 第二峰
N_MU1, N_MU2 = 61, 91

# 「寬一點」的最小 σ（度）
SIGMA1_MIN = 6.0
SIGMA2_MIN = 6.0
# 上限與網格密度
SIGMA1_MAX = 20.0
SIGMA2_MAX = 25.0
N_S1, N_S2 = 20, 20

# α、w
ALPHA_RANGE = (0.0, 0.60)
N_ALPHA = 25
W_RANGE = (0.05, 0.95)
N_W = 19

# 搜尋策略
COARSE_STRIDE = 5
REFINE_SPAN = 2

# 輸出端再模糊（整體再寬化；0=不做）
SIGMA_OUT = 0.0


# ---------- helpers ----------
def trap_area(y, x):
    if hasattr(np, "trapezoid"):  # numpy>=2.0
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def sanitize_sheet_name(name: str) -> str:
    name = re.sub(r"[:\\/?*\[\]]", "_", str(name))
    return name[:31]


def find_angle_col(cols):
    for hint in ANGLE_COL_HINTS:
        for c in cols:
            if str(c).lower() == hint.lower():
                return c
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ["angle", "theta", "deg"]):
            return c
    return cols[0]


def find_target_col(cols):
    for c in cols:
        if TARGET_KEYWORD in str(c).lower():
            return c
    raise ValueError("找不到包含 'sim_shrinkage' 的目標欄位")


def gaussian_norm(theta_deg, mu, sigma):
    theta_deg = np.asarray(theta_deg, float)
    if sigma <= 0:
        g = np.zeros_like(theta_deg)
        g[np.argmin(np.abs(theta_deg - mu))] = 1.0
    else:
        g = np.exp(-0.5 * ((theta_deg - mu) / sigma) ** 2)
    a = trap_area(g, theta_deg)
    return g / a if a > 0 else g


def lambertian_norm(theta_deg):
    th = np.deg2rad(theta_deg)
    L = np.cos(th).clip(min=0.0)
    a = trap_area(L, theta_deg)
    return L / a if a > 0 else L


def gaussian_kernel_1d(sigma_deg, step_deg):
    if sigma_deg <= 0 or step_deg <= 0:
        return np.array([1.0])
    half_n = int(np.ceil((4 * sigma_deg) / step_deg))
    xs = np.arange(-half_n, half_n + 1) * step_deg
    ker = np.exp(-0.5 * (xs / sigma_deg) ** 2)
    ker /= ker.sum()
    return ker


def blur_signal(x, y, sigma_deg):
    if sigma_deg <= 0:
        return y.copy()
    step = float(np.median(np.diff(x)))
    ker = gaussian_kernel_1d(sigma_deg, step)
    pad = len(ker) // 2
    ypad = np.pad(y, (pad, pad), mode="reflect")
    return np.convolve(ypad, ker, mode="same")[pad:-pad]


def coarse_indices(n, stride):
    idx = list(range(0, n, stride))
    if (n - 1) not in idx:
        idx.append(n - 1)
    return idx


def around_grid(val, grid, span=2):
    i = int(np.argmin(np.abs(grid - val)))
    lo, hi = max(0, i - span), min(len(grid) - 1, i + span)
    return grid[lo : hi + 1]


def fit_two_peak(
    theta,
    y,
    mu1_range,
    mu2_range,
    n_mu1,
    n_mu2,
    s1_min,
    s1_max,
    s2_min,
    s2_max,
    n_s1,
    n_s2,
    alpha_range,
    n_alpha,
    w_range,
    n_w,
    coarse_stride=5,
    refine_span=2,
):
    area = trap_area(y, theta)
    y_norm = y / (area if area > 0 else 1.0)
    L = lambertian_norm(theta)

    mu1_grid = np.linspace(*mu1_range, n_mu1)
    mu2_grid = np.linspace(*mu2_range, n_mu2)
    s1_grid = np.linspace(max(s1_min, 1e-3), s1_max, n_s1)
    s2_grid = np.linspace(max(s2_min, 1e-3), s2_max, n_s2)
    a_grid = np.linspace(*alpha_range, n_alpha)
    w_grid = np.linspace(*w_range, n_w)

    G1_bank = {
        (float(mu), float(s)): gaussian_norm(theta, mu, s)
        for mu in mu1_grid
        for s in s1_grid
    }
    G2_bank = {
        (float(mu), float(s)): gaussian_norm(theta, mu, s)
        for mu in mu2_grid
        for s in s2_grid
    }

    best = {"mse": np.inf}
    i_mu1 = coarse_indices(len(mu1_grid), coarse_stride)
    i_mu2 = coarse_indices(len(mu2_grid), coarse_stride)
    i_s1 = coarse_indices(len(s1_grid), coarse_stride)
    i_s2 = coarse_indices(len(s2_grid), coarse_stride)
    i_a = coarse_indices(len(a_grid), max(1, coarse_stride // 2))
    i_w = coarse_indices(len(w_grid), max(1, coarse_stride // 2))

    for i1 in i_mu1:
        for j1 in i_s1:
            G1 = G1_bank[(float(mu1_grid[i1]), float(s1_grid[j1]))]
            for i2 in i_mu2:
                for j2 in i_s2:
                    G2 = G2_bank[(float(mu2_grid[i2]), float(s2_grid[j2]))]
                    for wi in i_w:
                        w = float(w_grid[wi])
                        mix = w * G1 + (1 - w) * G2
                        for ai in i_a:
                            a = float(a_grid[ai])
                            model = (1 - a) * mix + a * L
                            mse = float(np.mean((model - y_norm) ** 2))
                            if mse < best["mse"]:
                                best = {
                                    "mse": mse,
                                    "mu1": float(mu1_grid[i1]),
                                    "sigma1": float(s1_grid[j1]),
                                    "mu2": float(mu2_grid[i2]),
                                    "sigma2": float(s2_grid[j2]),
                                    "w": w,
                                    "alpha": a,
                                }

    # refine
    def _ar(val, grid):
        return around_grid(val, grid, span=refine_span)

    mu1_r = _ar(best["mu1"], mu1_grid)
    s1_r = _ar(best["sigma1"], s1_grid)
    mu2_r = _ar(best["mu2"], mu2_grid)
    s2_r = _ar(best["sigma2"], s2_grid)
    a_r = _ar(best["alpha"], a_grid)
    w_r = _ar(best["w"], w_grid)

    def _G1(mu, s):
        return G1_bank.get((float(mu), float(s)), gaussian_norm(theta, mu, s))

    def _G2(mu, s):
        return G2_bank.get((float(mu), float(s)), gaussian_norm(theta, mu, s))

    for mu1 in mu1_r:
        for s1 in s1_r:
            G1 = _G1(mu1, s1)
            for mu2 in mu2_r:
                for s2 in s2_r:
                    G2 = _G2(mu2, s2)
                    for w in w_r:
                        mix = float(w) * G1 + (1 - float(w)) * G2
                        for a in a_r:
                            model = (1 - float(a)) * mix + float(a) * L
                            mse = float(np.mean((model - y_norm) ** 2))
                            if mse < best["mse"]:
                                best = {
                                    "mse": mse,
                                    "mu1": float(mu1),
                                    "sigma1": float(s1),
                                    "mu2": float(mu2),
                                    "sigma2": float(s2),
                                    "w": float(w),
                                    "alpha": float(a),
                                }

    # build curves
    G1 = gaussian_norm(theta, best["mu1"], best["sigma1"])
    G2 = gaussian_norm(theta, best["mu2"], best["sigma2"])
    L = lambertian_norm(theta)
    mix_unit = best["w"] * G1 + (1 - best["w"]) * G2
    shape_unit = (1 - best["alpha"]) * mix_unit + best["alpha"] * L

    y_fit = shape_unit * trap_area(y, theta)
    if SIGMA_OUT > 0:
        y_fit = blur_signal(theta, y_fit, SIGMA_OUT)

    params = {
        "mse": best["mse"],
        "mu1_deg": best["mu1"],
        "sigma1_deg": best["sigma1"],
        "FWHM1_deg≈": 2.354820045 * best["sigma1"],
        "mu2_deg": best["mu2"],
        "sigma2_deg": best["sigma2"],
        "FWHM2_deg≈": 2.354820045 * best["sigma2"],
        "w": best["w"],
        "alpha": best["alpha"],
        "sigma_out_deg": SIGMA_OUT,
    }

    curves_core = pd.DataFrame(
        {
            "angle_deg": theta,
            "target": y,
            "fit_2peaks": y_fit,
            "peak1_unit": G1,
            "peak2_unit": G2,
            "lambert_unit": L,
            "mixture_unit": shape_unit,
        }
    )
    return curves_core, params


def save_excel_or_csv_sheets(curves_map: dict, params_df: pd.DataFrame, xlsx_path: str):
    p = Path(xlsx_path)
    last_err = None
    for engine in [None, "openpyxl", "xlsxwriter"]:
        try:
            with pd.ExcelWriter(p, engine=engine) as w:
                for sname, df in curves_map.items():
                    df.to_excel(w, index=False, sheet_name=sname)
                params_df.to_excel(w, index=False, sheet_name="params_summary")
            print(f"Saved -> {p} (engine={engine or 'auto'})")
            return
        except Exception as e:
            last_err = e
    # fallback: CSV
    base = p.with_suffix("")
    for sname, df in curves_map.items():
        csvp = f"{base}_{sname}.csv"
        df.to_csv(csvp, index=False)
        print(f"[CSV] {csvp}")
    params_csv = f"{base}_params_summary.csv"
    params_df.to_csv(params_csv, index=False)
    print(f"[WARN] 無法寫 .xlsx（{last_err}），已改存 CSV；參數彙總：{params_csv}")


# ---------- main ----------
def main():
    book = pd.read_excel(INPUT_XLSX, sheet_name=None)
    curves_map, params_rows = {}, []

    for sheet_name, df in book.items():
        angle_col = find_angle_col(df.columns)
        target_col = find_target_col(df.columns)

        theta = df[angle_col].to_numpy(float)
        y = df[target_col].to_numpy(float)
        m = np.isfinite(theta) & np.isfinite(y)
        theta, y = theta[m], y[m]

        # ★ 保留 EXP_ 欄位（照原樣帶出，套用相同的有效列遮罩 m）
        exp_cols = [c for c in df.columns if str(c).startswith(EXP_PREFIX)]
        exp_keep = (
            df.loc[m, exp_cols].reset_index(drop=True) if exp_cols else pd.DataFrame()
        )

        curves_core, params = fit_two_peak(
            theta,
            y,
            MU1_RANGE,
            MU2_RANGE,
            N_MU1,
            N_MU2,
            SIGMA1_MIN,
            SIGMA1_MAX,
            SIGMA2_MIN,
            SIGMA2_MAX,
            N_S1,
            N_S2,
            ALPHA_RANGE,
            N_ALPHA,
            W_RANGE,
            N_W,
            COARSE_STRIDE,
            REFINE_SPAN,
        )

        # 將 EXP_ 欄位插入輸出（放在 angle/target 後面）
        if not exp_keep.empty:
            curves = pd.concat(
                [
                    curves_core[["angle_deg", "target"]],
                    exp_keep,
                    curves_core.drop(columns=["angle_deg", "target"]),
                ],
                axis=1,
            )
        else:
            curves = curves_core

        safe_name = sanitize_sheet_name(f"fit__{sheet_name}")
        curves_map[safe_name] = curves

        params_rows.append(
            {
                "sheet": sheet_name,
                "angle_col": angle_col,
                "target_col": target_col,
                **params,
            }
        )

        print(
            f"[done] {sheet_name}: mse={params['mse']:.6g}, mu1={params['mu1_deg']:.2f}, "
            f"s1={params['sigma1_deg']:.2f}, mu2={params['mu2_deg']:.2f}, s2={params['sigma2_deg']:.2f}, "
            f"alpha={params['alpha']:.3f}, w={params['w']:.3f}"
        )

    params_df = pd.DataFrame(params_rows)
    save_excel_or_csv_sheets(curves_map, params_df, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
