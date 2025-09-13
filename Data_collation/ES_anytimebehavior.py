# -*- coding: utf-8 -*-
"""
Anytime behavior for ES (μ+λ, unique-design #evals)
Outputs:
(A) Method robustness: per-run curves + Mean ± Std-Error (shaded)
(B) Population diversity: per-run (mean ± std) in 1×3 small-multiples
(C) Step-size σ: per-run (1×3) and Mean σ ± Std-Error (shaded), if column 'sigma' exists
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= Settings =========
BASE = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\runs"
RUN_GLOB = "run*"
KEY_COLS = ["Design_S2", "Design_S3", "Design_A3"]
ROUND_DECIMALS = 5
POP_STATS = "all"  # "all" or "evaluated_only" for per-generation stats
SAVE_DPI = 1000

# Robustness figure target metric
Y_KEY = "best"  # 'best' / 'max' / 'mean' / 'std'

# Zoom (export as a separate file)
EXPORT_ZOOM = False
ZOOM_FRAC = 0.30  # first 30% of common range
ZOOM_MIN_EVALS = 300  # or at least 300 evals

# Optional smoothing (moving average on display only; 1 = off)
SMOOTH_K = 1  # recommend 1 or 3

# 彙整多個 sigma 欄位的策略
SIGMA_PATTERN = r"^sigma\d+$"  # 會自動抓 sigma1、sigma2、...
SIGMA_AGG = "geomean"  # 'geomean' | 'rms' | 'mean' | 'max'

# ===========================


def parse_gen(p: str) -> int:
    m = re.search(r"gen(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 10**9


def keys_from_df(df: pd.DataFrame):
    arr = np.round(df[KEY_COLS].to_numpy(dtype=float), ROUND_DECIMALS)
    return list(map(tuple, arr))


def moving_avg(y, k=1):
    if k <= 1:
        return y
    y = np.asarray(y, float)
    kernel = np.ones(k, float) / k
    out = np.convolve(np.nan_to_num(y, nan=np.nan), kernel, mode="same")
    # keep NaNs at leading/trailing positions where not enough points
    half = k // 2
    out[:half] = y[:half]
    out[-half or None :] = y[-half or None :]
    return out


def load_one_run(run_dir: str, pattern: str = "fitness_gen*_max*.csv"):
    """Return one run's series on cumulative #evals of unique designs."""
    files = sorted(glob.glob(os.path.join(run_dir, pattern)), key=parse_gen)
    if not files:
        raise FileNotFoundError(run_dir)

    first_gen = min(
        parse_gen(f) for f in files if re.search(r"gen(\d+)", os.path.basename(f))
    )
    seen_keys = set()
    cum_evals = []
    max_list, best_list, mean_list, std_list, sigma_list = [], [], [], [], []
    best_so_far = -np.inf
    cum = 0

    for f in files:
        g = parse_gen(f)
        df = pd.read_csv(f)

        # fitness & role columns
        if "fitness" not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                continue
            df["fitness"] = df[num_cols[0]]
        if "role" not in df.columns:
            df["role"] = "child"

        # first gen: parent_old + child; later: child only
        cand = (
            df[df["role"].isin(["parent_old", "child"])]
            if g == first_gen
            else df[df["role"] == "child"]
        )

        # unique-design filter (within-gen unique & across-gen new)
        if not set(KEY_COLS).issubset(cand.columns):
            raise KeyError(f"Missing {KEY_COLS} in {f}")
        cand_keys = keys_from_df(cand)
        unique_ordered, seen_this_gen = [], set()
        for k in cand_keys:
            if k not in seen_this_gen:
                unique_ordered.append(k)
                seen_this_gen.add(k)
        new_keys = [k for k in unique_ordered if k not in seen_keys]
        n_new = len(new_keys)
        cum += n_new
        cum_evals.append(cum)
        seen_keys.update(new_keys)

        # per-generation stats set
        if POP_STATS == "evaluated_only":
            if n_new > 0:
                mask = np.zeros(len(cand), dtype=bool)
                new_set, once = set(new_keys), set()
                for i, k in enumerate(cand_keys):
                    if (k in new_set) and (k not in once):
                        mask[i] = True
                        once.add(k)
                use_vals = (
                    pd.to_numeric(cand.loc[mask, "fitness"], errors="coerce")
                    .dropna()
                    .to_numpy()
                )
            else:
                use_vals = np.array([], dtype=float)
        else:
            use_vals = pd.to_numeric(df["fitness"], errors="coerce").dropna().to_numpy()

        if use_vals.size == 0:
            gmax = gmean = gstd = np.nan
        else:
            gmax = float(np.max(use_vals))
            gmean = float(np.mean(use_vals))
            gstd = float(np.std(use_vals, ddof=1)) if use_vals.size > 1 else 0.0

        max_list.append(gmax)
        mean_list.append(gmean)
        std_list.append(gstd)
        if not np.isnan(gmax):
            best_so_far = max(best_so_far, gmax)
        best_list.append(best_so_far)

        # sigma of this generation (if provided; CMA-ES usually one sigma per gen)
        sigma_list.append(aggregate_sigma_from_df(df))

    return {
        "x": np.array(cum_evals, float),
        "max": np.array(max_list, float),
        "best": np.array(best_list, float),
        "mean": np.array(mean_list, float),
        "std": np.array(std_list, float),
        "sigma": np.array(sigma_list, float),
    }


def step_align(x_src, y_src, x_grid):
    """Right-step align (piecewise constant)."""
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    xg = np.asarray(x_grid, float)
    idx = np.searchsorted(x_src, xg, side="right") - 1
    y = np.full_like(xg, np.nan, dtype=float)
    valid = idx >= 0
    y[valid] = y_src[idx[valid]]
    return y


def safe_std_err(Y):
    """Compute mean, std, stderr across runs without ddof warnings."""
    mask = ~np.isnan(Y)
    n = mask.sum(axis=0).astype(float)
    Y_filled = np.where(mask, Y, 0.0)
    mean = np.divide(Y_filled.sum(axis=0), n, out=np.zeros_like(n), where=n > 0)
    sqdiff = np.where(mask, (Y - mean) ** 2, 0.0)
    ss = sqdiff.sum(axis=0)
    var = np.divide(ss, np.maximum(n - 1, 1), out=np.zeros_like(ss), where=n > 1)
    std = np.sqrt(var)
    stderr = np.divide(std, np.sqrt(n), out=np.zeros_like(std), where=n > 0)
    return mean, std, stderr, n

import re


def aggregate_sigma_from_df(df) -> float:
    """
    從一個世代的 DataFrame 取出 sigma 們並彙整成單一標量。
    支援：sigma、sigma1..sigmaK；自動忽略 NaN/<=0 的值（避免 log 問題）。
    """
    # 先找 sigma1..sigmaK
    cols = [c for c in df.columns if re.match(SIGMA_PATTERN, c, flags=re.IGNORECASE)]
    if not cols and "sigma" in df.columns:
        cols = ["sigma"]

    if not cols:
        return np.nan

    A = pd.to_numeric(df[cols].values.reshape(-1), errors="coerce")
    A = A[np.isfinite(A) & (A > 0)]
    if A.size == 0:
        return np.nan

    if SIGMA_AGG.lower() == "geomean":
        return float(np.exp(np.mean(np.log(A))))
    elif SIGMA_AGG.lower() == "rms":
        return float(np.sqrt(np.mean(A**2)))
    elif SIGMA_AGG.lower() == "mean":
        return float(np.mean(A))
    elif SIGMA_AGG.lower() == "max":
        return float(np.max(A))
    else:
        # 預設用幾何平均
        return float(np.exp(np.mean(np.log(A))))


def main():
    # -------- load runs --------
    run_dirs = [d for d in glob.glob(os.path.join(BASE, RUN_GLOB)) if os.path.isdir(d)]
    if not run_dirs:
        raise RuntimeError(f"No run directories under: {BASE}")

    series, names = [], []
    for rd in sorted(run_dirs):
        s = load_one_run(rd)
        if len(s["x"]) == 0:
            continue
        series.append(s)
        names.append(os.path.basename(rd))

    # common x grid (union of #evals)
    all_x = np.array(sorted(set(np.concatenate([s["x"] for s in series]))), float)

    # ===== (A) Robustness: Mean ± Std-Error band =====
    Y = np.vstack([step_align(s["x"], s[Y_KEY], all_x) for s in series])
    mean_curve, std_curve, stderr, n_eff = safe_std_err(Y)

    full_mask = n_eff == len(series)
    full_end = (
        int(np.max(np.where(full_mask)[0])) if np.any(full_mask) else len(all_x) - 1
    )

    plt.figure(figsize=(10, 5.6))
    for nm, s in zip(names, series):
        y = step_align(s["x"], s[Y_KEY], all_x)
        m = all_x <= s["x"][-1]
        plt.step(
            all_x[m],
            moving_avg(y[m], SMOOTH_K),
            where="post",
            linestyle="--",
            alpha=0.55,
            label=nm,
        )
    plt.step(
        all_x[: full_end + 1],
        moving_avg(mean_curve[: full_end + 1], SMOOTH_K),
        where="post",
        color="crimson",
        lw=2.2,
        label=f"Mean {Y_KEY.capitalize()}",
    )
    if full_end < len(all_x) - 1:
        plt.step(
            all_x[full_end:],
            moving_avg(mean_curve[full_end:], SMOOTH_K),
            where="post",
            color="crimson",
            lw=2.2,
            linestyle=":",
            label="Mean (partial data)",
        )
    lo = mean_curve[: full_end + 1] - stderr[: full_end + 1]
    hi = mean_curve[: full_end + 1] + stderr[: full_end + 1]
    plt.fill_between(
        all_x[: full_end + 1],
        lo,
        hi,
        step="post",
        color="crimson",
        alpha=0.18,
        label="Std Error",
    )
    plt.title(f"{Y_KEY.capitalize()} vs #evaluations ({len(names)} runs)")
    plt.xlabel("Number of evaluations (unique designs, cumulative)")
    plt.ylabel("Fitness")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_png = os.path.join(BASE, f"online_{Y_KEY}_3runs_mean_stderr_band.png")
    out_pdf = os.path.join(BASE, f"online_{Y_KEY}_3runs_mean_stderr_band.pdf")
    plt.savefig(out_png, dpi=SAVE_DPI)
    plt.savefig(out_pdf)
    plt.close()

    if EXPORT_ZOOM and np.any(full_mask):
        full_x = all_x[: full_end + 1]
        x_frac_idx = max(1, int(np.floor(ZOOM_FRAC * len(full_x))))
        x_eval_idx = int(np.searchsorted(full_x, ZOOM_MIN_EVALS, side="right") - 1)
        z_end = min(max(x_frac_idx, x_eval_idx), full_end)
        z = np.arange(0, z_end + 1, dtype=int)
        plt.figure(figsize=(7.2, 4.6))
        for nm, s in zip(names, series):
            y = step_align(s["x"], s[Y_KEY], all_x)
            plt.step(
                all_x[z],
                moving_avg(y[z], SMOOTH_K),
                where="post",
                linestyle="--",
                alpha=0.55,
                label=nm,
            )
        plt.step(
            all_x[z],
            moving_avg(mean_curve[z], SMOOTH_K),
            where="post",
            color="crimson",
            lw=2.2,
            label=f"Mean {Y_KEY.capitalize()}",
        )
        plt.fill_between(
            all_x[z],
            (mean_curve[z] - stderr[z]),
            (mean_curve[z] + stderr[z]),
            step="post",
            color="crimson",
            alpha=0.20,
            label="Std Error",
        )
        plt.title(f"{Y_KEY.capitalize()} vs #evaluations (zoomed)")
        plt.xlabel("Number of evaluations (unique designs, cumulative)")
        plt.ylabel("Fitness")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(BASE, f"online_{Y_KEY}_3runs_mean_stderr_band_zoom.png"),
            dpi=SAVE_DPI,
        )
        plt.savefig(
            os.path.join(BASE, f"online_{Y_KEY}_3runs_mean_stderr_band_zoom.pdf")
        )
        plt.close()

    
    # ===== (C) Sigma: per-run (1×3) and Mean ± Std-Error (if available) =====
    has_sigma = any(np.isfinite(s["sigma"]).any() for s in series)
    if has_sigma:
        # Mean σ ± Std-Error (shaded)
        Ysig = np.vstack([step_align(s["x"], s["sigma"], all_x) for s in series])
        mean_sig, std_sig, se_sig, n_sig = safe_std_err(Ysig)
        full_mask_sig = n_sig == len(series)
        full_end_sig = (
            int(np.max(np.where(full_mask_sig)[0]))
            if np.any(full_mask_sig)
            else len(all_x) - 1
        )

        plt.figure(figsize=(10, 5.2))
        for nm, s in zip(names, series):
            y = step_align(s["x"], s["sigma"], all_x)
            m = all_x <= s["x"][-1]
            plt.step(
                all_x[m],
                moving_avg(y[m], SMOOTH_K),
                where="post",
                linestyle="--",
                alpha=0.55,
                label=nm,
            )
        plt.step(
            all_x[: full_end_sig + 1],
            moving_avg(mean_sig[: full_end_sig + 1], SMOOTH_K),
            where="post",
            color="purple",
            lw=2.2,
            label="Mean σ",
        )
        if full_end_sig < len(all_x) - 1:
            plt.step(
                all_x[full_end_sig:],
                moving_avg(mean_sig[full_end_sig:], SMOOTH_K),
                where="post",
                color="purple",
                lw=2.2,
                linestyle=":",
                label="Mean σ (partial)",
            )
        plt.fill_between(
            all_x[: full_end_sig + 1],
            mean_sig[: full_end_sig + 1] - se_sig[: full_end_sig + 1],
            mean_sig[: full_end_sig + 1] + se_sig[: full_end_sig + 1],
            step="post",
            color="purple",
            alpha=0.18,
            label="Std Error",
        )
        plt.title("Step-size σ vs #evaluations (Mean ± Std-Error)")
        plt.xlabel("Number of evaluations (unique designs, cumulative)")
        plt.ylabel("Step-size σ")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, "sigma_mean_stderr_band.png"), dpi=SAVE_DPI)
        plt.savefig(os.path.join(BASE, "sigma_mean_stderr_band.pdf"))
        plt.close()

    # ---- summaries
    # robustness summary CSV
    dfA = pd.DataFrame(
        {
            "evals": all_x,
            "mean": mean_curve,
            "std_err": stderr,
            "n_runs": n_eff.astype(int),
        }
    )
    for nm, s in zip(names, series):
        dfA[f"{nm}_{Y_KEY}"] = step_align(s["x"], s[Y_KEY], all_x)
    dfA.to_csv(
        os.path.join(BASE, f"online_{Y_KEY}_3runs_mean_stderr_band_summary.csv"),
        index=False,
    )

    # diversity summary CSV (per run mean/std aligned to common grid)
    dfB = pd.DataFrame({"evals": all_x})
    for nm, s in zip(names, series):
        dfB[f"{nm}_mean"] = step_align(s["x"], s["mean"], all_x)
        dfB[f"{nm}_std"] = step_align(s["x"], s["std"], all_x)
    dfB.to_csv(os.path.join(BASE, "diversity_mean_std_runs_summary.csv"), index=False)

    # sigma summary CSV
    if has_sigma:
        dfC = pd.DataFrame(
            {
                "evals": all_x,
                "mean_sigma": mean_sig,
                "se_sigma": se_sig,
                "n_runs": n_sig.astype(int),
            }
        )
        for nm, s in zip(names, series):
            dfC[f"{nm}_sigma"] = step_align(s["x"], s["sigma"], all_x)
        dfC.to_csv(os.path.join(BASE, "sigma_vs_evals_summary.csv"), index=False)


if __name__ == "__main__":
    main()
