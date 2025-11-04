# -*- coding: utf-8 -*-
"""
Anytime behavior for ES (μ+λ, unique-design evaluations)

Outputs
(A) Method robustness: per-run curves + Mean ± Std-Error (shaded band)
    - 主圖：只在所有 run 都有資料的「共同區段」畫實線與陰影
    - 前/後的 partial 區段以虛線表示（不寫進主 CSV）
    - 另輸出放大圖（前段）
    - Summary：online_<Y_KEY>_summary.csv（共同區段）
               online_<Y_KEY>_summary_full.csv（含 partial，方便檢查）

(B) Population diversity：每個 run 的 (mean ± std) — 1×N 小 multiples

(C) Step-size σ
    - 自動抓取 sigma1/2/3… 欄位並彙整（預設幾何平均）
    - 每 run 的 σ vs #evals（1×N）
    - Mean σ ± Std-Error（共同區段畫陰影）

Notes
- x 軸是「累積 #evaluations（不重算父母、跨世代相同設計只算一次）」。
- 族群統計 POP_STATS = "all"（整個族群）或 "evaluated_only"（本代新評估者）。
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ========= Settings =========
BASE = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\runs"  # 包含 run1, run2, ...
RUN_GLOB = "run*"
KEY_COLS = ["Design_S2", "Design_S3", "Design_A3"]
ROUND_DECIMALS = 5
POP_STATS = "all"  # "all" or "evaluated_only"
SAVE_DPI = 300

# Robustness 圖所用的 y 指標
Y_KEY = "best"  # 'best' / 'max' / 'mean' / 'std'

# Zoom（共同區段前 30% 或至少 300 evals）
EXPORT_ZOOM = True
ZOOM_FRAC = 0.30
ZOOM_MIN_EVALS = 300

# 平滑（僅顯示用；1 表示不平滑）
SMOOTH_K = 1

# σ 欄位自動彙整
SIGMA_PATTERN = r"^sigma\d+$"  # 自動抓 sigma1/2/3...
SIGMA_AGG = "geomean"  # 'geomean' | 'rms' | 'mean' | 'max'

# 顯示格線（論文常關閉）
SHOW_GRID = False
# ===========================


# === 期刊風格設定 ===
mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
        "figure.figsize": (7.5, 4),
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,  # 全域關閉格線（個別圖仍可覆蓋）
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 0.8,
    }
)
mpl.rcParams["savefig.dpi"] = SAVE_DPI


# ---------- helpers ----------
def parse_gen(p: str) -> int:
    m = re.search(r"gen(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 10**9


def keys_from_df(df: pd.DataFrame):
    arr = np.round(df[KEY_COLS].to_numpy(dtype=float), ROUND_DECIMALS)
    return list(map(tuple, arr))


def moving_avg(y: np.ndarray, k: int = 1) -> np.ndarray:
    """Centered rolling mean with pandas (handles NaN nicely)."""
    if k <= 1:
        return np.asarray(y, float)
    return (
        pd.Series(y, dtype=float)
        .rolling(k, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def aggregate_sigma_from_df(df: pd.DataFrame) -> float:
    """
    從單一世代的 DataFrame 取出 sigma 們並彙整成單一標量。
    支援：sigma、sigma1/sigma2/...；忽略 NaN 與 <=0（避免 log 問題）。
    """
    cols = [c for c in df.columns if re.match(SIGMA_PATTERN, c, flags=re.IGNORECASE)]
    if not cols and "sigma" in df.columns:
        cols = ["sigma"]
    if not cols:
        return np.nan

    A = pd.to_numeric(df[cols].values.reshape(-1), errors="coerce")
    A = A[np.isfinite(A) & (A > 0)]
    if A.size == 0:
        return np.nan

    agg = SIGMA_AGG.lower()
    if agg == "geomean":
        return float(np.exp(np.mean(np.log(A))))
    if agg == "rms":
        return float(np.sqrt(np.mean(A**2)))
    if agg == "mean":
        return float(np.mean(A))
    if agg == "max":
        return float(np.max(A))
    return float(np.exp(np.mean(np.log(A))))  # default geomean


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


def safe_std_err(Y: np.ndarray):
    """Compute mean, std, stderr across runs without ddof warnings (ignore NaN)."""
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


def remove_grid_from_current_fig():
    """確保存檔前不會有格線（若中途有程式呼叫 grid(True) 也會被關掉）。"""
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.grid(False)
    return fig


# ---------- core ----------
def load_one_run(run_dir: str, pattern: str = "fitness_gen*_max*.csv"):
    """回傳單一 run 的序列：x=#evals(累積, unique 設計)，以及 per-gen 統計量與 sigma。"""
    files = sorted(glob.glob(os.path.join(run_dir, pattern)), key=parse_gen)
    if not files:
        raise FileNotFoundError(run_dir)

    first_gen = min(
        parse_gen(f) for f in files if re.search(r"gen(\d+)", os.path.basename(f))
    )
    seen_keys = set()
    cum = 0
    cum_evals = []
    max_list, best_list, mean_list, std_list, sigma_list = [], [], [], [], []
    best_so_far = -np.inf

    for f in files:
        g = parse_gen(f)
        df = pd.read_csv(f)

        # fitness / role
        if "fitness" not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                continue
            df["fitness"] = df[num_cols[0]]
        if "role" not in df.columns:
            df["role"] = "child"

        # 評估集合：首代 parent_old+child，其後 child
        cand = (
            df[df["role"].isin(["parent_old", "child"])]
            if g == first_gen
            else df[df["role"] == "child"]
        )

        # unique 設計（同代去重、跨代只算新）
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

        # 族群統計
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

        # σ：從 sigma1/2/3… 或 sigma 彙整
        sigma_list.append(aggregate_sigma_from_df(df))

    return {
        "x": np.array(cum_evals, float),
        "max": np.array(max_list, float),
        "best": np.array(best_list, float),
        "mean": np.array(mean_list, float),
        "std": np.array(std_list, float),
        "sigma": np.array(sigma_list, float),
    }


def main():
    # 讀取 runs
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

    # 共同 x 網格（所有 run 的 union）
    all_x = np.array(sorted(set(np.concatenate([s["x"] for s in series]))), float)

    # ===== (A) Method robustness: Mean ± Std-Error（共同區段） =====
    Y = np.vstack([step_align(s["x"], s[Y_KEY], all_x) for s in series])
    mean_curve, std_curve, stderr, n_eff = safe_std_err(Y)

    # 共同區段的起訖索引
    full_mask = n_eff == len(series)
    if np.any(full_mask):
        full_idx = np.where(full_mask)[0]
        full_start = int(full_idx[0])
        full_end = int(full_idx[-1])
    else:
        full_start, full_end = 0, len(all_x) - 1

    # 主圖
    plt.figure()  # 使用 rcParams.figure.figsize
    # 各 run（虛線 + 步階）
    # for nm, s in zip(names, series):
    #     y = step_align(s["x"], s[Y_KEY], all_x)
    #     mask = all_x <= s["x"][-1]
    #     plt.step(
    #         all_x[mask],
    #         moving_avg(y[mask], SMOOTH_K),
    #         where="post",
    #         linestyle="--",
    #         alpha=0.55,
    #         label=nm,
    #     )
    # 前段 partial（若有）
    # if full_start > 0:
    #     plt.step(
    #         all_x[:full_start],
    #         moving_avg(mean_curve[:full_start], SMOOTH_K),
    #         where="post",
    #         color="crimson",
    #         lw=2.0,
    #         linestyle=":",
    #         label="Mean (partial)",
    #     )
    # 共同區段：實線 + 標準誤陰影
    plt.step(
        all_x[full_start : full_end + 1],
        moving_avg(mean_curve[full_start : full_end + 1], SMOOTH_K),
        where="post",
        color="crimson",
        lw=2.2,
        label=f"Mean {Y_KEY.capitalize()}",
    )
    lo = mean_curve[full_start : full_end + 1] - stderr[full_start : full_end + 1]
    hi = mean_curve[full_start : full_end + 1] + stderr[full_start : full_end + 1]
    plt.fill_between(
        all_x[full_start : full_end + 1],
        lo,
        hi,
        step="post",
        color="crimson",
        alpha=0.18,
        label="Std Error",
    )
    # 尾段 partial（若有）
    # if full_end < len(all_x) - 1:
    #     plt.step(
    #         all_x[full_end:],
    #         moving_avg(mean_curve[full_end:], SMOOTH_K),
    #         where="post",
    #         color="crimson",
    #         lw=2.0,
    #         linestyle=":",
    #         label="Mean (partial)",
    #     )

    #plt.title(f"Best vs #evaluations ({len(names)} runs)")
    plt.xlabel("Number of evaluations")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    if SHOW_GRID:
        plt.grid(True, linestyle="--", alpha=0.4)
    fig = remove_grid_from_current_fig()
    out_png = os.path.join(BASE, f"online_{Y_KEY}_3runs_mean_stderr_band.png")
    out_pdf = os.path.join(BASE, f"online_{Y_KEY}_3runs_mean_stderr_band.pdf")
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Zoom（共同區段前段）
    if EXPORT_ZOOM and np.any(full_mask):
        full_x = all_x[full_start : full_end + 1]
        x_frac_idx = max(0, int(np.floor(ZOOM_FRAC * len(full_x))) - 1)
        x_eval_idx = int(np.searchsorted(full_x, ZOOM_MIN_EVALS, side="right") - 1)
        z_end = min(max(x_frac_idx, x_eval_idx), len(full_x) - 1)
        z = np.arange(full_start, full_start + z_end + 1, dtype=int)

        plt.figure()
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
        #plt.title(f"{Y_KEY.capitalize()} vs #evaluations (zoomed)")
        plt.xlabel("Number of evaluations")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        if SHOW_GRID:
            plt.grid(True, linestyle="--", alpha=0.4)
        fig = remove_grid_from_current_fig()
        out_png_zoom = os.path.join(
            BASE, f"online_{Y_KEY}_3runs_mean_stderr_band_zoom.png"
        )
        out_pdf_zoom = os.path.join(
            BASE, f"online_{Y_KEY}_3runs_mean_stderr_band_zoom.pdf"
        )
        fig.savefig(out_png_zoom, dpi=SAVE_DPI, bbox_inches="tight")
        fig.savefig(out_pdf_zoom, bbox_inches="tight")
        plt.close(fig)

    # Summary CSV：主檔（共同區段）
    mask_common = n_eff == len(names)
    df_common = pd.DataFrame(
        {
            "evals": all_x[mask_common],
            "mean": mean_curve[mask_common],
            "std_err": stderr[mask_common],
            "n_runs": n_eff[mask_common].astype(int),
        }
    )
    for nm, s in zip(names, series):
        df_common[f"{nm}_{Y_KEY}"] = step_align(s["x"], s[Y_KEY], all_x)[mask_common]
    df_common.to_csv(os.path.join(BASE, f"online_{Y_KEY}_summary.csv"), index=False)

    # Summary CSV：完整（含 partial，方便除錯）
    df_full = pd.DataFrame(
        {
            "evals": all_x,
            "mean": mean_curve,
            "std_err": stderr,
            "n_runs": n_eff.astype(int),
        }
    )
    for nm, s in zip(names, series):
        df_full[f"{nm}_{Y_KEY}"] = step_align(s["x"], s[Y_KEY], all_x)
    df_full.to_csv(os.path.join(BASE, f"online_{Y_KEY}_summary_full.csv"), index=False)

    # ===== (B) Diversity：每 run 的 mean ± std（1×N） =====
    n = len(series)
    fig, axes = plt.subplots(1, n, sharey=True)
    if n == 1:
        axes = [axes]
    # y-range 用全體共同範圍
    ymins, ymaxs = [], []
    for s in series:
        ymins.append(np.nanmin(s["mean"] - s["std"]))
        ymaxs.append(np.nanmax(s["mean"] + s["std"]))
    y_min, y_max = np.nanmin(ymins), np.nanmax(ymaxs)
    for ax, nm, s in zip(axes, names, series):
        x, m, sd = s["x"], s["mean"], s["std"]
        ax.fill_between(x, m - sd, m + sd, step="post", alpha=0.18)
        ax.step(x, moving_avg(m, SMOOTH_K), where="post", lw=2.0, label="mean")
        ax.set_title(nm)
        ax.set_xlabel("#evals (unique, cum)")
        ax.set_ylim(y_min - 0.02 * (y_max - y_min), y_max + 0.02 * (y_max - y_min))
        if SHOW_GRID:
            ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Fitness")
    fig.suptitle("Population diversity per run: mean ± std", fontsize=13)
    remove_grid_from_current_fig()
    fig.savefig(
        os.path.join(BASE, "diversity_mean_std_runs_smallmultiples.png"),
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(BASE, "diversity_mean_std_runs_smallmultiples.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # Diversity summary（對齊到 all_x，方便後續比較）
    dfB = pd.DataFrame({"evals": all_x})
    for nm, s in zip(names, series):
        dfB[f"{nm}_mean"] = step_align(s["x"], s["mean"], all_x)
        dfB[f"{nm}_std"] = step_align(s["x"], s["std"], all_x)
    dfB.to_csv(os.path.join(BASE, "diversity_mean_std_runs_summary.csv"), index=False)

    # ===== (C) Sigma：每 run 小圖 + Mean σ ± Std-Error =====
    has_sigma = any(np.isfinite(s["sigma"]).any() for s in series)
    if has_sigma:
        # 每 run 小圖
        fig, axes = plt.subplots(1, n, sharey=True)
        if n == 1:
            axes = [axes]
        sig_ymin = np.nanmin([np.nanmin(s["sigma"]) for s in series])
        sig_ymax = np.nanmax([np.nanmax(s["sigma"]) for s in series])
        for ax, nm, s in zip(axes, names, series):
            x, sig = s["x"], s["sigma"]
            ax.step(x, moving_avg(sig, SMOOTH_K), where="post", lw=2.0)
            ax.set_title(nm)
            ax.set_xlabel("#evals (unique, cum)")
            ax.set_ylim(
                sig_ymin - 0.02 * (sig_ymax - sig_ymin),
                sig_ymax + 0.02 * (sig_ymax - sig_ymin),
            )
            if SHOW_GRID:
                ax.grid(True, linestyle="--", alpha=0.4)
        axes[0].set_ylabel("Step-size σ")
        fig.suptitle("Step-size σ per run", fontsize=14)
        remove_grid_from_current_fig()
        fig.savefig(
            os.path.join(BASE, "sigma_vs_evals_runs.png"),
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )
        fig.savefig(os.path.join(BASE, "sigma_vs_evals_runs.pdf"), bbox_inches="tight")
        plt.close(fig)

        # Mean σ ± SE（共同區段）
        all_sig = np.vstack([step_align(s["x"], s["sigma"], all_x) for s in series])
        mean_sig, std_sig, se_sig, n_sig = safe_std_err(all_sig)
        full_mask_sig = n_sig == len(series)
        if np.any(full_mask_sig):
            idx = np.where(full_mask_sig)[0]
            s_start, s_end = int(idx[0]), int(idx[-1])
        else:
            s_start, s_end = 0, len(all_x) - 1

        plt.figure()
        # for nm, s in zip(names, series):
        #     y = step_align(s["x"], s["sigma"], all_x)
        #     mask = all_x <= s["x"][-1]
        #     plt.step(
        #         all_x[mask],
        #         moving_avg(y[mask], SMOOTH_K),
        #         where="post",
        #         linestyle="--",
        #         alpha=0.55,
        #         label=nm,
        #     )
        # if s_start > 0:
        #     plt.step(
        #         all_x[:s_start],
        #         moving_avg(mean_sig[:s_start], SMOOTH_K),
        #         where="post",
        #         color="purple",
        #         lw=2.0,
        #         linestyle=":",
        #         label="Mean σ (partial)",
        #     )
        plt.step(
            all_x[s_start : s_end + 1],
            moving_avg(mean_sig[s_start : s_end + 1], SMOOTH_K),
            where="post",
            color="purple",
            lw=2.2,
            label="Mean σ",
        )
        plt.fill_between(
            all_x[s_start : s_end + 1],
            mean_sig[s_start : s_end + 1] - se_sig[s_start : s_end + 1],
            mean_sig[s_start : s_end + 1] + se_sig[s_start : s_end + 1],
            step="post",
            color="purple",
            alpha=0.18,
            label="Std Error",
        )
        # if s_end < len(all_x) - 1:
        #     plt.step(
        #         all_x[s_end:],
        #         moving_avg(mean_sig[s_end:], SMOOTH_K),
        #         where="post",
        #         color="purple",
        #         lw=2.0,
        #         linestyle=":",
        #         label="Mean σ (partial)",
        #     )
        #plt.title("Step-size σ vs #evaluations (Mean ± Std-Error)")
        plt.xlabel("Number of evaluations")
        plt.ylabel("Step-size σ")
        plt.legend(loc="lower right")
        if SHOW_GRID:
            plt.grid(True, linestyle="--", alpha=0.4)
        fig = remove_grid_from_current_fig()
        fig.savefig(
            os.path.join(BASE, "sigma_mean_stderr_band.png"),
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(BASE, "sigma_mean_stderr_band.pdf"), bbox_inches="tight"
        )
        plt.close(fig)

        # σ summary
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
        dfC.to_csv(os.path.join(BASE, "sigma_vs_evals_summary_full.csv"), index=False)

        # σ 主檔（共同區段）
        mask_sig_common = n_sig == len(names)
        dfC_common = pd.DataFrame(
            {
                "evals": all_x[mask_sig_common],
                "mean_sigma": mean_sig[mask_sig_common],
                "se_sigma": se_sig[mask_sig_common],
                "n_runs": n_sig[mask_sig_common].astype(int),
            }
        )
        for nm, s in zip(names, series):
            dfC_common[f"{nm}_sigma"] = step_align(s["x"], s["sigma"], all_x)[
                mask_sig_common
            ]
        dfC_common.to_csv(os.path.join(BASE, "sigma_vs_evals_summary.csv"), index=False)


if __name__ == "__main__":
    main()
