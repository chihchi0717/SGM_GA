# -*- coding: utf-8 -*-
"""
Per-generation mean of geometric parameters (Design_S2/S3/A3) vs #evaluations
- μ+λ；首代計 μ+λ，其後只計 child
- 曾出現過的設計(Design_S2,S3,A3)不重算；x 軸 = 累積 unique #evals
- POP_STATS_PARAM = "all": 用整個族群(父+子)算平均
                       "evaluated_only": 只用本代「新評估」且去重後的個體算平均
輸出：每個參數一張（所有 run 疊加 + Mean ± Std-Error 陰影），以及對齊後 CSV。
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ========= 設定 =========
BASE = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\runs"
RUN_GLOB = "run*"
CSV_PATTERN = "fitness_gen*_max*.csv"

PARAM_COLS = ["Design_S2", "Design_S3", "Design_A3"]
KEY_COLS = ["Design_S2", "Design_S3", "Design_A3"]  # 判斷「同一設計」的鍵
ROUND_DECIMALS = 5

POP_STATS_PARAM = "all"  # "all" or "evaluated_only"
SMOOTH_K = 1  # 顯示平滑(1=不平滑)
SAVE_DPI = 300
SHOW_GRID = False
# =======================

# === 期刊風格（和你現有圖一致） ===
mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.figsize": (7.5, 4),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": SAVE_DPI,
        "axes.grid": False,
    }
)


# ---------- 工具 ----------
def parse_gen(p):
    m = re.search(r"gen(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 10**9


def keys_from_df(df):
    arr = np.round(df[KEY_COLS].to_numpy(dtype=float), ROUND_DECIMALS)
    return list(map(tuple, arr))


def moving_avg(y, k=1):
    if k <= 1:
        return np.asarray(y, float)
    return (
        pd.Series(y, dtype=float)
        .rolling(k, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def step_align(x_src, y_src, x_grid):
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    xg = np.asarray(x_grid, float)
    idx = np.searchsorted(x_src, xg, side="right") - 1
    y = np.full_like(xg, np.nan, dtype=float)
    valid = idx >= 0
    y[valid] = y_src[idx[valid]]
    return y


def safe_std_err(Y):
    mask = ~np.isnan(Y)
    n = mask.sum(axis=0).astype(float)
    Yf = np.where(mask, Y, 0.0)
    mean = np.divide(Yf.sum(axis=0), n, out=np.zeros_like(n), where=n > 0)
    ss = np.where(mask, (Y - mean) ** 2, 0.0).sum(axis=0)
    var = np.divide(ss, np.maximum(n - 1, 1), out=np.zeros_like(ss), where=n > 1)
    std = np.sqrt(var)
    se = np.divide(std, np.sqrt(n), out=np.zeros_like(std), where=n > 0)
    return mean, std, se, n


# ---------- 讀一個 run：回傳每代的參數平均 ----------
def load_param_means_one_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, CSV_PATTERN)), key=parse_gen)
    if not files:
        raise FileNotFoundError(run_dir)

    first_gen = min(
        parse_gen(f) for f in files if re.search(r"gen(\d+)", os.path.basename(f))
    )
    seen_keys = set()
    cum = 0
    x_list = []  # 累積 unique #evals

    # 每個參數一條序列
    means = {p: [] for p in PARAM_COLS}

    for f in files:
        g = parse_gen(f)
        df = pd.read_csv(f)

        # 欄位處理
        if "fitness" not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                continue
            df["fitness"] = df[num_cols[0]]
        if "role" not in df.columns:
            df["role"] = "child"

        # 本代要「評估」的集合（只為了累積 unique #evals 用）
        cand = (
            df[df["role"].isin(["parent_old", "child"])]
            if g == first_gen
            else df[df["role"] == "child"]
        )
        cand_keys = keys_from_df(cand)
        # 同代去重，跨代只算新
        unique_ordered, seen_this = [], set()
        for k in cand_keys:
            if k not in seen_this:
                unique_ordered.append(k)
                seen_this.add(k)
        new_keys = [k for k in unique_ordered if k not in seen_keys]
        seen_keys.update(new_keys)
        cum += len(new_keys)
        x_list.append(cum)

        # 這一代用來「計算參數平均」的資料集
        if POP_STATS_PARAM == "evaluated_only":
            if len(new_keys) == 0:
                use = pd.DataFrame(columns=df.columns)
            else:
                mask = np.zeros(len(cand), dtype=bool)
                new_set, once = set(new_keys), set()
                for i, k in enumerate(cand_keys):
                    if (k in new_set) and (k not in once):
                        mask[i] = True
                        once.add(k)
                use = cand.loc[mask]
        else:
            use = df  # 整個族群（父+子）

        # 逐參數取平均
        for p in PARAM_COLS:
            if p in use.columns:
                vals = pd.to_numeric(use[p], errors="coerce").dropna().to_numpy()
                means[p].append(float(np.mean(vals)) if vals.size > 0 else np.nan)
            else:
                means[p].append(np.nan)

    return {
        "name": os.path.basename(run_dir),
        "x": np.array(x_list, float),
        "param_means": {p: np.array(v, float) for p, v in means.items()},
    }


# ---------- 主程式 ----------
def main():
    run_dirs = [d for d in glob.glob(os.path.join(BASE, RUN_GLOB)) if os.path.isdir(d)]
    series = [load_param_means_one_run(rd) for rd in sorted(run_dirs)]
    names = [s["name"] for s in series]

    # 全部 x 的聯集（right-step 對齊）
    all_x = np.array(sorted(set(np.concatenate([s["x"] for s in series]))), float)

    for p in PARAM_COLS:
        # 對齊到共同 x
        Y = np.vstack([step_align(s["x"], s["param_means"][p], all_x) for s in series])
        mean_curve, std_curve, stderr, n_eff = safe_std_err(Y)

        # 只在「所有 run 都有資料」的共同區段畫陰影與實線
        full_mask = n_eff == len(series)
        if np.any(full_mask):
            idx = np.where(full_mask)[0]
            i0, i1 = int(idx[0]), int(idx[-1])
        else:
            i0, i1 = 0, len(all_x) - 1

        # 繪圖
        plt.figure()
        # 各 run（虛線）
        # for nm, s in zip(names, series):
        #     y = step_align(s["x"], s["param_means"][p], all_x)
        #     mask = all_x <= s["x"][-1]
        #     plt.step(
        #         all_x[mask],
        #         moving_avg(y[mask], SMOOTH_K),
        #         where="post",
        #         linestyle="--",
        #         alpha=0.6,
        #         label=nm,
        #     )
        # 前段 partial
        # if i0 > 0:
        #     plt.step(
        #         all_x[:i0],
        #         moving_avg(mean_curve[:i0], SMOOTH_K),
        #         where="post",
        #         color="crimson",
        #         lw=2.0,
        #         linestyle=":",
        #         label="Mean (partial)",
        #     )
        # 共同區段：平均 + SE 陰影
        plt.step(
            all_x[i0 : i1 + 1],
            moving_avg(mean_curve[i0 : i1 + 1], SMOOTH_K),
            where="post",
            color="crimson",
            lw=2.2,
            label=f"Mean {p}",
        )
        plt.fill_between(
            all_x[i0 : i1 + 1],
            mean_curve[i0 : i1 + 1] - stderr[i0 : i1 + 1],
            mean_curve[i0 : i1 + 1] + stderr[i0 : i1 + 1],
            step="post",
            color="crimson",
            alpha=0.18,
            label="Std Error",
        )
        # 尾段 partial
        # if i1 < len(all_x) - 1:
        #     plt.step(
        #         all_x[i1:],
        #         moving_avg(mean_curve[i1:], SMOOTH_K),
        #         where="post",
        #         color="crimson",
        #         lw=2.0,
        #         linestyle=":",
        #         label="Mean (partial)",
        #     )

        title_tail = " (all pop)" if POP_STATS_PARAM == "all" else " (evaluated-only)"
        # plt.title(f"{p} mean vs #evaluations{title_tail}")
        plt.xlabel("Number of evaluations")

        # 根據參數設定 Y 軸單位
        unit_map = {
            "Design_S2": "(mm)",
            "Design_S3": "(mm)",
            "Design_A3": "(deg)",
        }
        y_label = f"{p} {unit_map.get(p, '')}".strip()
        plt.ylabel(y_label)

        if SHOW_GRID:
            plt.grid(True, linestyle="--", alpha=0.4)
        # 小圖例
        plt.legend(
            loc="best",
            fontsize=9,
            frameon=True,
            handlelength=1.2,
            handletextpad=0.5,
            borderpad=0.25,
            labelspacing=0.25,
            columnspacing=0.8,
            ncol=2,
        )
        # 存檔
        stem = f"param_mean_{p}_{POP_STATS_PARAM}"
        plt.savefig(
            os.path.join(BASE, f"{stem}.png"), dpi=SAVE_DPI, bbox_inches="tight"
        )
        plt.savefig(os.path.join(BASE, f"{stem}.pdf"), bbox_inches="tight")
        plt.close()

        # 對齊後 CSV（含共同/partial 的 n_runs）
        df = pd.DataFrame(
            {
                "evals": all_x,
                "mean": mean_curve,
                "std_err": stderr,
                "n_runs": n_eff.astype(int),
            }
        )
        for nm, s in zip(names, series):
            df[f"{nm}_{p}"] = step_align(s["x"], s["param_means"][p], all_x)
        df.to_csv(os.path.join(BASE, f"{stem}_summary_full.csv"), index=False)

        # 只輸出共同區段主檔
        mask_common = n_eff == len(names)
        df_common = df.loc[mask_common].copy()
        df_common.to_csv(os.path.join(BASE, f"{stem}_summary.csv"), index=False)


if __name__ == "__main__":
    main()
