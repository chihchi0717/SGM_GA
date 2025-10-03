# Predicted vs Actual (journal style)
# - 讀取 Excel 前三個工作表（s2/s3/a3）
# - 不顯示視窗，只儲存 PNG
# - 軸標籤：Actual S2 (S2^a) / Predicted S2 (S2^p)；可用開關關閉符號或單位
# - 外框關閉（只留座標軸）

import os, re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== 路徑 =====
excel_path = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202509\predicted_actual.xlsx"
out_dir = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202509\predicted_actual_plots_v8"
os.makedirs(out_dir, exist_ok=True)

# ===== 顯示開關 =====
SHOW_SYMBOLS = False  # False: 不顯示 (S2^a)、(S2^p)
WITH_UNITS = True  # False: 不加單位
UNIT_BRACKET = False  # True: 單位用 [mm]；False: 用 (mm)

# ===== 期刊風格 =====
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
        "axes.grid": False,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 0.8,
    }
)

# ===== 欄位自動偵測 =====
ACTUAL_TOKENS = [
    "實際",
    "actual",
    "measured",
    "measure",
    "reference",
    "ref",
    "gt",
    "groundtruth",
    "ground",
    "true",
    "target",
    "label",
    "real",
    "obs",
    "observed",
    " r",
    "_r",
    "(r)",
    " a",
    "_a",
    "(a)",
]
PRED_TOKENS = [
    "預測",
    "predict",
    "pred",
    "predicted",
    "estimate",
    "estimated",
    "est",
    "hat",
    "yhat",
    "model",
    "sim",
    "simulation",
    " p",
    "_p",
    "(p)",
]
EXCLUDE_TOKENS = ["design"]


def _norm(s):
    return re.sub(r"\s+", "", str(s).lower())


def _has_token(name, toks):
    return any(t in _norm(name) for t in toks)


def _excluded(name):
    return any(t in _norm(name) for t in EXCLUDE_TOKENS)


def _param_from_sheet_name(name, fallback):
    n = _norm(name)
    if "s2" in n:
        return "s2"
    if "s3" in n:
        return "s3"
    if "a3" in n:
        return "a3"
    return fallback


def _pick_columns(df, tag):
    cols = list(df.columns)
    param_filtered = [c for c in cols if (tag in str(c).lower()) and not _excluded(c)]
    search_space = (
        param_filtered if param_filtered else [c for c in cols if not _excluded(c)]
    )
    actual_cands = [c for c in search_space if _has_token(c, ACTUAL_TOKENS)]
    pred_cands = [c for c in search_space if _has_token(c, PRED_TOKENS)]
    if not actual_cands or not pred_cands:
        for c in cols:
            if _excluded(c):
                continue
            if c not in search_space:
                if _has_token(c, ACTUAL_TOKENS):
                    actual_cands.append(c)
                if _has_token(c, PRED_TOKENS):
                    pred_cands.append(c)
    if not actual_cands or not pred_cands:
        num_cols = [
            c
            for c in cols
            if (not _excluded(c)) and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(num_cols) >= 2:
            return num_cols[0], num_cols[1]
        raise ValueError("No suitable numeric columns found.")
    return actual_cands[0], pred_cands[0]


# ===== 軸標籤產生器：文字 + (可選符號) + (可選單位) =====
_sym_map = {
    "s2a": r"$S_{2}^{a}$",
    "s2p": r"$S_{2}^{p}$",
    "s3a": r"$S_{3}^{a}$",
    "s3p": r"$S_{3}^{p}$",
    "a3a": r"$A_{3}^{a}$",
    "a3p": r"$A_{3}^{p}$",
}
_unit_map = {"s2": r"$\mathrm{mm}$", "s3": r"$\mathrm{mm}$", "a3": r"$^{\circ}$"}


def _label(
    base_text,
    symbol_tex=None,
    unit_tex=None,
    show_symbols=True,
    with_units=True,
    unit_bracket=False,
):
    s = base_text
    if show_symbols and symbol_tex:
        s += f" ({symbol_tex})"
    if with_units and unit_tex:
        s += f" [{'{}'}]".format(unit_tex) if unit_bracket else f" ({unit_tex})"
    return s


def axis_labels(
    tag, show_symbols=SHOW_SYMBOLS, with_units=WITH_UNITS, unit_bracket=UNIT_BRACKET
):
    name = {"s2": "S2", "s3": "S3", "a3": "A3"}.get(tag, "X")
    x = _label(
        f"Actual {name}",
        _sym_map.get(f"{tag}a"),
        _unit_map.get(tag),
        show_symbols,
        with_units,
        unit_bracket,
    )
    y = _label(
        f"Predicted {name}",
        _sym_map.get(f"{tag}p"),
        _unit_map.get(tag),
        show_symbols,
        with_units,
        unit_bracket,
    )
    return x, y


# ===== 小工具：只保留座標軸（left/bottom），隱藏外框（top/right）=====
def show_only_axes(ax, keep=("left", "bottom")):
    for s in ("left", "bottom", "right", "top"):
        ax.spines[s].set_visible(s in keep)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.yaxis.set_ticks_position("left" if "left" in keep else "none")
    ax.xaxis.set_ticks_position("bottom" if "bottom" in keep else "none")


# ===== 主程式 =====
xls = pd.ExcelFile(excel_path)
sheet_names = xls.sheet_names[:3]  # 預期 ["s2","s3","a3"]
logs = []

for i, sh in enumerate(sheet_names):
    df = pd.read_excel(xls, sheet_name=sh).dropna(how="all").dropna(axis=1, how="all")
    tag = _param_from_sheet_name(
        sh, fallback=["s2", "s3", "a3"][i] if i < 3 else f"sheet{i+1}"
    )
    act_col, pred_col = _pick_columns(df, tag)

    y_true = pd.to_numeric(df[act_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")
    mask = ~(y_true.isna() | y_pred.isna())
    y_true, y_pred = y_true[mask], y_pred[mask]

    # ===== 樣式與顏色 =====
    LINESTYLE = "--"  # 1:1 線：虛線
    LINEWIDTH = 1.6
    MARKER = "o"  # 圓點
    MARKER_S = 30  # 點大小 (scatter 的 s)
    HOLLOW_POINTS = False  # True: 中空；False: 實心
    REF_LINE_COLOR = "#BDBDBD"  # 淺灰
    PREMIUM_BLUE = "#1f4f82"  # 高級藍

    fig, ax = plt.subplots()

    # ---- 只要座標軸、不要外框 ----
    show_only_axes(ax, keep=("left", "bottom"))
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="inout", length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", direction="in",    length=3, width=1.0)
    
    # 1:1 參考線（淺灰虛線，底層）
    xy_min = float(min(y_true.min(), y_pred.min()))
    xy_max = float(max(y_true.max(), y_pred.max()))
    if xy_min == xy_max:
        xy_min -= 1.0
        xy_max += 1.0
    ax.plot(
        [xy_min, xy_max],
        [xy_min, xy_max],
        linestyle=LINESTYLE,
        linewidth=LINEWIDTH,
        color=REF_LINE_COLOR,
        label=r"Reference line ($y=x$)",
        zorder=2,
    )

    # 資料點（高級藍；中空/實心可切換；在線上層）
    ax.scatter(
        y_true,
        y_pred,
        marker=MARKER,
        s=MARKER_S,
        facecolors="none" if HOLLOW_POINTS else PREMIUM_BLUE,
        edgecolors=PREMIUM_BLUE,
        linewidths=1.2,
        zorder=3,
    )

    # 軸標籤
    xlab, ylab = axis_labels(tag)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    out_png = os.path.join(out_dir, f"{tag}_pred_vs_actual.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    mae = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else float("nan")
    logs.append(
        f"[{sh}] tag={tag} actual='{act_col}' pred='{pred_col}' n={len(y_true)} MAE={mae:.6g} -> {out_png}"
    )

with open(os.path.join(out_dir, "log_v8.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(logs))
print("\n".join(logs))
