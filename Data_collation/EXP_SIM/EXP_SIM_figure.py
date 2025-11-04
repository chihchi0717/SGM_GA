import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks, savgol_filter
import os
import sys
import re

sys.stdout.reconfigure(encoding="utf-8")
# === 畫圖風格設定 ===
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
        "savefig.dpi": 300,
        "figure.figsize": (7.5, 4),
    }
)

# === 設定 ===
base_folder = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\Optimize_Regression\[0.82, 0.66, 63.58]"
excel_path = os.path.join(base_folder, "EXP_SIM_design.xlsx")
output_folder = os.path.join(base_folder, "EXP_SIM_design")
os.makedirs(output_folder, exist_ok=True)

# --- 實驗數據標準化設定 (開關) ---
# True: 以總能量最高的實驗為基準進行標準化 (用於比較相對強度)
# False: 各實驗獨立計算百分比 (用於觀察各自的分佈形狀)
normalize_by_max_exp = False

# 您可以在此處指定要繪製的所有實驗數據欄位
# exp_columns = ["EXP1-1", "EXP2-1", "EXP3-1", "EXP1S-1", "EXP2S-1", "EXP3S-1"]

all_peaks = []
xls = pd.ExcelFile(excel_path)


# 自定義排序函式，用於將實驗欄位排序 (例如 EXP1 -> EXP2 -> EXP1S)
def custom_sort_key_for_exp_cols(col_name):
    match = re.search(r"EXP(\d+)(S?)-(\d+)", col_name)
    if not match:
        return (2, 0, 0)  # 將不匹配的項目排到最後

    num1 = int(match.group(1))
    s_part = match.group(2)
    num2 = int(match.group(3))

    # 先按有無 'S' 分組 (無 S 的組為 0，有 S 的組為 1)，再按數字排序
    group = 1 if s_part == "S" else 0
    return (group, num1, num2)


for sheet_name in xls.sheet_names:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # === 修改：自動偵測並排序所有實驗欄位 ===
        exp_columns = sorted(
            [
                col
                for col in df.columns
                if col.startswith("EXP") and not col.startswith("EXP_angle")
            ],
            key=custom_sort_key_for_exp_cols,
        )
        # === 選取模擬資料欄位，並依類型分開 ===
        all_sim_cols = [
            col
            for col in df.columns
            if col.startswith("SIM_") and not col.startswith("SIM_angle")
        ]
        design_cols = sorted([col for col in all_sim_cols if "design" in col])
        shrinkage_cols = sorted([col for col in all_sim_cols if "shrinkage" in col])

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # --- 繪製 Design 模擬 (黑色實線) ---
        if design_cols:
            for sim_col in design_cols:
                y_sim = df[sim_col].dropna().to_numpy()
                x_sim = df["SIM_angle"].dropna().to_numpy()
                total_sim_energy = y_sim.sum()
                if total_sim_energy > 0:
                    y_sim = y_sim / total_sim_energy * 100
                else:
                    continue
                ax2.bar(
                    x_sim,
                    y_sim,
                    label="SIM_design",
                    color="#343333",
                    linestyle="-",
                    linewidth=1.5,
                )

        # --- 繪製 Shrinkage 模擬 (紅色實線) ---
        if shrinkage_cols:
            for sim_col in shrinkage_cols:
                y_sim = df[sim_col].dropna().to_numpy()
                x_sim = df["SIM_angle"].dropna().to_numpy()
                total_sim_energy = y_sim.sum()
                if total_sim_energy > 0:
                    y_sim = y_sim / total_sim_energy * 100
                else:
                    continue
                ax2.bar(
                    x_sim,
                    y_sim,
                    label="SIM_shrinkage",
                    color="#D53500",  # "#D53500",
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.7,
                )

        # --- 實驗資料 (折線圖+散點) ---
        if exp_columns:
            # 修改：定義多樣的線條、標記與顏色，以便循環使用
            linestyles = ["-", "--", ":", "-."]
            markers = ["o", "s", "^", "D", "v", "p", "*", "+"]
            exp_colormap = plt.cm.Blues  # 使用色彩區分度更高的色板
            exp_colors = exp_colormap(np.linspace(0.3, 1, len(exp_columns)))

            # 根據開關選擇標準化方法
            if normalize_by_max_exp:
                # --- 方法一：以能量最高的實驗為基準進行標準化 ---
                exp_totals = {}
                for col in exp_columns:
                    if col in df.columns:
                        exp_totals[col] = df[col].dropna().sum()

                max_exp_total = 0
                if exp_totals:
                    max_exp_total = max(exp_totals.values())

                if max_exp_total > 0:
                    for idx, col in enumerate(exp_columns):
                        if col not in df.columns:
                            continue
                        y_exp = df[col].dropna().to_numpy()
                        x_exp = df["EXP_angle"].dropna().to_numpy()
                        y_exp_percent = y_exp / max_exp_total * 100

                        # 修改：循環選取顏色、線條與標記
                        current_color = exp_colors[idx]
                        current_linestyle = linestyles[idx % len(linestyles)]
                        current_marker = markers[idx % len(markers)]

                        # 取出 "EXP數字" 部分，重新命名
                        match = re.match(r"EXP(\d+)", col)
                        if match:
                            exp_number = int(match.group(1))
                            exp_label = f"EXP{exp_number}"   # 只保留 EXP1, EXP2, EXP3
                        else:
                            exp_label = col

                        ax1.plot(
                            x_exp,
                            y_exp_percent,
                            label=exp_label,   # 用新的簡化名稱
                            color=current_color,
                            linestyle=current_linestyle,
                            marker=current_marker,
                            markersize=4,
                            alpha=0.8,
                        )

            else:
                # --- 方法二：各實驗獨立標準化 ---
                for idx, col in enumerate(exp_columns):
                    if col not in df.columns:
                        continue
                    y_exp = df[col].dropna().to_numpy()
                    x_exp = df["EXP_angle"].dropna().to_numpy()
                    total_energy = y_exp.sum()

                    if total_energy > 0:
                        y_exp_percent = y_exp / total_energy * 100
                    else:
                        y_exp_percent = y_exp

                    # 修改：循環選取顏色、線條與標記
                    current_color = exp_colors[idx]
                    current_linestyle = linestyles[idx % len(linestyles)]
                    current_marker = markers[idx % len(markers)]

                    exp_mapping = {}
                    new_index = 1
                    for original_name in exp_columns:
                        exp_mapping[original_name] = f"EXP{new_index}"
                        new_index += 1

                    ax1.plot(
                        x_exp,
                        y_exp_percent,
                        label=exp_mapping.get(col, col),  # 用重新編號的名稱
                        color=current_color,
                        linestyle=current_linestyle,
                        marker=current_marker,
                        markersize=4,
                        alpha=0.8,
                    )

        # === 標籤與格式 ===
        ax1.set_xlabel("Angle (°)")
        ax1.set_xticks(np.arange(0, 181, 10))
        ax1.set_ylabel("Measured Intensity (%)", color="steelblue")   # 左軸藍色
        ax2.set_ylabel("Simulated Intensity (%)", color="#D53500")  # 右軸紅色

        ax1.tick_params(axis="y", colors="steelblue")
        ax2.tick_params(axis="y", colors="#D53500")
        ax1.spines["left"].set_color("steelblue")
        ax1.spines["left"].set_linewidth(1.2)
        ax2.spines["right"].set_color("#D53500")
        ax2.spines["right"].set_linewidth(1.2)
        ax1.yaxis.label.set_color("steelblue")
        ax2.yaxis.label.set_color("#D53500")

        # === 圖例與標題 ===
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # fig.legend(
        #     lines1 + lines2,
        #     labels1 + labels2,
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, -0.1),
        #     ncol=5,
        #     fontsize=9,
        # )
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        plt.title(f"Elevation {sheet_name}° - EXP vs SIM", fontsize=12, weight="bold")

        # === 儲存圖檔 ===
        save_path = os.path.join(
            output_folder, f"LightDistribution_{sheet_name}deg.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {save_path}")

    except Exception as e:
        print(f"❌ Error in sheet {sheet_name}: {e}")
