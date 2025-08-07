import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks, savgol_filter
import os
import sys

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
# excel_path = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\MOO_knee_[0.76,0.9,59]\SIM_561\EXP_SIM.xlsx"
# output_folder = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\MOO_knee_[0.76,0.9,59]\SIM_561\[0.76, 0.9, 59]_N1.2536_F2_61_53_sub0._light10"
excel_path = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\[0.76,0.9,59]_Compensate\SIM_561\EXP_SIM.xlsx"
output_folder = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\[0.76,0.9,59]_Compensate\SIM_561\[0.76, 0.9, 59]_N1.2536_F2_61_53_sub0._light10"
os.makedirs(output_folder, exist_ok=True)


exp_columns = ["EXP1-1"]  # , "EXP2-1"
# exp_colors = ["#8c1aff", "#1f77b4", "#003366"]

# sim_colors = ["tab:green", "tab:orange"]
sim_colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 最多10種顏色

all_peaks = []
xls = pd.ExcelFile(excel_path)

for sheet_name in xls.sheet_names:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # exp_columns = [col for col in df.columns if col.startswith("EXP")]
        exp_colors = plt.cm.tab20(np.linspace(0, 1, len(exp_columns)))

        column_sim = [col for col in df.columns if col.startswith("SIM_")]
        column_sim = [
            "SIM_[0.76, 0.9, 59]_N1.2536_F2_61_53_sub0._light10",
            # "SIM_[0.76, 0.69, 50.21]_N1.2536_PGI02_F2_54_69_sub0.6",
            # "SIM_[0.76, 0.9, 59]_N1.3_F0",
        ]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # --- 模擬資料 (改成散點圖) ---
        for i, sim_col in enumerate(column_sim):
            if sim_col not in df.columns:
                continue

            y_sim = df[sim_col].dropna().to_numpy()
            x_sim = df["SIM_angle"].dropna().to_numpy()

            total_sim_energy = y_sim.sum()
            if total_sim_energy > 0:
                y_sim = y_sim / total_sim_energy * 100
            else:
                continue
            y_sim_smooth = savgol_filter(y_sim, window_length=5, polyorder=3)
            peaks_sim, _ = find_peaks(y_sim_smooth)

            # === 改用散點圖顯示 SIM，顏色為綠色 ===
            # ax2.scatter(x_sim, y_sim, label=sim_col, color="tab:green", s=20)
            ax2.bar(x_sim, y_sim, width=0.8, label=sim_col, color="tab:green")

            for p in peaks_sim:
                all_peaks.append(
                    {
                        "Sheet": sheet_name,
                        "Source": sim_col,
                        "Angle": x_sim[p],
                        "Peak_Value": y_sim_smooth[p],
                    }
                )

        # --- 實驗資料 ---
        for idx, col in enumerate(exp_columns):
            if col not in df.columns:
                continue

            y_exp = df[col].dropna().to_numpy()
            x_exp = df["EXP_angle"].dropna().to_numpy()

            total_energy = y_exp.sum()
            if total_energy > 0:
                y_exp = y_exp / total_energy * 100
            else:
                continue
            y_exp_smooth = savgol_filter(y_exp, window_length=8, polyorder=3)
            peaks_exp, _ = find_peaks(y_exp_smooth)
            ax1.plot(x_exp, y_exp, label=f"{col}", color=exp_colors[idx])  # , alpha=0.3
            ax1.scatter(x_exp, y_exp, label=f"{col}", color=exp_colors[idx], s=10)
            # ax1.plot(x_exp, y_exp_smooth, label=f"{col}", color=exp_colors[idx])
            ax1.scatter(
                x_exp[peaks_exp],
                y_exp_smooth[peaks_exp],
                s=40,
                color=exp_colors[idx],
                linewidths=0.5,
            )

            for p in peaks_exp:
                all_peaks.append(
                    {
                        "Sheet": sheet_name,
                        "Source": col,
                        "Angle": x_exp[p],
                        "Peak_Value": y_exp_smooth[p],
                    }
                )

        # === 標籤與格式 ===
        ax1.set_xlabel("Angle (°)")
        ax1.set_xticks(np.arange(0, 181, 10))
        ax1.set_ylabel("Measured Intensity (%)", color="tab:blue")
        ax2.set_ylabel("Simulated Intensity (%)", color="green")

        ax1.tick_params(axis="y", colors="tab:blue")
        ax2.tick_params(axis="y", colors="green")
        ax1.spines["left"].set_linewidth(1.2)
        ax2.spines["right"].set_linewidth(1.2)
        ax1.yaxis.label.set_color("tab:blue")
        ax2.yaxis.label.set_color("green")

        # 圖例與標題
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        # === 圖例放在底部，多欄排版 ===
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            fontsize=9,
        )
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        plt.title(f"Elevation {sheet_name}° - EXP vs SIM", fontsize=12, weight="bold")
        # plt.tight_layout(pad=0.8)

        # 儲存圖
        save_path = os.path.join(
            output_folder, f"LightDistribution_{sheet_name}deg.png"
        )
        # plt.savefig(save_path)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {save_path}")

    except Exception as e:
        print(f"❌ Error in sheet {sheet_name}: {e}")

# === 匯出波峰資料 ===
peak_df = pd.DataFrame(all_peaks)
peak_output = os.path.join(output_folder, "All_Peaks_Export.xlsx")
peak_df.to_excel(peak_output, index=False)
print(f"✅ All peaks saved to: {peak_output}")
