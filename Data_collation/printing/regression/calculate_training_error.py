# -*- coding: utf-8 -*-
"""
calculate_training_error.py
- 專用於計算模型在整個訓練集上的擬合誤差。
- 此腳本的流程是：
  1. 使用「全部」的資料來訓練長度與角度模型。
  2. 使用訓練好的模型，回頭去預測「全部」的資料。
  3. 計算每一筆資料的預測尺寸/角度與真實量測值之間的誤差。
  4. (新增) 同時計算尺寸 (s2, s3) 的預測收縮率與實際收縮率的誤差。
  5. 匯總所有誤差，並計算總體的 MAE 和 RMSE。
  6. (新增) 選擇性地產生並儲存視覺化診斷圖表。
- 這個腳本的目的是評估模型的「擬合能力」，而非「泛化能力」。
"""
import argparse
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

# 【新增】導入繪圖函式庫
import matplotlib.pyplot as plt
import seaborn as sns

# --- 從現有模組導入所需功能 ---
from compensation_utils import (
    FEATURES,
    TARGETS,
    average_by_structure,
)
from model_main import build_models  # 借用模型建立函式


def calculate_in_sample_error(
    df_all_data: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    在整個資料集上訓練模型，並計算其擬合誤差（訓練誤差）。

    Args:
        df_all_data: 包含所有樣本的完整 DataFrame。
        args: 從命令列解析的參數，用於建立模型。

    Returns:
        一個元組，包含 (結果DataFrame, 長度模型物件, 角度模型物件)。
    """
    print("\n[訓練集評估] 正在使用全部資料訓練模型...")

    # 1. 使用全部資料建立模型
    model_len, model_ang = build_models(args, df_all_data)
    print("模型訓練完成。")

    # 2. 在同一批資料上進行預測
    print("正在對訓練集進行預測...")
    pred_len_df = model_len.predict_df(df_all_data)
    pred_ang_df = model_ang.predict_df(df_all_data)

    # 3. 計算尺寸、角度與誤差
    results_df = df_all_data[FEATURES].copy()

    # --- 尺寸 s2 ---
    results_df["Actual_DIP_s2(mm)"] = df_all_data["DIP_s2(mm)"]
    predicted_dip_s2 = df_all_data["Design_s2(mm)"] * (1 - pred_len_df["delta_s2"])
    results_df["Predicted_DIP_s2(mm)"] = predicted_dip_s2
    results_df["Error_s2(mm)"] = predicted_dip_s2 - df_all_data["DIP_s2(mm)"]

    # --- 尺寸 s3 ---
    results_df["Actual_DIP_s3(mm)"] = df_all_data["DIP_s3(mm)"]
    predicted_dip_s3 = df_all_data["Design_s3(mm)"] * (1 - pred_len_df["delta_s3"])
    results_df["Predicted_DIP_s3(mm)"] = predicted_dip_s3
    results_df["Error_s3(mm)"] = predicted_dip_s3 - df_all_data["DIP_s3(mm)"]

    # --- 角度 a3 ---
    results_df["Actual_DIP_a3(deg)"] = df_all_data["DIP_a3(deg)"]
    results_df["Predicted_DIP_a3(deg)"] = pred_ang_df["DIP_a3(deg)"]
    results_df["Error_a3(deg)"] = (
        pred_ang_df["DIP_a3(deg)"] - df_all_data["DIP_a3(deg)"]
    )

    # === 【新增】計算尺寸收縮率 (s2, s3) 與其誤差 ===
    # 實際收縮率 (來自資料)
    results_df["Actual_delta_s2(%)"] = df_all_data["delta_s2"] * 100
    results_df["Actual_delta_s3(%)"] = df_all_data["delta_s3"] * 100

    # 預測收縮率 (來自模型)
    results_df["Predicted_delta_s2(%)"] = pred_len_df["delta_s2"] * 100
    results_df["Predicted_delta_s3(%)"] = pred_len_df["delta_s3"] * 100

    # 收縮率誤差 (Predicted - Actual)
    results_df["Error_delta_s2(%)"] = (
        pred_len_df["delta_s2"] - df_all_data["delta_s2"]
    ) * 100
    results_df["Error_delta_s3(%)"] = (
        pred_len_df["delta_s3"] - df_all_data["delta_s3"]
    ) * 100

    print("誤差計算完成。")
    return results_df, model_len, model_ang


def _generate_diagnostic_plots_for_target(
    results_df: pd.DataFrame,
    target_name: str,
    design_col: str,
    unit: str,
    prefix: str,
):
    """
    為單一目標（如 s2, a3）產生一組標準的診斷圖表。
    這是一個輔助函式，由 generate_and_save_plots 調用。
    """
    # 根據目標名稱決定欄位名稱
    if target_name.startswith("delta"):
        actual_col = f"Actual_{target_name}({unit})"
        predicted_col = f"Predicted_{target_name}({unit})"
        error_col = f"Error_{target_name}({unit})"
        title_prefix = f"Shrinkage {target_name.replace('delta_', '')}"
    else:
        actual_col = f"Actual_DIP_{target_name}({unit})"
        predicted_col = f"Predicted_DIP_{target_name}({unit})"
        error_col = f"Error_{target_name}({unit})"
        title_prefix = (
            f"Dimension {target_name}" if unit == "mm" else f"Angle {target_name}"
        )

    # --- 圖表 1: 預測值 vs. 實際值 ---
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=results_df, x=actual_col, y=predicted_col, alpha=0.7)
    min_val = min(results_df[actual_col].min(), results_df[predicted_col].min())
    max_val = max(results_df[actual_col].max(), results_df[predicted_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y=x)")
    plt.title(f"{title_prefix}: Predicted vs. Actual Values")
    plt.xlabel(f"Actual Value ({unit})")
    plt.ylabel(f"Predicted Value ({unit})")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plot_path_1 = f"{prefix}_{target_name}_predicted_vs_actual.png"
    plt.savefig(plot_path_1)
    plt.close()
    print(f"    - 已儲存圖表: {plot_path_1}")

    # --- 圖表 2: 誤差分佈圖 ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x=error_col, kde=True, bins=20)
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.title(f"Distribution of {title_prefix} Error")
    plt.xlabel(f"Error ({unit}) [Predicted - Actual]")
    plt.ylabel("Frequency")
    plot_path_2 = f"{prefix}_{target_name}_error_distribution.png"
    plt.savefig(plot_path_2)
    plt.close()
    print(f"    - 已儲存圖表: {plot_path_2}")

    # --- 圖表 3: 誤差 vs. 設計值 ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x=design_col, y=error_col, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--", lw=2)
    plt.title(f"{title_prefix} Error vs. Design Value")
    plt.xlabel(f"Design Value ({design_col})")
    plt.ylabel(f"Error ({unit})")
    plot_path_3 = f"{prefix}_{target_name}_error_vs_design.png"
    plt.savefig(plot_path_3)
    plt.close()
    print(f"    - 已儲存圖表: {plot_path_3}")


# 【修改】產生並儲存診斷圖表的函式
def generate_and_save_plots(results_df: pd.DataFrame, prefix: str = "training_error"):
    """
    根據結果 DataFrame 產生並儲存一系列診斷圖表。
    """
    print("\n正在產生診斷圖表...")
    plt.style.use("seaborn-v0_8-whitegrid")

    # 定義要繪圖的目標及其對應的設計欄位和單位
    targets_to_plot = {
        "s2": {"unit": "mm", "design_col": "Design_s2(mm)"},
        "s3": {"unit": "mm", "design_col": "Design_s3(mm)"},
        "a3": {"unit": "deg", "design_col": "Design_a3(deg)"},
        "delta_s2": {"unit": "%", "design_col": "Design_s2(mm)"},
        "delta_s3": {"unit": "%", "design_col": "Design_s3(mm)"},
    }

    # 循環為每個目標產生圖表
    for target_name, config in targets_to_plot.items():
        print(f"\n--- 正在為 '{target_name}' 產生圖表 ---")
        _generate_diagnostic_plots_for_target(
            results_df=results_df,
            target_name=target_name,
            design_col=config["design_col"],
            unit=config["unit"],
            prefix=prefix,
        )


def main():
    ap = argparse.ArgumentParser(
        description="計算模型在整個訓練集上的擬合誤差並可選擇性產生診斷圖表"
    )
    ap.add_argument(
        "--file",
        type=str,
        default="analysis_results_0.6_0.9.xlsx",
        help="輸入的 Excel 檔案路徑",
    )
    ap.add_argument("--sheet", type=str, default="Sheet1", help="Excel 中的工作表名稱")
    ap.add_argument(
        "--average", action="store_true", help="是否對相同結構的樣本進行平均"
    )
    ap.add_argument(
        "--save-results",
        type=str,
        default="training_error_results_0.6_0.9.xlsx",
        help="儲存詳細評估結果的 Excel 檔路徑",
    )
    ap.add_argument("--save-plots", action="store_true", help="是否產生並儲存診斷圖表")
    # 【新增】儲存平均資料的參數
    ap.add_argument(
        "--save-averaged-data",
        type=str,
        default=None,
        help="如果設定了 --average，則將平均後的資料儲存到指定的 Excel 檔案路徑",
    )

    # --- 模型與特徵參數 ---
    group_len = ap.add_argument_group("Length Model Parameters")
    group_len.add_argument(
        "--length-model", type=str, default="huber", choices=["ols", "huber", "rf"]
    )
    group_len.add_argument("--add-interactions", action="store_true")
    group_len.add_argument("--add-len-aa-interact", action="store_true")
    group_len.add_argument("--len-huber-alpha", type=float, default=1e-3)
    group_len.add_argument("--len-huber-eps", type=float, default=1.35)
    group_len.add_argument("--len-huber-max-iter", type=int, default=2000)
    group_len.add_argument("--scale-length", action="store_true", default=True)
    group_len.add_argument("--len-rf-n-est", type=int, default=300)
    group_len.add_argument("--len-rf-max-depth", type=int, default=None)
    group_len.add_argument("--len-rf-min-leaf", type=int, default=1)
    group_len.add_argument("--len-rf-max-features", type=float, default=1.0)
    group_len.add_argument("--len-rf-criterion", type=str, default="squared_error")
    group_len.add_argument("--len-add-ratios", action="store_true")
    group_len.add_argument("--len-add-sincos", action="store_true")

    group_ang = ap.add_argument_group("Angle Model Parameters")
    group_ang.add_argument(
        "--angle-model",
        type=str,
        default="huber",
        choices=["ols", "rf", "huber", "xgb"],
    )
    group_ang.add_argument("--add-angle-interactions", action="store_true")
    group_ang.add_argument("--add-angle-aa-interact", action="store_true")
    group_ang.add_argument("--angle-poly", type=int, default=2)
    group_ang.add_argument("--angle-ridge", type=float, default=1e-2)
    group_ang.add_argument("--rf-n-est", type=int, default=300)
    group_ang.add_argument("--rf-max-depth", type=int, default=None)
    group_ang.add_argument("--rf-min-leaf", type=int, default=1)
    group_ang.add_argument("--angle-huber-max-iter", type=int, default=2000)
    group_ang.add_argument("--angle-huber-alpha", type=float, default=1e-2)
    group_ang.add_argument("--angle-huber-eps", type=float, default=10)
    group_ang.add_argument("--scale-angle", action="store_true", default=True)
    group_ang.add_argument("--add-angle-sincos", action="store_true")
    group_ang.add_argument("--add-ratios", action="store_true")

    args = ap.parse_args()

    # --- 1. 讀取與處理資料 ---
    print(f"[資料處理] 讀取檔案: {args.file} | 工作表: {args.sheet}")
    try:
        df_raw = pd.read_excel(args.file, sheet_name=args.sheet)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{args.file}'。")
        sys.exit(1)

    required_cols = FEATURES + ["DIP_s2(mm)", "DIP_s3(mm)", "DIP_a3(deg)"]
    if not all(col in df_raw.columns for col in required_cols):
        print("錯誤: 檔案中缺少必要的欄位。")
        sys.exit(1)

    eps = 1e-9
    df_raw["delta_s2"] = (df_raw["Design_s2(mm)"] - df_raw["DIP_s2(mm)"]) / (
        df_raw["Design_s2(mm)"] + eps
    )
    df_raw["delta_s3"] = (df_raw["Design_s3(mm)"] - df_raw["DIP_s3(mm)"]) / (
        df_raw["Design_s3(mm)"] + eps
    )

    if args.average:
        print("對相同結構的樣本進行平均...")
        cols_to_average = TARGETS + ["DIP_s2(mm)", "DIP_s3(mm)"]
        cols_to_average = sorted(list(set(cols_to_average)))
        df_use = df_raw.groupby(FEATURES, as_index=False)[cols_to_average].mean()

        # 【新增】如果使用者指定了路徑，就儲存平均後的資料
        if args.save_averaged_data:
            try:
                df_use.to_excel(args.save_averaged_data, index=False)
                print(f"[已儲存] 平均後的訓練資料已儲存至 -> {args.save_averaged_data}")
            except Exception as e:
                print(f"\n[錯誤] 無法儲存平均後的資料檔案: {e}")
    else:
        df_use = df_raw.copy()

    initial_rows = len(df_use)
    cols_for_dropna = FEATURES + TARGETS + ["DIP_s2(mm)", "DIP_s3(mm)"]
    existing_cols_for_dropna = [col for col in cols_for_dropna if col in df_use.columns]
    df_use.dropna(subset=existing_cols_for_dropna, inplace=True)
    if len(df_use) < initial_rows:
        print(f"提示：已從資料中移除了 {initial_rows - len(df_use)} 個包含缺失值的行。")

    if len(df_use) == 0:
        print("錯誤：處理後沒有可用的資料行。請檢查您的輸入檔案。")
        sys.exit(1)

    # --- 2. 執行訓練集誤差計算 ---
    results_df, model_len, model_ang = calculate_in_sample_error(df_use, args)

    # --- 3. 匯總與顯示結果 ---
    print("\n\n" + "=" * 50)
    print("=== 訓練集擬合誤差總結 ===")
    print("=" * 50)

    # 尺寸與角度誤差
    mae_s2 = results_df["Error_s2(mm)"].abs().mean()
    rmse_s2 = np.sqrt((results_df["Error_s2(mm)"] ** 2).mean())
    mae_s3 = results_df["Error_s3(mm)"].abs().mean()
    rmse_s3 = np.sqrt((results_df["Error_s3(mm)"] ** 2).mean())
    mae_a3 = results_df["Error_a3(deg)"].abs().mean()
    rmse_a3 = np.sqrt((results_df["Error_a3(deg)"] ** 2).mean())

    # 【新增】收縮率誤差
    mae_delta_s2 = results_df["Error_delta_s2(%)"].abs().mean()
    rmse_delta_s2 = np.sqrt((results_df["Error_delta_s2(%)"] ** 2).mean())
    mae_delta_s3 = results_df["Error_delta_s3(%)"].abs().mean()
    rmse_delta_s3 = np.sqrt((results_df["Error_delta_s3(%)"] ** 2).mean())

    print("\n--- 整體擬合誤差 (In-Sample Error) ---")
    print(f"  s2 尺寸 MAE : {mae_s2:.6f} mm")
    print(f"  s2 尺寸 RMSE: {rmse_s2:.6f} mm")
    print(f"  s2 收縮率 MAE : {mae_delta_s2:.6f} %")
    print(f"  s2 收縮率 RMSE: {rmse_delta_s2:.6f} %")
    print("-" * 25)
    print(f"  s3 尺寸 MAE : {mae_s3:.6f} mm")
    print(f"  s3 尺寸 RMSE: {rmse_s3:.6f} mm")
    print(f"  s3 收縮率 MAE : {mae_delta_s3:.6f} %")
    print(f"  s3 收縮率 RMSE: {rmse_delta_s3:.6f} %")
    print("-" * 25)
    print(f"  a3 角度 MAE : {mae_a3:.6f} deg")
    print(f"  a3 角度 RMSE: {rmse_a3:.6f} deg")

    # --- 4. 儲存詳細結果與模型係數 ---
    if args.save_results:
        try:
            with pd.ExcelWriter(args.save_results) as writer:
                results_df.to_excel(
                    writer, index=False, sheet_name="Training_Error_Details"
                )
                print(f"\n[已儲存] 詳細的訓練集誤差結果已儲存至 -> {args.save_results}")
                if hasattr(model_len, "get_coefficients_df"):
                    len_coeffs = model_len.get_coefficients_df()
                    if len_coeffs is not None and not len_coeffs.empty:
                        len_coeffs.to_excel(writer, sheet_name="len_model_coeffs")
                        print("    - 已儲存長度模型係數")
                if hasattr(model_ang, "get_coefficients_df"):
                    ang_coeffs = model_ang.get_coefficients_df()
                    if ang_coeffs is not None and not ang_coeffs.empty:
                        ang_coeffs.to_excel(writer, sheet_name="ang_model_coeffs")
                        print("    - 已儲存角度模型係數")
        except Exception as e:
            print(f"\n[錯誤] 無法儲存結果檔案: {e}")

    # --- 5. (新增) 產生並儲存圖表 ---
    if args.save_plots:
        # 使用 Excel 檔名作為圖檔前綴
        plot_prefix = (
            args.save_results.replace(".xlsx", "")
            if args.save_results
            else "training_error"
        )
        generate_and_save_plots(results_df, prefix=plot_prefix)


if __name__ == "__main__":
    main()


# python calculate_training_error.py --average --save-averaged-data averaged_data.xlsx --add-interactions --add-len-aa-interact --scale-length --scale-angle --add-angle-interactions --save-results 20250905training_error_results_0.6_0.9.xlsx --save-plots
