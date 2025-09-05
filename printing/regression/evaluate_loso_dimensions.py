# -*- coding: utf-8 -*-
"""
evaluate_loso_dimensions.py
- 專用於執行「留一結構交叉驗證 (Leave-One-Structure-Out, LOSO)」的腳本。
- 此腳本的核心目標是：
  1. 針對每一個獨特的結構，將其作為測試集，其餘所有結構作為訓練集來訓練模型。
  2. 使用訓練好的模型預測測試集的偏差 (delta_s2, delta_s3) 與角度 (DIP_a3)。
  3. 將預測的偏差轉換回實際的物理尺寸 (DIP_s2, DIP_s3)。
  4. 計算預測尺寸/角度與真實量測值之間的誤差 (單位: mm, deg)。
  5. 匯總所有結構的誤差，並計算總體的平均絕對誤差 (MAE) 和均方根誤差 (RMSE)。
"""
import argparse
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# --- 從現有模組導入所需功能 ---
# 假設所有 .py 檔案都位於同一目錄下
from compensation_utils import (
    FEATURES,
    TARGETS,
    average_by_structure,
)
from main import build_models  # 直接從 main.py 借用模型建立函式


def evaluate_loso_cv_with_dimensions(
    df_all_data: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """
    執行 LOSO 交叉驗證，並計算 s2, s3, a3 的實際尺寸/角度誤差。

    Args:
        df_all_data: 包含所有樣本的完整 DataFrame。
        args: 從命令列解析的參數，用於建立模型。

    Returns:
        一個 DataFrame，包含每個結構的詳細預測結果和誤差。
    """
    # 【修改】改用 groupby 來處理獨立結構，而不是建立 __key__ 欄位
    structure_groups = df_all_data.groupby(FEATURES)
    num_structures = len(structure_groups)
    print(f"\n[LOSO 評估] 發現 {num_structures} 個獨立結構。將逐一進行留一法驗證...")

    all_results: List[Dict[str, Any]] = []

    # 【修改】遍歷由 groupby 產生的每個結構組
    for i, (structure_params, df_test) in enumerate(structure_groups):

        # 1. 分割訓練集與測試集
        test_indices = df_test.index
        df_train = df_all_data.drop(test_indices).copy()

        # 將結構參數轉換為字典，方便後續使用
        structure_dict = dict(zip(FEATURES, structure_params))

        print(
            f"  ({i+1}/{num_structures}) 正在處理結構: s2={structure_dict['Design_s2(mm)']}, s3={structure_dict['Design_s3(mm)']}, a3={structure_dict['Design_a3(deg)']}"
        )

        # 2. 使用訓練集建立模型 (包含長度與角度模型)
        model_len, model_ang = build_models(args, df_train)

        # 3. 在測試集上進行預測
        pred_len_df = model_len.predict_df(df_test)
        pred_ang_df = model_ang.predict_df(df_test)

        # 4. 計算尺寸、角度與誤差
        for idx, row in df_test.iterrows():
            # --- s2, s3 尺寸計算 ---
            design_s2 = row["Design_s2(mm)"]
            design_s3 = row["Design_s3(mm)"]
            actual_dip_s2 = row["DIP_s2(mm)"]
            actual_dip_s3 = row["DIP_s3(mm)"]
            predicted_delta_s2 = pred_len_df.loc[idx, "delta_s2"]
            predicted_delta_s3 = pred_len_df.loc[idx, "delta_s3"]
            predicted_dip_s2 = design_s2 * (1 - predicted_delta_s2)
            predicted_dip_s3 = design_s3 * (1 - predicted_delta_s3)
            error_s2_mm = predicted_dip_s2 - actual_dip_s2
            error_s3_mm = predicted_dip_s3 - actual_dip_s3

            # --- a3 角度計算 ---
            actual_dip_a3 = row["DIP_a3(deg)"]
            predicted_dip_a3 = pred_ang_df.loc[idx, "DIP_a3(deg)"]
            error_a3_deg = predicted_dip_a3 - actual_dip_a3

            # 【修改】建立結果字典，並將結構參數字典合併進來
            result_row = {
                "Actual_DIP_s2(mm)": actual_dip_s2,
                "Predicted_DIP_s2(mm)": predicted_dip_s2,
                "Error_s2(mm)": error_s2_mm,
                "Actual_DIP_s3(mm)": actual_dip_s3,
                "Predicted_DIP_s3(mm)": predicted_dip_s3,
                "Error_s3(mm)": error_s3_mm,
                "Actual_DIP_a3(deg)": actual_dip_a3,
                "Predicted_DIP_a3(deg)": predicted_dip_a3,
                "Error_a3(deg)": error_a3_deg,
            }
            # 將代表結構的參數 (s1, s2, s3, a1, a2, a3) 加入到結果中
            result_row.update(structure_dict)
            all_results.append(result_row)

    return pd.DataFrame(all_results)


def main():
    # --- 參數解析 (與之前相同) ---
    ap = argparse.ArgumentParser(
        description="執行 LOSO 交叉驗證並計算 s2/s3/a3 物理尺寸與角度誤差"
    )
    ap.add_argument(
        "--file",
        type=str,
        default="analysis_results_0.6_0.9.xlsx",
        help="輸入的 Excel 檔案路徑",
    )
    ap.add_argument("--sheet", type=str, default="Sheet1", help="Excel 中的工作表名稱")
    ap.add_argument(
        "--average",
        action="store_true",
        help="是否對相同結構的樣本進行平均 (建議在 LOSO 中不要啟用，以評估原始數據)",
    )
    ap.add_argument(
        "--save-results",
        type=str,
        default="loso_dimension_results_0.6_09.xlsx",
        help="儲存詳細評估結果的 Excel 檔路徑",
    )
    

    group_len = ap.add_argument_group("Length Model Parameters")
    group_len.add_argument(
        "--length-model", type=str, default="huber", choices=["ols", "huber", "rf"]
    )
    group_len.add_argument(
        "--add-interactions",
        action="store_true",
        help="為長度模型加入 '邊長*邊長' 和 '邊長*角度' 交互作用",
    )
    group_len.add_argument(
        "--add-len-aa-interact",
        action="store_true",
        help="為長度模型加入 '角度*角度' 交互作用",
    )
    group_len.add_argument("--len-huber-alpha", type=float, default=1e-3)
    group_len.add_argument("--len-huber-eps", type=float, default=1.35)
    group_len.add_argument("--len-huber-max-iter", type=int, default=2000)
    group_len.add_argument("--scale-length", action="store_true")
    group_len.add_argument("--len-rf-n-est", type=int, default=300)
    group_len.add_argument("--len-rf-max-depth", type=int, default=None)
    group_len.add_argument("--len-rf-min-leaf", type=int, default=1)
    group_len.add_argument("--len-rf-max-features", type=float, default=1.0)
    group_len.add_argument("--len-rf-criterion", type=str, default="squared_error")
    group_len.add_argument("--len-add-ratios", action="store_true")
    group_len.add_argument("--len-add-sincos", action="store_true")

    group_ang = ap.add_argument_group("Angle Model Parameters (Required by builder)")
    group_ang.add_argument(
        "--angle-model", type=str, default="huber", choices=["ols", "rf", "huber"]
    )
    group_ang.add_argument("--add-angle-interactions", action="store_true")
    group_ang.add_argument("--add-angle-aa-interact", action="store_true")
    group_ang.add_argument("--angle-poly", type=int, default=2)
    group_ang.add_argument("--angle-ridge", type=float, default=1e-2)
    group_ang.add_argument("--rf-n-est", type=int, default=300)
    group_ang.add_argument("--rf-max-depth", type=int, default=None)
    group_ang.add_argument("--rf-min-leaf", type=int, default=1)
    group_ang.add_argument("--angle-huber-max-iter", type=int, default=1000)
    group_ang.add_argument("--angle-huber-alpha", type=float, default=1e-2)
    group_ang.add_argument("--angle-huber-eps", type=float, default=10)
    group_ang.add_argument("--scale-angle", action="store_true")
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

    df_use = average_by_structure(df_raw) if args.average else df_raw.copy()
    if args.average:
        print(
            "警告: 已對相同結構的樣本進行平均。若要評估每個獨立樣本，請不要使用 --average 旗標。"
        )

    initial_rows = len(df_use)
    df_use.dropna(
        subset=FEATURES + TARGETS + ["DIP_s2(mm)", "DIP_s3(mm)", "DIP_a3(deg)"],
        inplace=True,
    )
    if len(df_use) < initial_rows:
        print(f"提示：已從資料中移除了 {initial_rows - len(df_use)} 個包含缺失值的行。")

    if len(df_use) == 0:
        print("錯誤：處理後沒有可用的資料行。請檢查您的輸入檔案。")
        sys.exit(1)

    # --- 2. 執行 LOSO 評估 ---
    loso_results_df = evaluate_loso_cv_with_dimensions(df_use, args)

    # --- 3. 匯總與顯示結果 (與之前相同) ---
    print("\n\n" + "=" * 50)
    print("=== LOSO 尺寸與角度誤差評估總結 ===")
    print("=" * 50)

    mae_s2 = loso_results_df["Error_s2(mm)"].abs().mean()
    rmse_s2 = np.sqrt((loso_results_df["Error_s2(mm)"] ** 2).mean())

    mae_s3 = loso_results_df["Error_s3(mm)"].abs().mean()
    rmse_s3 = np.sqrt((loso_results_df["Error_s3(mm)"] ** 2).mean())

    mae_a3 = loso_results_df["Error_a3(deg)"].abs().mean()
    rmse_a3 = np.sqrt((loso_results_df["Error_a3(deg)"] ** 2).mean())

    print("\n--- 整體誤差 (Overall Error) ---")
    print(f"  s2 尺寸預測 MAE : {mae_s2:.6f} mm")
    print(f"  s2 尺寸預測 RMSE: {rmse_s2:.6f} mm")
    print("-" * 25)
    print(f"  s3 尺寸預測 MAE : {mae_s3:.6f} mm")
    print(f"  s3 尺寸預測 RMSE: {rmse_s3:.6f} mm")
    print("-" * 25)
    print(f"  a3 角度預測 MAE : {mae_a3:.6f} deg")
    print(f"  a3 角度預測 RMSE: {rmse_a3:.6f} deg")

    # --- 4. 儲存詳細結果 ---
    if args.save_results:
        try:
            # 重新排列欄位順序，讓結構參數在前
            cols_order = FEATURES + [
                col for col in loso_results_df.columns if col not in FEATURES
            ]
            loso_results_df = loso_results_df[cols_order]

            loso_results_df.to_excel(
                args.save_results, index=False, sheet_name="LOSO_Dimension_Details"
            )
            print(f"\n[已儲存] 詳細的 LOSO 評估結果已儲存至 -> {args.save_results}")
        except Exception as e:
            print(f"\n[錯誤] 無法儲存結果檔案: {e}")


if __name__ == "__main__":
    main()
