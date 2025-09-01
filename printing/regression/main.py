# -*- coding: utf-8 -*-
"""
main.py
- 主執行檔，整合所有模組。
- 流程：
  1. 解析命令列參數 (包含選擇補償策略)。
  2. 讀取並預處理資料。
  3. 根據參數建立並訓練模型。
  4. 執行選擇的補償策略。
  5. 驗證結果並產出評估報告。
"""
import argparse
import numpy as np
import pandas as pd
import sys

# 從自訂模組導入所需功能
from compensation_utils import (
    FEATURES,
    TARGETS,
    apply_geometric_constraints,
    average_by_structure,
    evaluate_overall,
    evaluate_per_structure,
    evaluate_kfold_cv,
    evaluate_loso_cv,
)
from compensation_models import (
    LinearOLS,
    LengthModelRF,
    LengthModelHuber,
    AngleModelOLS,
    AngleModelHuber,
    AngleModelRF,
)
from compensation_strategies import (
    compensate_with_jacobian,
    compensate_without_jacobian,
    compensate_with_random_search,
    compensate_with_genetic_algorithm,
)


def build_models(args, df_train):
    """根據參數建立並訓練長度與角度模型。"""
    # 建立長度模型
    if args.length_model == "ols":
        model_len = LinearOLS(ridge=1e-9)
    elif args.length_model == "huber":
        model_len = LengthModelHuber(
            alpha=args.len_huber_alpha,
            epsilon=args.len_huber_eps,
            max_iter=args.len_huber_max_iter,
            scale=args.scale_length,
            add_ratios=args.len_add_ratios,
            add_sincos=args.len_add_sincos,
            add_interactions=args.add_interactions,
        )
    else:  # rf
        model_len = LengthModelRF(
            n_estimators=args.len_rf_n_est,
            max_depth=args.len_rf_max_depth,
            min_samples_leaf=args.len_rf_min_leaf,
            random_state=42,
            add_ratios=args.len_add_ratios,
            add_sincos=args.len_add_sincos,
            max_features=args.len_rf_max_features,
            criterion=args.len_rf_criterion,
            add_interactions=args.add_interactions,
        )
    model_len.fit(df_train)

    # 建立角度模型
    if args.angle_model == "ols":
        model_ang = AngleModelOLS(
            degree=args.angle_poly,
            ridge=args.angle_ridge,
            add_sincos=args.add_angle_sincos,
            add_ratios=args.add_ratios,
            add_interactions=args.add_angle_interactions,
        )
    elif args.angle_model == "huber":
        model_ang = AngleModelHuber(
            alpha=args.angle_huber_alpha,
            epsilon=args.angle_huber_eps,
            max_iter=args.angle_huber_max_iter,
            scale=args.scale_angle,
            add_sincos=args.add_angle_sincos,
            add_ratios=args.add_ratios,
            add_interactions=args.add_angle_interactions,
        )
    else:  # rf
        model_ang = AngleModelRF(
            n_estimators=args.rf_n_est,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_leaf,
            random_state=42,
            add_sincos=args.add_angle_sincos,
            add_ratios=args.add_ratios,
            add_interactions=args.add_angle_interactions,
        )
    model_ang.fit(df_train)

    return model_len, model_ang


def main():
    # --- 參數解析 ---
    ap = argparse.ArgumentParser(description="迭代式事前預測補償工具")
    ap.add_argument(
        "--file",
        type=str,
        default="analysis_results0831.xlsx",
        help="輸入的 Excel 檔案路徑",
    )
    ap.add_argument("--sheet", type=str, default="Sheet1", help="Excel 中的工作表名稱")
    ap.add_argument(
        "--average", action="store_true", help="是否對相同結構的樣本進行平均"
    )
    ap.add_argument("--save-avg", type=str, default=None, help="儲存平均後資料的路徑")
    ap.add_argument("--eval", action="store_true", help="是否執行模型評估")
    ap.add_argument(
        "--cv", type=str, default="0", help="交叉驗證：整數 K-fold 或 'loso'"
    )
    ap.add_argument("--save-report", type=str, default=None, help="儲存評估報告的路徑")
    ap.add_argument(
        "--add-interactions", action="store_true", help="為【長度模型】加入交互作用特徵"
    )

    # 長度模型參數
    ap.add_argument(
        "--length-model", type=str, default="huber", choices=["ols", "huber", "rf"]
    )
    ap.add_argument("--len-huber-alpha", type=float, default=1e-3)
    ap.add_argument("--len-huber-eps", type=float, default=1.35)
    ap.add_argument("--len-huber-max-iter", type=int, default=2000)
    ap.add_argument("--scale-length", action="store_true")
    ap.add_argument("--len-rf-n-est", type=int, default=300)
    ap.add_argument("--len-rf-max-depth", type=int, default=None)
    ap.add_argument("--len-rf-min-leaf", type=int, default=1)
    ap.add_argument("--len-rf-max-features", type=float, default=1.0)
    ap.add_argument("--len-rf-criterion", type=str, default="squared_error")
    ap.add_argument("--len-add-ratios", action="store_true")
    ap.add_argument("--len-add-sincos", action="store_true")

    # 角度模型參數
    ap.add_argument(
        "--angle-model", type=str, default="rf", choices=["ols", "rf", "huber"]
    )
    ap.add_argument(
        "--angle-poly",
        type=int,
        default=2,
        help="OLS 角度模型的多項式階數 (若啟用交互作用則此項無效)",
    )
    ap.add_argument("--angle-ridge", type=float, default=1e-2)
    ap.add_argument("--rf-n-est", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-leaf", type=int, default=1)
    ap.add_argument(
        "--angle-huber-max-iter",
        type=int,
        default=2000,
        help="角度 Huber 模型的最大迭代次數",
    )
    ap.add_argument(
        "--add-angle-interactions",
        action="store_true",
        help="為【角度模型】加入二次交互作用特徵",
    )
    ap.add_argument(
        "--angle-huber-alpha",
        type=float,
        default=1e-3,
        help="角度 Huber 模型的 Alpha (正規化強度)",
    )
    ap.add_argument(
        "--angle-huber-eps",
        type=float,
        default=1.35,
        help="角度 Huber 模型的 Epsilon (穩健性參數)",
    )
    ap.add_argument("--scale-angle", action="store_true")
    ap.add_argument("--add-angle-sincos", action="store_true")
    ap.add_argument("--add-ratios", action="store_true")

    # 策略選擇開關
    ap.add_argument(
        "--strategy",
        type=str,
        default="jacobian",
        choices=["jacobian", "direct", "search", "ga"],
        help="選擇補償策略: 'jacobian' (使用Jacobian矩陣) 或 'direct' (直接回饋)",
    )
    args = ap.parse_args()

    # [修正] 將 strategy 參數轉為小寫，使其不區分大小寫
    if args.strategy:
        args.strategy = args.strategy.lower()

    # --- 1. 讀取與處理資料 ---
    try:
        df_raw = pd.read_excel(args.file, sheet_name=args.sheet)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{args.file}'。請確認檔案路徑是否正確。")
        sys.exit(1)

    # [修正] 增加對必要欄位的檢查，並使用正確的欄位名稱
    required_columns = FEATURES + [
        "DIP_s2(mm)",
        "DIP_s3(mm)",
        "DIP_a3(deg)",
    ]

    missing_columns = [col for col in required_columns if col not in df_raw.columns]

    if missing_columns:
        print("\n錯誤: 輸入的 Excel 檔案中缺少必要的欄位。")
        print(f"缺少欄位: {', '.join(missing_columns)}")
        print(f"檔案中找到的欄位: {', '.join(df_raw.columns)}")
        sys.exit(1)

    # [修正] 根據您的回饋，將目標變數改為相對誤差率
    eps = 1e-9  # 避免除以零
    df_raw["delta_s2"] = (df_raw["Design_s2(mm)"] - df_raw["DIP_s2(mm)"]) / (
        df_raw["Design_s2(mm)"] + eps
    )
    df_raw["delta_s3"] = (df_raw["Design_s3(mm)"] - df_raw["DIP_s3(mm)"]) / (
        df_raw["Design_s3(mm)"] + eps
    )

    df_use = average_by_structure(df_raw) if args.average else df_raw.copy()

    # 在訓練前，確保移除任何包含 NaN 的資料列，避免模型出錯
    initial_rows = len(df_use)
    df_use.dropna(subset=FEATURES + TARGETS, inplace=True)
    if len(df_use) < initial_rows:
        print(
            f"警告：已從訓練資料中移除了 {initial_rows - len(df_use)} 行，因為它們包含缺失值 (NaN)。"
        )

    if args.average and args.save_avg:
        df_use.to_excel(args.save_avg, index=False)
        print(f"平均後的資料已儲存至：{args.save_avg}")

    # --- 2. 訓練模型 ---
    print("\n=== 正在訓練模型 ===")
    model_len, model_ang = build_models(args, df_use)
    print("模型訓練完成。")

    # --- 3. 執行補償策略 ---
    target_design_independent = {
        "Design_s2(mm)": 0.59,#0.75,
        "Design_s3(mm)": 0.59,#0.69,
        "Design_a3(deg)": 33,#58,
    }
    target_design = apply_geometric_constraints(target_design_independent)

    print("\n--- 初始目標設計 (已應用幾何約束) ---")
    for k, v in target_design.items():
        print(f"  - {k}: {v:.6f}")

    compensated_design = None
    if args.strategy == "jacobian":
        compensated_design = compensate_with_jacobian(
            model_len, model_ang, target_design
        )
    elif args.strategy == "direct":
        compensated_design = compensate_without_jacobian(
            model_len, model_ang, target_design
        )
    elif args.strategy == "search":
        compensated_design = compensate_with_random_search(
            model_len, model_ang, target_design, df_use
        )
    elif args.strategy == "ga":
        compensated_design = compensate_with_genetic_algorithm(
            model_len, model_ang, target_design, df_use
        )

    if compensated_design is None:
        print(f"錯誤：策略 '{args.strategy}' 未成功執行或未返回設計方案。")
        sys.exit(1)

    # --- 4. 驗證最終結果 ---
    print("\n\n--- 最終補償設計方案 ---")
    for k, v in compensated_design.items():
        print(f"  - {k}: {v:.6f}")

    print("\n--- 驗證最終設計方案 ---")
    compensated_df = pd.DataFrame([compensated_design])
    final_pred_len = model_len.predict_df(compensated_df)
    final_pred_ang = model_ang.predict_df(compensated_df)
    final_pred_ds2 = final_pred_len["delta_s2"].iloc[0]
    final_pred_ds3 = final_pred_len["delta_s3"].iloc[0]
    final_pred_ma3 = final_pred_ang["DIP_a3(deg)"].iloc[0]

    # [修正] 根據您的回饋，使用正確的公式反推最終成品尺寸
    final_s2_pred = compensated_design["Design_s2(mm)"] * (1 - final_pred_ds2)
    final_s3_pred = compensated_design["Design_s3(mm)"] * (1 - final_pred_ds3)
    final_a3_pred = final_pred_ma3  # 角度模型預測的是絕對值，維持不變

    # 使用預測出的 s2, s3, a3，透過幾何約束反算出 s1, a1, a2
    final_product_independent = {
        "Design_s2(mm)": final_s2_pred,
        "Design_s3(mm)": final_s3_pred,
        "Design_a3(deg)": final_a3_pred,
    }
    final_product_all = apply_geometric_constraints(final_product_independent)

    final_s1 = final_product_all["Design_s1(mm)"]
    final_s2 = final_product_all["Design_s2(mm)"]
    final_s3 = final_product_all["Design_s3(mm)"]
    final_a1 = final_product_all["Design_a1(deg)"]
    final_a2 = final_product_all["Design_a2(deg)"]
    final_a3 = final_product_all["Design_a3(deg)"]

    print("\n  - 預測的最終成品尺寸:")
    print(f"    - s1: {final_s1:.6f} mm (目標: {target_design['Design_s1(mm)']:.6f})")
    print(f"    - s2: {final_s2:.6f} mm (目標: {target_design['Design_s2(mm)']:.6f})")
    print(f"    - s3: {final_s3:.6f} mm (目標: {target_design['Design_s3(mm)']:.6f})")
    print(f"    - a1: {final_a1:.6f} deg (目標: {target_design['Design_a1(deg)']:.6f})")
    print(f"    - a2: {final_a2:.6f} deg (目標: {target_design['Design_a2(deg)']:.6f})")
    print(f"    - a3: {final_a3:.6f} deg (目標: {target_design['Design_a3(deg)']:.6f})")

    # --- 5. 執行評估與儲存報告 ---
    if args.eval or args.cv != "0" or args.save_report:
        print("\n\n=== 模型評估 ===")

        def model_builder(df_tr):
            return build_models(args, df_tr)

        overall = evaluate_overall(df_use, model_len, model_ang)
        print("\n-- 整體評估 (訓練集) --\n", overall.round(4))

        if args.save_report:
            with pd.ExcelWriter(args.save_report) as xl:
                overall.to_excel(xl, index=False, sheet_name="overall_train")

                per_struct = evaluate_per_structure(df_raw, model_len, model_ang)
                per_struct.to_excel(xl, index=False, sheet_name="per_structure")

                if args.cv.isdigit() and int(args.cv) > 1:
                    k = int(args.cv)
                    cv_df = evaluate_kfold_cv(df_use, k=k, model_builder=model_builder)
                    print(f"\n-- {k}-折交叉驗證 --\n", cv_df.round(4))
                    cv_df.to_excel(xl, index=False, sheet_name=f"{k}_fold_cv")
                elif args.cv.lower() == "loso":
                    loso_overall, loso_detail = evaluate_loso_cv(
                        df_use, model_builder=model_builder
                    )
                    print("\n-- LOSO 交叉驗證 (總體) --\n", loso_overall.round(4))
                    loso_overall.to_excel(xl, index=False, sheet_name="loso_cv_overall")
                    loso_detail.to_excel(xl, index=False, sheet_name="loso_cv_detail")

                # [新增] 儲存模型係數
                if hasattr(model_len, "get_coefficients_df"):
                    len_coeffs = model_len.get_coefficients_df()
                    if len_coeffs is not None:
                        len_coeffs.to_excel(xl, sheet_name="len_model_coeffs")
                        print("    - 已儲存長度模型係數")

                if hasattr(model_ang, "get_coefficients_df"):
                    ang_coeffs = model_ang.get_coefficients_df()
                    if ang_coeffs is not None:
                        ang_coeffs.to_excel(xl, sheet_name="ang_model_coeffs")
                        print("    - 已儲存角度模型係數")

                # 儲存本次執行的參數設定
                params_df = pd.DataFrame([vars(args)])
                params_df.to_excel(xl, index=False, sheet_name="run_parameters")

            print(f"\n[已儲存] 評估報告 -> {args.save_report}")


if __name__ == "__main__":
    try:
        # 確保終端機能正確顯示中文字元
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()


# python main.py --average --eval --cv 5 --length-model huber --len-huber-alpha 1 --len-huber-eps 1000 --len-huber-max-iter 10000 --scale-length --add-interactions --len-add-ratios --len-add-sincos --angle-model huber --angle-huber-alpha 1e-1 --angle-huber-eps 1000 --scale-angle --angle-ridge 0.01 --angle-huber-max-iter 10000  --add-angle-interactions  --add-ratios --save-report "model_report0831.xlsx" --strategy ga
