import os
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")
from analyze_profile import (
    analyze_profile,
)  # 引入剛剛的函式（另存為 analyze_profile.py）

# === 設定資料夾 ===
folder_path = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\202508\NEAF"
figure_path = os.path.join(folder_path, "figures")
summary_list = []

# === 遍歷資料夾內所有 CSV ===
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        print(f"\n處理檔案：{file}")
        try:
            angle_df = analyze_profile(file_path, save_dir=figure_path, show_plot=False)
            if angle_df is not None:
                angle_df.insert(0, "Filename", os.path.splitext(file)[0])
                summary_list.append(angle_df)
        except Exception as e:
            print(f"失敗：{file}，原因：{e}")

# === 合併與儲存角度總表 ===
if summary_list:
    all_angles_df = pd.concat(summary_list, ignore_index=True)
    summary_csv_path = os.path.join(folder_path, "角度總表.csv")
    all_angles_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n所有角度已儲存：{summary_csv_path}")
else:
    print("⚠ 沒有有效角度結果。")
