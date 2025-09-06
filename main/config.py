import os
import numpy as np

# === ES 參數 ===
POP_SIZE = 10
OFFSPRING_PARENT_RATIO = 10
OFFSPRING_SIZE = POP_SIZE * OFFSPRING_PARENT_RATIO
INITIAL_SIGMA_FACTOR = 0.15
N_GENERATIONS = 100
SIDE_BOUND = [0.6, 0.9]
ANGLE_BOUND = [30.0, 90.0]

# === 並行處理設定 ===
MAX_WORKERS = os.cpu_count() or 4

# === 模型設定 ===
MODEL_TYPE = "Huber"  # 可選: 'Huber', 'RF', 'OLS'

# === 模型特徵設定 (預設值) ===
ADD_RATIOS = False
ADD_SINCOS = False
ADD_INTERACTIONS = True  # s*s, s*a 交互作用
ADD_AA_INTERACT = True  # a*a 交互作用

# === 演算法內部常數 ===
N_VARS = 3  # 變數數量 (s1, s2, a1)
TAU_PRIME = 1 / np.sqrt(2 * N_VARS)
TAU = 1 / np.sqrt(2 * np.sqrt(N_VARS))
VAR_RANGES = np.array(
    [
        SIDE_BOUND[1] - SIDE_BOUND[0],
        SIDE_BOUND[1] - SIDE_BOUND[0],
        ANGLE_BOUND[1] - ANGLE_BOUND[0],
    ],
    dtype=float,
)
SIGMA_MIN = VAR_RANGES * 0.01
GLOBAL_SEED = 42

# === 重複評估檢查 ===
CHECK_DUPLICATES = True  # 是否檢查並跳過與歷史紀錄中相似的參數
DUPLICATE_TOLERANCE = [0.001, 0.001, 0.05]  # 判斷重複的容許誤差 [s1, s2, a1]

# === 路徑設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
SAVE_ROOT = os.path.join(BASE_DIR, "GA_population")
LOG_DIR = r"C:\Users\cchih\OneDrive - NTHU\msi"
TRAIN_DATA_PATH = os.path.join(
    BASE_DIR, "main", "model_excel", "analysis_results_0.6_0.9.xlsx"
)


# 確保目錄存在
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
