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

# === 路徑設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = os.path.join(BASE_DIR, "GA_population")
LOG_DIR = r"C:\Users\cchih\OneDrive - NTHU\msi"
TRAIN_DATA_PATH = r".\model_excel\analysis_results_0.6_0.9.xlsx"

# 確保目錄存在
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
