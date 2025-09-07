import os
import numpy as np

# === ES 參數 ===
# μ: 父代數量
POP_SIZE = 3
# λ: 後代數量
OFFSPRING_SIZE = 3 * POP_SIZE
N_GENERATIONS = 100
INITIAL_SIGMA_FACTOR = 0.15  # 初始 Sigma 與變數範圍的比例
SIDE_BOUND = [0.6, 0.9]
ANGLE_BOUND = [30.0, 90.0]

# === 演化策略設定 ===
SELECTION_STRATEGY = "plus"
MUTATION_ADAPTATION = "adaptive"

# === (新增) 多樣性控制 ===
# 是否啟用懲罰機制以避免過早收斂
USE_DIVERSITY_CONTROL = True
# 懲罰強度因子。數值越大，對相似個體的懲罰越重。建議範圍: 0.05 ~ 0.5
DIVERSITY_PENALTY_FACTOR = 0.15

# === 並行處理設定 ===
MAX_WORKERS = os.cpu_count() or 4

# === 模型設定 ===
MODEL_TYPE = "Huber"
ADD_RATIOS = False
ADD_SINCOS = False
ADD_INTERACTIONS = True
ADD_AA_INTERACT = True

# === 演算法內部常數 ===
N_VARS = 3
TAU_PRIME = 1 / np.sqrt(2 * N_VARS)
TAU = 1 / np.sqrt(2 * np.sqrt(N_VARS))
VAR_RANGES = np.array(
    [
        SIDE_BOUND[1] - SIDE_BOUND[0],
        SIDE_BOUND[1] - SIDE_BOUND[0],
        ANGLE_BOUND[1] - ANGLE_BOUND[0],
    ]
)

# === 路徑設定 ===
LOG_DIR = r"C:\Users\cchih\OneDrive - NTHU\msi"
TRAIN_DATA_PATH = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\GA\SGM_GA\main\model_excel\analysis_results_0.6_0.9.xlsx"
