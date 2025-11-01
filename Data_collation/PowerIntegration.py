import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# === User settings ===
folder = Path(
    r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\Optimize_Regression\[0.45, 0.57, 79.91]\SIM_fillet\[0.45, 0.57, 79.91]_N1.2536_F2_53_34_sub0.6_new"
)
save_per_file_breakdown = True
summary_csv_path = folder / "polar_summary.csv"

# From RCD data
ANGLE_TARGET = {
    10: 0.29148,
    20: 0.28760,
    30: 0.28616,
    40: 0.29145,
    50: 0.28646,
    60: 0.29230,
    70: 0.28101,
    80: 0.26990,
}


def read_polar_file(path: Path):
    theta_list, trans_list, refl_list = [], [], []
    after_two_col = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
            if len(nums) == 2:
                after_two_col = True
                continue
            if not after_two_col:
                continue
            if len(nums) == 3:
                theta_list.append(float(nums[0]))
                trans_list.append(float(nums[1]))
                refl_list.append(float(nums[2]))
    theta_deg = np.asarray(theta_list, dtype=float)
    I_trans = np.asarray(trans_list, dtype=float)
    I_refl = np.asarray(refl_list, dtype=float)

    order = np.argsort(theta_deg)
    return theta_deg[order], I_trans[order], I_refl[order]


def midpoint_bin_widths(theta_rad: np.ndarray):
    if theta_rad.size == 0:
        return theta_rad
    edges = np.empty(theta_rad.size + 1, dtype=float)
    if theta_rad.size >= 2:
        edges[1:-1] = 0.5 * (theta_rad[1:] + theta_rad[:-1])
        left_gap = theta_rad[1] - theta_rad[0]
        right_gap = theta_rad[-1] - theta_rad[-2]
    else:
        edges[1:-1] = np.nan
        left_gap = right_gap = 0.0
    edges[0] = max(0.0, theta_rad[0] - 0.5 * left_gap)
    edges[-1] = min(math.pi, theta_rad[-1] + 0.5 * right_gap)
    return np.diff(edges)


def process_one_file(path: Path, save_breakdown: bool = True):
    theta_deg, I_trans, I_refl = read_polar_file(path)
    if theta_deg.size == 0:
        return None

    angle_deg = int(re.findall(r"polar-(\d+)", path.name)[0])
    if angle_deg not in ANGLE_TARGET:
        raise ValueError(f"No P_target defined for angle {angle_deg}° in ANGLE_TARGET.")
    P_target = ANGLE_TARGET[angle_deg]

    theta_rad = np.deg2rad(theta_deg)
    I_sum = I_trans + I_refl

    # A = ∫ (I_trans + I_refl) sinθ dθ
    A = np.trapezoid(I_sum * np.sin(theta_rad), theta_rad)

    # Δφ = P_target / A
    delta_phi_rad = P_target / A
    delta_phi_deg = math.degrees(delta_phi_rad)

    # dΩ_i = sinθ_i * Δθ_i * Δφ
    dtheta_i = midpoint_bin_widths(theta_rad)
    sin_theta = np.sin(theta_rad)
    dOmega = sin_theta * dtheta_i * delta_phi_rad

    P_bin_trans = I_trans * dOmega
    P_bin_refl = I_refl * dOmega
    P_bin_sum = P_bin_trans + P_bin_refl

    P_trans = P_bin_trans.sum()
    P_refl = P_bin_refl.sum()
    P_total = P_bin_sum.sum()

    if save_breakdown:
        df = pd.DataFrame(
            {
                "theta_deg": theta_deg,
                "I_trans_W_per_sr": I_trans,
                "I_refl_W_per_sr": I_refl,
                "I_sum_W_per_sr": I_sum,
                "sin_theta": sin_theta,
                "dtheta_rad": dtheta_i,
                "delta_phi_rad": np.full_like(theta_rad, delta_phi_rad),
                "dOmega_sr": dOmega,
                "P_bin_trans_W": P_bin_trans,
                "P_bin_refl_W": P_bin_refl,
                "P_bin_sum_W": P_bin_sum,
            }
        )
        out_path = path.with_name(path.stem + "_breakdown.csv")
        df.to_csv(out_path, index=False)

    return {
        "file": path.name,
        "angle_deg": angle_deg,
        "A_int": A,
        "delta_phi_rad": delta_phi_rad,
        "delta_phi_deg": delta_phi_deg,
        "P_trans_W": P_trans,
        "P_refl_W": P_refl,
        "P_total_W": P_total,
        "P_target_W": P_target,
        "abs_err_W": P_total - P_target,
        "rel_err_%": (P_total / P_target - 1.0) * 100.0 if P_target != 0 else np.nan,
        "n_samples": int(theta_deg.size),
        "theta_min_deg": float(theta_deg.min()),
        "theta_max_deg": float(theta_deg.max()),
    }


def main():
    files = sorted(folder.glob("polar-*.txt"))
    if not files:
        print(f"No files matched in folder:\n{folder}")
        return
    rows = []
    for fp in files:
        try:
            res = process_one_file(fp, save_breakdown=save_per_file_breakdown)
            if res is not None:
                rows.append(res)
                print(
                    f"Processed {fp.name}: Δφ={res['delta_phi_rad']:.6f} rad "
                    f"({res['delta_phi_deg']:.3f}°)  "
                    f"P_trans={res['P_trans_W']:.6f} W, P_refl={res['P_refl_W']:.6f} W,"
                )
            else:
                print(f"Skipped empty file: {fp.name}")
        except Exception as e:
            print(f"Error processing {fp.name}: {e}")
    if rows:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(summary_csv_path, index=False)
        print("\n=== Summary saved ===")
        print(summary_csv_path)


if __name__ == "__main__":
    main()
