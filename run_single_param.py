import os
import shutil

from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

ANGLE_WEIGHTS = [1, 2, 5, 7, 5, 8.5, 1.5, 2]


def main():
    # sid_ang = [0.46, 0.95, 85]
    # sid_ang = [0.0502, 0.0355, 45] #liao
    sid_ang = [0.45, 0.57, 79.91]
    # sid_ang = [0.48, 0.98, 85]
    # sid_ang = [0.62, 0.96, 67]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # output_dir = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\MOO_best_PS_OM[0.9, 0.9, 30]\SIM"
    # folder = os.path.join(output_dir, "[0.9, 0.9, 30]_N1.3_F2_5_3_sub0.6)_allele")
    output_dir = r"C:\Users\cchih\Desktop\NTHU\MasterThesis\research_log\best_params\Optimize_Regression\[0.45, 0.57, 79.91]\SIM_fillet"
    folder = os.path.join(output_dir, "[0.45, 0.57, 79.91]_N1.2536_F2_53_34_sub0.6")
    os.makedirs(folder, exist_ok=True)

    # Copy simulation macro
    scm_src = os.path.join(base_dir, "Macro", "Sim.scm")
    shutil.copy2(scm_src, os.path.join(folder, "Sim.scm"))

    # Build CAD model
    Build_model(
        sid_ang,
        mode="triangle",
        folder=folder,
        fillet=2,
        light_source_length=1,
        radius_inside=0.053,
        radius_vertex=0.034,
        substrate=0.6,
    )

    # Run TracePro simulation
    tracepro_fast(os.path.join(folder, "Sim.scm"))

    # Evaluate fitness
    fitness, efficiency, process_score, angle_effs = evaluate_fitness(folder, sid_ang)

    print(f"Fitness: {fitness:.6f}")
    print(f"Efficiency: {efficiency:.6f}")
    print(f"Process Score: {process_score:.6f}")

    print("\nWeights and per-angle efficiencies:")
    for angle, eff, w in zip(range(10, 90, 10), angle_effs, ANGLE_WEIGHTS):
        print(f"  Angle {angle}° -> eff: {eff:.6f}, weight: {w}")


if __name__ == "__main__":
    main()
