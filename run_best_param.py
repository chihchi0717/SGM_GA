import os
import shutil

from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

ANGLE_WEIGHTS = [1, 2, 5, 7, 5, 8.5, 1.5, 2]


def main():
    sid_ang = [0.46, 0.95, 85]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "GA_population", "best_param")
    os.makedirs(folder, exist_ok=True)

    # Copy simulation macro
    scm_src = os.path.join(base_dir, "Macro", "Sim.scm")
    shutil.copy2(scm_src, os.path.join(folder, "Sim.scm"))

    # Build CAD model
    Build_model(sid_ang, mode="triangle", folder=folder)

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
