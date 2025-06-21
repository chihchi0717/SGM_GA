import os
import shutil

from PYtoAutocad import Build_model
from TracePro_fast import tracepro_fast
from txt_new import evaluate_fitness

# === Configuration Parameters ===
# Parameters for PYtoAutocad.Build_model
MODE = "triangle"
FILLET = 1
FILLET_RADIUS = 0.022
LIGHT_SIZE = 0.5

# Parameters for txt_new.evaluate_fitness
RETURN_UNIFORMITY = False
PROCESS_WEIGHT = 2.0

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
    Build_model(
        sid_ang,
        mode=MODE,
        folder=folder,
        fillet=FILLET,
        radius=FILLET_RADIUS,
        light_source_length=LIGHT_SIZE,
    )

    # Run TracePro simulation
    tracepro_fast(os.path.join(folder, "Sim.scm"))

    # Evaluate fitness
    fitness, efficiency, process_score, angle_effs = evaluate_fitness(
        folder,
        sid_ang,
        return_uniformity=RETURN_UNIFORMITY,
        process_weight=PROCESS_WEIGHT,
    )

    print(f"Fitness: {fitness:.6f}")
    print(f"Efficiency: {efficiency:.6f}")
    print(f"Process Score: {process_score:.6f}")

    print("\nWeights and per-angle efficiencies:")
    for angle, eff, w in zip(range(10, 90, 10), angle_effs, ANGLE_WEIGHTS):
        print(f"  Angle {angle}° -> eff: {eff:.6f}, weight: {w}")


if __name__ == "__main__":
    main()
