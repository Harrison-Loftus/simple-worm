import numpy as np
from pathlib import Path

dt = 1.0e-2

SCRIPT_NAME = "bifurcation_sweep_for_paper"

BASE_DIR = Path("outputs") / SCRIPT_NAME
anim_dir = BASE_DIR / "animations"

epsilon_p_vals = np.array([1.0])
I_vals = np.arange(0.0, 0.408, 0.008)

velocities = np.full(
    (len(epsilon_p_vals), len(I_vals)),
    np.nan
)

for i, epsilon_p in enumerate(epsilon_p_vals):
    for j, I in enumerate(I_vals):

        file = anim_dir / (
            f"worm_positions_eps{epsilon_p:.2f}_I{I:.4f}.npy"
        )

        if not file.exists():
            print(f"Missing file: {file}")
            continue

        worm_positions = np.load(file)

        # Shape: (time steps, body points, coordinates)
        com = np.mean(worm_positions, axis=1)

        # Final 20 seconds
        n_steps = int(20 / dt)
        com_final = com[-n_steps:]

        # Net centre-of-mass displacement
        displacement = com_final[-1] - com_final[0]

        # Actual elapsed time
        elapsed_time = (len(com_final) - 1) * dt

        velocity = np.linalg.norm(displacement) / elapsed_time

        velocities[i, j] = velocity

        print(
            f"I={I:.3f}, "
            f"velocity={velocity:.5f} mm/s"
        )

np.save(BASE_DIR / "velocities.npy", velocities)