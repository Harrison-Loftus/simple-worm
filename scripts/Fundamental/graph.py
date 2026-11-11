import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")

I_vals = [0.25]

frequencies = []
wavelengths = []

velocities = []


fig, ax = plt.subplots(figsize=(10, 6))

for I_val in I_vals:

    data = np.load(
        OUTPUT_DIR / f"simulation_I_{I_val:.3f}.npz"
    )

    times = data["times"]
    A_V = data["A_V"]

    # Midpoint muscle activation
    A_V_mid = A_V[1, :]

    ax.plot(times, A_V_mid, label=f"I = {I_val:.2f}")

ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$A_V$")
ax.set_title("Midpoint ventral muscle activation against time")
ax.legend()


plt.show()



fig, ax = plt.subplots(figsize=(10, 6))

for I_val in I_vals:

    data = np.load(
        OUTPUT_DIR / f"simulation_I_{I_val:.3f}.npz"
    )

    times = data["times"]
    kappa = data["kappas"]

    # Midpoint curvature
    kappa_mid = kappa[1, :]

    ax.plot(times, kappa_mid, label=f"I = {I_val:.2f}")

ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\kappa$")
ax.set_title("Midpoint curvature against time")
ax.legend()


plt.show()