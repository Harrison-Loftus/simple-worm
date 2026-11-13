

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.animation import *

OUTPUT_DIR = Path("outputs/Sweeps/Descretisation/N_vals_sweep")

N_vals = [3*2**i for i in range(4,8)]
print(N_vals)
frequencies = []
wavelengths = []

velocities = []

periods = []

fig, ax = plt.subplots(figsize=(10, 6))

for N_val in N_vals:

    data = np.load(OUTPUT_DIR / f"simulation_I_{N_val:.3f}.npz")

    worm_positions = data["worm_positions"]

    # anim = create_worm_animation(worm_positions, dt=data["config_dt"])
    # anim.save(OUTPUT_DIR / f"worm_animation_N_{N_val:.3f}.mp4", writer="ffmpeg", fps=int(1/data["config_dt"]))

    times = data["times"]
    A_V = data["A_V"]

    period = data["period"]
    periods.append(period)

    frequencies.append(data["frequency"])

    wavelength = data["wavelength"]
    wavelengths.append(wavelength)

    velocities.append(data["mean_velocity"])

    # Midpoint muscle activation
    A_V_mid = A_V[1, :]

    ax.plot(times, A_V_mid, label=f"N = {N_val:.2f}")

ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$A_V$")
ax.set_title("Midpoint ventral muscle activation against time")
ax.legend()


plt.show()



fig, ax = plt.subplots(figsize=(10, 6))

for N_val in N_vals:

    data = np.load(
        OUTPUT_DIR / f"simulation_I_{N_val:.3f}.npz"
    )

    times = data["times"]
    kappa = data["kappas"]

    # Midpoint curvature
    kappa_mid = kappa[1, :]

    ax.plot(times, kappa_mid, label=f"N = {N_val:.2f}")

ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\kappa$")
ax.set_title("Midpoint curvature against time")
ax.legend()


plt.show()

relative_period_errors = np.abs(np.array(periods) - periods[-1]) / periods[-1]

plt.figure(figsize=(10, 6))
plt.plot(N_vals, relative_period_errors, marker='o')
plt.xlabel("N")
plt.ylabel("Relative Period Error")
plt.title("Relative Period Error vs N")
plt.show()

print("Relative Period Errors:", relative_period_errors)

print("Frequencies:", frequencies)
print("Wavelengths:", wavelengths)
print("Velocities:", velocities)
print("Periods:", periods)