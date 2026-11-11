import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Directory containing saved results
SCRIPT_NAME = "bifurcation_sweep_for_paper"
BASE_DIR = Path("outputs") / SCRIPT_NAME

# Parameter values from original sweep
I_vals = np.arange(0.0, 0.408, 0.008)

# Load saved results
wavelengths = np.load(BASE_DIR / "wavelengths.npy")
frequencies = np.load(BASE_DIR / "frequencies.npy")
periods = np.load(BASE_DIR / "periods.npy")
velocities = np.load(BASE_DIR / "velocities.npy")
max_curvatures = np.load(BASE_DIR / "max_curvatures.npy")

# Proprioceptive strength epsilon_p = 1.0
i = 0

# Quantities to plot
quantities = [
    (wavelengths[i], "Wavelength", r"Wavelength $\lambda/L$", "wavelength"),
    (frequencies[i], "Frequency", "Frequency (Hz)", "frequency"),
    (periods[i], "Period", "Period (s)", "period"),
    (velocities[i], "Velocity", "Velocity (mm/s)", "velocity"),
    (max_curvatures[i], "Maximum Curvature", r"Maximum curvature (mm$^{-1}$)", "max_curvature")
]

# Create and save separate figures
for values, title, ylabel, filename in quantities:

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        I_vals,
        values,
        'ko',
        markersize=4
    )

    ax.set_xlabel(r"Tonic input $I$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f"{title} against tonic input", fontsize=16)

    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()

    fig.savefig(
        BASE_DIR / f"bifurcation_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)