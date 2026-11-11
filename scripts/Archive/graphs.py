
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

PAPER_DIR = Path("outputs/FOR_PAPER/Viscosity")

ORIGINAL_SCRIPT = "Viscosity_sweep_Denham"  

wavelengths = np.load(
    PAPER_DIR / f"wavelengths_{ORIGINAL_SCRIPT}.npy"
)

frequencies = np.load(
    PAPER_DIR / f"frequencies_{ORIGINAL_SCRIPT}.npy"
)

K_vals = [1.57, 7.5, 12, 18, 30, 45]
epsilon_p_vals = np.linspace(0.0, 1.0, 11)

markers = {
    1.57: "x",
    7.5: "o",
    12: "s",
    18: "D",
    30: "^",
    45: "*"
}

fig, ax = plt.subplots(figsize=(8, 6))

cmap = plt.get_cmap("PuRd")
norm = plt.Normalize(
    vmin=np.min(epsilon_p_vals),
    vmax=np.max(epsilon_p_vals)
)

for i, K_val in enumerate(K_vals):
    if K_val == 1.57:
        continue
    for j, eps in enumerate(epsilon_p_vals):
        if eps < 0.3:
            continue

        freq = frequencies[i, j]
        wave = wavelengths[i, j]

        if not (np.isfinite(freq) and np.isfinite(wave)):
            continue

        ax.scatter(
            freq,
            wave,
            marker=markers[K_val],
            c=[eps],
            cmap=cmap,
            norm=norm,
            s=65
        )

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"Proprioceptive strength $\varepsilon_p$", fontsize=13)

# Legend
legend_elements = [
    Line2D(
        [0], [0],
        marker=markers[K],
        color="black",
        linestyle="None",
        markersize=8,
        label=fr"$K = {K}$"
    )
    for K in K_vals if K!= 1.57
]

ax.legend(
    handles=legend_elements,
    title="Drag ratio",
    loc="best"
)

# Labels
ax.set_xlabel("Frequency (Hz)", fontsize=14)
ax.set_ylabel("Wavelength (mm)", fontsize=14)

# NEW TITLE
ax.set_title(
    "Effect of Proprioceptive Strength on\n"
    "Simulated C. elegans Forward Locomotion",
    fontsize=15,
    pad=15
)

ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()

fig.savefig(
    PAPER_DIR / "strength_modulation_titled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()








#--------range plots---------
ORIGINAL_SCRIPT = "Viscosity_sweep_Denham_range"  


wavelengths = np.load(
    PAPER_DIR / f"wavelengths_{ORIGINAL_SCRIPT}.npy"
)

frequencies = np.load(
    PAPER_DIR / f"frequencies_{ORIGINAL_SCRIPT}.npy"
)

K_vals = [1.57, 7.5, 12, 18, 30, 45]
range_vals = np.linspace(0.0, 1.0, 11)

markers = {
    1.57: "x",
    7.5: "o",
    12: "s",
    18: "D",
    30: "^",
    45: "*"
}

fig, ax = plt.subplots(figsize=(8, 6))

cmap = plt.get_cmap("Greens")
norm = plt.Normalize(
    vmin=np.min(range_vals),
    vmax=np.max(range_vals)
)

for i, K_val in enumerate(K_vals):
    if K_val == 1.57:
        continue
    for j, range_val in enumerate(range_vals):
        if range_val < 0.3:
            continue

        freq = frequencies[i, j]
        wave = wavelengths[i, j]

        if not (np.isfinite(freq) and np.isfinite(wave)):
            continue

        ax.scatter(
            freq,
            wave,
            marker=markers[K_val],
            c=[range_val],
            cmap=cmap,
            norm=norm,
            s=65
        )

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"Proprioceptive Range", fontsize=13)

# Legend
legend_elements = [
    Line2D(
        [0], [0],
        marker=markers[K],
        color="black",
        linestyle="None",
        markersize=8,
        label=fr"$K = {K}$"
    )
    for K in K_vals if K!= 1.57
]

ax.legend(
    handles=legend_elements,
    title="Drag ratio",
    loc="best"
)

# Labels
ax.set_xlabel("Frequency (Hz)", fontsize=14)
ax.set_ylabel("Wavelength (mm)", fontsize=14)

# NEW TITLE
ax.set_title(
    "Effect of Proprioceptive Range on\n"
    "Simulated C. elegans Forward Locomotion",
    fontsize=15,
    pad=15
)

ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()

fig.savefig(
    PAPER_DIR / "range_modulation_titled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()





