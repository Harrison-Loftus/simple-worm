import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Bible.simulation_config import *
from scripts.Bible.neural_model import *
from scripts.Bible.run_simulation import *
from scripts.Bible.kinematic_analysis import *

from scripts.Bible.animation import *

OUTPUT_DIR = Path("outputs/FOR_PAPER/Bifurcation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

I_vals = np.arange(0.2, 0.45, 0.05)

n = 0

wavelengths = np.full((len(I_vals),), np.nan, dtype=float)
frequencies = np.full((len(I_vals),), np.nan, dtype=float)
periods = np.full((len(I_vals),), np.nan, dtype=float)
max_curvatures = np.full((len(I_vals),), np.nan, dtype=float)
velocities = np.full((len(I_vals),), np.nan, dtype=float)

for i, I_val in enumerate(I_vals):
    n+=1
    print(f'Simulation {n}/{len(I_vals)}: I = {I_val:.3f}')
    config2 = SimulationConfig(I=I_val, T=100, max_time=200)

    worm_positions, kappas, times, peak_heights, peak_times, peak_indices, A_V_vals, A_D_vals, V_V_vals, V_D_vals  = example4(config2, ODEs)

    norm_wave, freq, period, max_curvature, mean_velocity = kinematic_values(kappas, times, worm_positions, peak_indices, peak_times, config2)

    wavelengths[i] = norm_wave
    frequencies[i] = freq
    periods[i] = period
    max_curvatures[i] = max_curvature
    velocities[i] = mean_velocity

    np.savez_compressed(
    OUTPUT_DIR / f"simulation_I_{I_val:.3f}.npz",
    I=I_val,
    worm_positions=worm_positions,
    kappas=kappas,
    times=times,
    peak_heights=peak_heights,
    peak_times=peak_times,
    peak_indices=peak_indices,
    A_V=A_V_vals,
    A_D=A_D_vals,
    V_V=V_V_vals,
    V_D=V_D_vals,
    wavelength=norm_wave,
    frequency=freq,
    period=period,
    max_curvature=max_curvature,
    mean_velocity=mean_velocity,
    **{
        f"config_{key}": value
        for key, value in vars(config2).items()
        if value is not None
    }
    )

np.savez_compressed(
    OUTPUT_DIR / "sweep_summary.npz",
    I_vals=I_vals,
    wavelengths=wavelengths,
    frequencies=frequencies,
    periods=periods,
    max_curvatures=max_curvatures,
    mean_velocities=velocities
)
