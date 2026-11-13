import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.Fundamental.simulation_config import *
from scripts.Fundamental.neuromuscular import *
from scripts.Fundamental.mechanical import *
from scripts.Fundamental.kinematic_analysis import *

from scripts.Fundamental.animation import *

OUTPUT_DIR = Path("outputs/Sweeps/Descretisation/N_vals_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_vals = [int(3*2**i) for i in range(4,10)]

n = 0

wavelengths = np.full((len(N_vals),), np.nan, dtype=float)
frequencies = np.full((len(N_vals),), np.nan, dtype=float)
periods = np.full((len(N_vals),), np.nan, dtype=float)
max_curvatures = np.full((len(N_vals),), np.nan, dtype=float)
velocities = np.full((len(N_vals),), np.nan, dtype=float)

for i, N_val in enumerate(N_vals):
    n+=1
    print(f'Simulation {n}/{len(N_vals)}: N = {N_val}')
    config = SimulationConfig(N=N_val)

    worm_positions, kappas, times, peak_heights, peak_times, peak_indices, A_V_vals, A_D_vals, V_V_vals, V_D_vals  = example4(config, ODEs)

    norm_wave, freq, period, max_curvature, mean_velocity = kinematic_values(kappas, times, worm_positions, peak_indices, peak_times, config)

    wavelengths[i] = norm_wave
    frequencies[i] = freq
    periods[i] = period
    max_curvatures[i] = max_curvature
    velocities[i] = mean_velocity

    np.savez_compressed(
    OUTPUT_DIR / f"simulation_I_{N_val:.3f}.npz",
    I=N_val,
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
        for key, value in vars(config).items()
        if value is not None
    }
    )

np.savez_compressed(
    OUTPUT_DIR / "sweep_summary.npz",
    N_vals=N_vals,
    wavelengths=wavelengths,
    frequencies=frequencies,
    periods=periods,
    max_curvatures=max_curvatures,
    mean_velocities=velocities
)
