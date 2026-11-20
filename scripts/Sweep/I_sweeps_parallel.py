import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Fundamental.simulation_config import *
from scripts.Fundamental.neuromuscular import *
from scripts.Fundamental.mechanical import *
from scripts.Fundamental.kinematic_analysis import *

from scripts.Fundamental.animation import *

OUTPUT_DIR = Path("outputs/FOR_PAPER/Parallel/Bifurcation/non_osc_fine")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

I_vals = np.arange(0.25, 0.325, 0.0025)

wavelengths = np.full((len(I_vals),), np.nan, dtype=float)
frequencies = np.full((len(I_vals),), np.nan, dtype=float)
periods = np.full((len(I_vals),), np.nan, dtype=float)
max_curvatures = np.full((len(I_vals),), np.nan, dtype=float)
velocities = np.full((len(I_vals),), np.nan, dtype=float)

def run_one(I_val):
    print(f"Simulation: I = {I_val:.5f}", flush=True)
    config2 = SimulationConfig(I=I_val, T=200, max_time=350)

    worm_positions, kappas, times, peak_heights, peak_times, peak_indices, A_V_vals, A_D_vals, V_V_vals, V_D_vals  = example4(config2, ODEs)

    norm_wave, freq, period, max_curvature, mean_velocity = kinematic_values(kappas, times, worm_positions, peak_indices, peak_times, config2)

    np.savez_compressed(
    OUTPUT_DIR / f"simulation_I_{I_val:.5f}.npz",
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

    return norm_wave, freq, period, max_curvature, mean_velocity


if __name__ == "__main__":
    # one process per I value; spawn so each worker starts a clean MPI/dolfinx
    n_workers = min(len(I_vals), os.cpu_count(), 8)
    with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as ex:
        results = list(ex.map(run_one, I_vals))

    for i, r in enumerate(results):
        wavelengths[i], frequencies[i], periods[i], max_curvatures[i], velocities[i] = r

    np.savez_compressed(
        OUTPUT_DIR / "sweep_summary.npz",
        I_vals=I_vals,
        wavelengths=wavelengths,
        frequencies=frequencies,
        periods=periods,
        max_curvatures=max_curvatures,
        mean_velocities=velocities
    )