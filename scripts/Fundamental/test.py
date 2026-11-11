import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Fundamental.simulation_config import *
from scripts.Fundamental.neural_model import *
from scripts.Fundamental.run_simulation import *
from scripts.Fundamental.kinematic_analysis import *

from scripts.Fundamental.animation import *

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

I_val = 0.25

config2 = SimulationConfig(I=I_val, T=500, max_time=1000)

worm_positions, kappas, times, peak_heights, peak_times, peak_indices, A_V_vals, A_D_vals, V_V_vals, V_D_vals = example4(config2, ODEs)

norm_wave, freq, period, max_curvature, mean_velocity = kinematic_values(kappas, times, worm_positions, peak_indices, peak_times, config2)

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


print("Wavelength: ", norm_wave)
print("Frequency: ", freq)
print("Period: ", period)
print("Max curvature: ", max_curvature)
print("Average velocity: ", mean_velocity)

animation = create_worm_animation(worm_positions, config2)

animation.save("worm_simulation.mp4", writer="ffmpeg", fps=int(1/config2.dt))



