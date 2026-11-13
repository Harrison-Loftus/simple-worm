import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Fundamental.simulation_config import *
from scripts.Fundamental.neuromuscular import *
from scripts.Fundamental.mechanical import *
from scripts.Fundamental.kinematic_analysis import *

from scripts.Fundamental.animation import *
import time
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tick = time.time()
config2 = SimulationConfig(N_controls=24)

worm_positions, kappas, times, peak_heights, peak_times, peak_indices, A_V_vals, A_D_vals, V_V_vals, V_D_vals = example4(config2, ODEs)

norm_wave, freq, period, max_curvature, mean_velocity = kinematic_values(kappas, times, worm_positions, peak_indices, peak_times, config2)



print("Wavelength: ", norm_wave)
print("Frequency: ", freq)
print("Period: ", period)
print("Max curvature: ", max_curvature)
print("Average velocity: ", mean_velocity)

animation = create_worm_animation(worm_positions, config2)

animation.save("worm_simulation_neural24.mp4", writer="ffmpeg", fps=int(1/config2.dt))
tock = time.time()

print(np.round(tock-tick,2))


