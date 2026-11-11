import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Bible.simulation_config import *
from scripts.Bible.neural_model import *
from scripts.Bible.run_simulation import *
from scripts.Bible.kinematic_analysis import *

from scripts.Bible.animation import *

config2 = SimulationConfig(I=0.25, T=100, max_time=200)

worm_positions, kappas, times, peak_heights, peak_times, peak_indices, A_V_vals, A_D_vals, V_V_vals, V_D_vals = example4(config2, ODEs)

norm_wave, freq, period, max_curvature, mean_velocity = kinematic_values(kappas, times, worm_positions, peak_indices, peak_times, config2)

print("Wavelength: ", norm_wave)
print("Frequency: ", freq)
print("Period: ", period)
print("Max curvature: ", max_curvature)
print("Average velocity: ", mean_velocity)

animation = create_worm_animation(worm_positions, config2)

animation.save("worm_simulation.mp4", writer="ffmpeg", fps=int(1/config2.dt))



