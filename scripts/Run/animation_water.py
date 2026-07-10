import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Johnson_Ranner.johnson_ranner_water import *

from pathlib import Path
import time

if __name__ == "__main__":
    tick = time.time()

    worm_positions, curvatures = example4()

    print(worm_positions.shape)
    f_worm = worm_positions[-1]
    print(f_worm.shape)

    maxcurv = np.max(curvatures)
    wave, freq = Hilbert_Transform(curvatures, N, N_controls, t_eval)

    print("Wavelength: ", np.round(wave, 2))
    print("Frequency Hz: ", np.round(freq, 2))
    print("Max curvature: ", np.round(maxcurv, 2))
   
    tock = time.time()
    print("time: ", np.round(tock - tick, 2))

    view_worm_pyqtgraph(worm_positions, dt)
    #view_curvature_pyqtgraph(curvatures, dt)