import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.Johnson_Ranner.johnson_ranner_agar import *

from pathlib import Path
import time

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

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
    
    plt.figure()
    plt.imshow(curvatures[int(-10/dt):], aspect='auto', extent=[t_eval[int(-10/dt)], t_eval[-1], 0 , L], origin='lower', cmap='bwr')
    cbar = plt.colorbar(label=r'Curvature (mm$^{-1}$)')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel('Time (s)', size=12)
    plt.ylabel('Body length (mm)', size=12)
    plt.title('Kymograph of a Recovered C. elegans in Agar', size=16)
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - recovered.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    view_worm_pyqtgraph(worm_positions, dt)
    #view_curvature_pyqtgraph(curvatures, dt)