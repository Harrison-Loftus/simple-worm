import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sweeps.forward_model_sweep import *
from scripts.sweeps.ODEs.amplitude import *

from scripts.hilbert_calc import Hilbert_Transform
from scripts.velocity_calc import Average_Velocity

from pathlib import Path

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem


if __name__ == "__main__":

    tick = time.time()

    t_eval = np.arange(0, T, dt)

    wavelengths = np.empty(len(amplitude_vals), dtype=object)
    frequencies = np.empty(len(amplitude_vals), dtype=object)
    velocities = np.empty(len(amplitude_vals), dtype=object)

    max_curvatures = np.empty(len(amplitude_vals), dtype=object)

    for i, A in enumerate(amplitude_vals):
        print(f'{i+1} out of {len(amplitude_vals)}: {np.round((i)/len(amplitude_vals)*100, 2)}% done')
        worm_positions, curvatures = simulation(A)
        print(worm_positions.shape)

        max_kappa = np.max(curvatures)
        max_curvatures[i] = max_kappa
        print("Curvature amplitude: ", max_kappa)

        #---------Hilber Transform----------
        wavelength, frequency = Hilbert_Transform(curvatures, N, N_controls, t_eval)
        wavelengths[i] = wavelength
        frequencies[i] = frequency
        print("Wavelength: ", wavelength)
        print("Frequency: " , frequency)

        #---------Worm Velocity------------
        velocity_x = Average_Velocity(worm_positions, t_eval)
        print("Velocity: ", velocity_x)
        velocities[i] = velocity_x
    
    tock = time.time()

    print("--- %s seconds ---" % (np.round(tock - tick, 2)))

    plt.figure()
    plt.plot(amplitude_vals, wavelengths, 'ko')
    plt.xlabel("Preferred curvature scaling")
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.title("Wavelength against preferred curvature scaling")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - wavelength.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()



    plt.figure()
    plt.plot(amplitude_vals, frequencies, 'ko')
    plt.xlabel("Preferred curvature scaling")
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.title("Frequency against preferred curvature scaling")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - frequency.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure()
    plt.plot(amplitude_vals, velocities,'ko')
    plt.xlabel("Preferred curvature scaling")
    plt.ylabel("Velocity " + r'$\mathrm{mm/s}$')
    plt.title("Velocity against preferred curvature scaling")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - velocity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    plt.figure()
    plt.plot(amplitude_vals, max_curvatures, 'ko')
    plt.xlabel("Preferred curvature scaling")
    plt.ylabel("Curvature amplitude " + r'$\mathrm{mm^{-1}}$')
    plt.title("Curvature amplitude against preferred cuvature scaling")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - curvature.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

