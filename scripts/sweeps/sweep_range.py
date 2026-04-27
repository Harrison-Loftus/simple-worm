from scripts.sweeps.forward_model_sweep import *
from ODEs.range import *

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

    wavelengths = np.empty(len(range_vals), dtype=object)
    frequencies = np.empty(len(range_vals), dtype=object)
    velocities = np.empty(len(range_vals), dtype=object)

    max_curvatures = np.empty(len(range_vals), dtype=object)

    for i, W_p in enumerate(W_p_vals):
        print(f'{i+1} out of {len(range_vals)}: {np.round((i)/len(range_vals)*100, 2)}% done')
        worm_positions, curvatures = simulation(W_p)
        print(worm_positions.shape)

        max_curvatures[i] = np.max(curvatures)

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
    plt.plot(range_vals, wavelengths, 'ko')
    plt.xlabel("Proprioceptive range")
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.title("Wavelength against proprioceptive range")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - wavelength.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    plt.figure()
    plt.plot(range_vals, frequencies, 'ko')
    plt.xlabel("Proprioceptive range")
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.title("Frequency against proprioceptive range")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - frequency.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure()
    plt.plot(range_vals, velocities,'ko')
    plt.xlabel("Proprioceptive range")
    plt.ylabel("Velocity " + r'$\mathrm{mm/s}$')
    plt.title("Velocity against proprioceptive range")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - velocity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    plt.figure()
    plt.plot(range_vals, max_curvatures, 'ko')
    plt.xlabel("Proprioceptive range")
    plt.ylabel("Curvature amplitude " + r'$\mathrm{mm^{-1}}$')
    plt.title("Curvature amplitude against proprioceptive range")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - curvature.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()