import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sweeps.forward_model_sweep import *
from scripts.sweeps.ODEs.epsilon_p import *

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

    wavelengths = np.empty(len(epsilon_p_vals), dtype=object)
    frequencies = np.empty(len(epsilon_p_vals), dtype=object)
    velocities = np.empty(len(epsilon_p_vals), dtype=object)

    max_curvatures = np.empty(len(epsilon_p_vals), dtype=object)

    for i, epsilon_p in enumerate(epsilon_p_vals):
        print(f'{i+1} out of {len(epsilon_p_vals)}: {np.round((i)/len(epsilon_p_vals)*100, 2)}% done')
        worm_positions, curvatures = simulation(epsilon_p)
        print(worm_positions.shape)

        max_kappa = np.max(curvatures)
        max_curvatures[i] = max_kappa 
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
    plt.plot(epsilon_p_vals, wavelengths, 'ko')
    plt.xlabel("Proprioceptive strength " + r'$\varepsilon_p$')
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.title("Wavelength against proprioceptive strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - wavelength.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()



    plt.figure()
    plt.plot(epsilon_p_vals, frequencies, 'ko')
    plt.xlabel("Proprioceptive strength " + r'$\varepsilon_p$')
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.title("Frequency against proprioceptive strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - frequency.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    plt.figure()
    plt.plot(epsilon_p_vals, velocities,'ko')
    plt.xlabel("Proprioceptive strength " + r'$\varepsilon_p$')
    plt.ylabel("velocity " + r'$\mathrm{mm/s}$')
    plt.title("Velocity against proprioceptive strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - velocity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


       
    plt.figure()
    plt.plot(epsilon_p_vals, max_curvatures, 'ko')
    plt.xlabel("Proprioceptive strength " + r'$\varepsilon_p$')
    plt.ylabel("Curvature amplitude " + r'$\mathrm{mm^{-1}}$')
    plt.title("Curvature amplitude against proprioceptive strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - curvature.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    
    