import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.sweeps.For_paper_sweep.forward_model_sweep_nondim import *
from scripts.sweeps.For_paper_sweep.ODEs.epsilon_g import *

from scripts.hilbert_calc4 import Hilbert_Transform
from scripts.velocity_calc import Average_Velocity

from pathlib import Path

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem


if __name__ == "__main__":

    tick = time.time()

    t_eval = np.arange(0, T, dt)

    wavelengths = np.empty(len(epsilon_g_vals), dtype=object)
    frequencies = np.empty(len(epsilon_g_vals), dtype=object)
    velocities = np.empty(len(epsilon_g_vals), dtype=object)

    max_curvatures = np.empty(len(epsilon_g_vals), dtype=object)

    for i, A in enumerate(epsilon_g_vals):
        print("epsilon_g: ", A)
        print(f'{i+1} out of {len(epsilon_g_vals)}: {np.round((i)/len(epsilon_g_vals)*100, 2)}% done')
        worm_positions, curvatures = simulation(A)
        print(worm_positions.shape)

        max_kappa = np.max(curvatures)
        max_curvatures[i] = max_kappa
        print("Curvature amplitude: ", max_kappa)

        #---------Hilber Transform----------

        control_indices = np.linspace(0, N-1, N_controls).astype(int)
        kappa_reduced = curvatures[control_indices, :]

        wavelength, frequency = Hilbert_Transform(kappa_reduced, N_controls, N_controls, t_eval)

        frequency = frequency / t_c

        wavelengths[i] = wavelength
        frequencies[i] = frequency 
        print("Wavelength: ", wavelength)
        print("Frequency: " , frequency)

        #---------Worm Velocity------------
        velocity_x = Average_Velocity(worm_positions, t_eval)
        
        velocity_x = velocity_x / t_c

        print("Velocity: ", velocity_x)
        velocities[i] = velocity_x
    
    tock = time.time()

    print("--- %s seconds ---" % (np.round(tock - tick, 2)))

    plt.figure()
    plt.plot(epsilon_g_vals, wavelengths, 'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.title("Wavelength against gap-junctional strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - wavelength.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()



    plt.figure()
    plt.plot(epsilon_g_vals, frequencies, 'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.title("Frequency against gap-junctional strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - frequency.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure()
    plt.plot(epsilon_g_vals, velocities,'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel("velocity " + r'$\mathrm{mm/s}$')
    plt.title("Velocity against gap-junctional strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - velocity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    plt.figure()
    plt.plot(epsilon_g_vals, max_curvatures, 'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel("Curvature amplitude " + r'$\mathrm{mm^{-1}}$')
    plt.title("Curvature amplitude against gap-junctional strength")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - curvature.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    