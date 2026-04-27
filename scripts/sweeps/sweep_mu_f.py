from scripts.sweeps.forward_model_sweep import *
from ODEs.mu_f import *

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

    wavelengths = np.empty(len(C_N_vals), dtype=object)
    frequencies = np.empty(len(C_N_vals), dtype=object)
    velocities = np.empty(len(C_N_vals), dtype=object)

    max_curvatures = np.empty(len(C_N_vals), dtype=object) 

    for i, C_N in enumerate(C_N_vals):
        print(f'{i+1} out of {len(C_N_vals)}: {np.round((i)/len(C_N_vals)*100, 2)}% done')
        worm_positions, curvatures = simulation(C_N)
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
    plt.plot(mu_f_mPas, wavelengths, 'ko')
    plt.xlabel("External fluid viscosity " + r'$\mathrm{mPa \cdot s}$')
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.xscale('log')
    plt.title("Wavelength against external fluid viscosity")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - wavelength.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    plt.figure()
    plt.plot(mu_f_mPas, frequencies, 'ko')
    plt.xlabel("External fluid viscosity " + r'$\mathrm{mPa \cdot s}$')
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.xscale('log')
    plt.title("Frequency against external fluid viscosity")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - frequency.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure()
    plt.plot(mu_f_mPas, velocities,'ko')
    plt.xlabel("External fluid viscosity " + r'$\mathrm{mPa \cdot s}$')
    plt.ylabel("Velocity " + r'$\mathrm{mm/s}$')
    plt.xscale('log')
    plt.title("Velocity against external fluid viscosity")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - velocity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    

    plt.figure()
    plt.plot(mu_f_mPas, max_curvatures, 'ko')
    plt.xlabel("External fluid viscosity " + r'$\mathrm{mPa \cdot s}$')
    plt.ylabel("Curvature amplitude " + r'$\mathrm{mm^{-1}}$')
    plt.xscale('log')
    plt.title("Curvature amplitude against external fluid viscosity")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - curvature.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()