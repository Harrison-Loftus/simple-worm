import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.sweeps.nondimensional.forward_model_sweep_nondim import *
from scripts.sweeps.nondimensional.ODEs.c_s import *

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

    wavelengths = np.empty(len(c_s_vals), dtype=object)
    frequencies = np.empty(len(c_s_vals), dtype=object)
    velocities = np.empty(len(c_s_vals), dtype=object)

    max_curvatures = np.empty(len(c_s_vals), dtype=object)

    for i, A in enumerate(c_s_vals):
        print("c_s: ", A)
        print(f'{i+1} out of {len(c_s_vals)}: {np.round((i)/len(c_s_vals)*100, 2)}% done')
        worm_positions, curvatures = simulation(A)
        print(worm_positions.shape)

        max_kappa = np.max(curvatures)
        max_curvatures[i] = max_kappa
        print("Curvature amplitude: ", max_kappa)

        #---------Hilber Transform----------
        wavelength, frequency = Hilbert_Transform(curvatures, N, N_controls, t_eval)

        frequency = frequency / t_c

        wavelengths[i] = wavelength
        frequencies[i] = frequency 
        print("Wavelength: ", wavelength)
        print("Frequency: ", frequency)

        #---------Worm Velocity------------
        velocity_x = Average_Velocity(worm_positions, t_eval)
        
        velocity_x = velocity_x / t_c

        print("Velocity: ", velocity_x)
        velocities[i] = velocity_x
    
    tock = time.time()

    print("--- %s seconds ---" % (np.round(tock - tick, 2)))

    plt.figure()
    plt.plot(c_s_vals, wavelengths, 'ko')
    plt.xlabel("Scale of nonlinear threshold " + r'$c_s$')
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.title("Wavelength against Scale of nonlinear threshold")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - wavelength.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()



    plt.figure()
    plt.plot(c_s_vals, frequencies, 'ko')
    plt.xlabel("Scale of nonlinear threshold " + r'$c_s$')
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.title("Frequency against Scale of nonlinear threshold")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - frequency.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    plt.figure()
    plt.plot(c_s_vals, velocities,'ko')
    plt.xlabel("Scale of nonlinear threshold " + r'$c_s$')
    plt.ylabel("velocity " + r'$\mathrm{mm/s}$')
    plt.title("Velocity against Scale of nonlinear threshold")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - velocity.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


       
    plt.figure()
    plt.plot(c_s_vals, max_curvatures, 'ko')
    plt.xlabel("Scale of nonlinear threshold " + r'$c_s$')
    plt.ylabel("Curvature amplitude " + r'$\mathrm{mm^{-1}}$')
    plt.title("Curvature amplitude against Scale of nonlinear threshold")
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - curvature.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()