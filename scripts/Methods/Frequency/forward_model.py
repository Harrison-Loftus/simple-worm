from numpy.linalg import norm
from dolfinx import fem
from matplotlib import pyplot as plt
import numpy as np
import ufl

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
    ControlSequenceFenics,
)
from simple_worm.material_parameters import MaterialParameters, MaterialParametersFenics
from simple_worm.worm import Worm
from simple_worm.util import f2n, v2f

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.Methods.Hilbert_Tests.hilbert_calc import Hilbert_Transform

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

# Parameters
N = 120  # Number of body points - recommend ~100

N_controls = 6
T = 30.0  # Final time - recommend several undulations
dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

L = 1.0
E = 0.1 # Youngs modulus N/mm^2
I_c = 2.0e-7 # second moment of cuticle area mm^4
r_c = 0.5e-3  # cuticle width mm
eta = 0.05 # viscosity of the cuticle N·s/mm^2

C_N = 5.2e-9 # Normal drag coefficient in water N·s/mm²
C_T = 3.3e-9 # Tangential drag coefficient in water N·s/mm²

C_N_agar = 128e-6 # Normal drag coefficient in agar N·s/mm²
C_T_agar = 3.2e-6 # Tangential drag coefficient in agar N·s/mm²

K_water = C_N / C_T
K_agar = C_N_agar / C_T_agar

t_c = 1.0

e = (E * I_c * t_c)/(L**4 * C_T_agar) / 2
eta_tilde = (eta * I_c)/(L**4 * C_T_agar) / 2


def plot_curve(x, filename="_tmp.png"):
    plt.figure(1)
    plt.plot(x[0], x[2])

    plt.axis("equal")
    plt.savefig(filename)



def example2(omega):
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
    N = 120
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    # controls holder but we will only use 7 different values
    N_controls = 6
    control = np.empty(N)

    # holder for other control directions
    zeroN = np.zeros(N)
    zeroNm = np.zeros(N - 1)

    # wave parameters
    A = 10.0
    lam = 0.66
    

    # specific forcing function
    def alpha_forcing(t, j):
        
        # j is point in numpy array
        # j_control is the corresponding control point
        j_control = (j * N_controls) // N
        
        # u_control is center point of control region
        u_control = (j_control + 0.5) / N_controls
        
        return A * np.sin(2.0 * np.pi / lam * u_control - 2 * np.pi * omega * t)

    t = 0.0
    kappas = []
    worm_positions = []
    while t < T:
        t += dt

        # update control
        control[:] = [alpha_forcing(t, j) for j in range(N)]

        # solve
        C = ControlsNumpy(alpha=control, beta=zeroN, gamma=zeroNm)
        ret = worm.update_solution(C.to_fenics(worm))

        # output variables as 'fenics functions
        # x = ret.x
        # curvature = ret.alpha

        ret_np = ret.to_numpy()
        x_np = ret_np.x
        x_np_frame = x_np.T

        curvature_np = ret_np.alpha
        
        worm_positions.append(x_np_frame.copy())
        kappas.append(curvature_np.copy())
        #plot_curve(x_np)

    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T


def example4(omega):
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """
    # holders for 'worm', u and control
    global N
    N = 120
    # holders for 'worm', u and control
    worm = Worm(N, dt)

    # set material parameters
    MP = MaterialParameters(
        K=2.0,  # ratio of drag coefficients
        K_rot=1.0,  # rotational drag coefficient
        A=10.0,  # bending rigidity
        B=0.1,  # bending viscosity
        C=1.0,  # twisting rigidity
        D=0.1,  # twisting viscosity
    )
    worm.initialise(MP)

    global N_controls
    N_controls = 6
    control = np.empty(N)
    
    # holder for other control directions
    zeroN = np.zeros(N)
    zeroNm = np.zeros(N - 1)
    
    # wave parameters
    Amp = 10.0
    lam = 0.66

    # specific forcing function
    def alpha_forcing(t, j):
        
        # j is point in numpy array
        # j_control is the corresponding control point
        j_control = (j * N_controls) // N
        
        # u_control is center point of control region
        u_control = (j_control + 0.5) / N_controls
        
        return Amp * np.sin(2.0 * np.pi / lam * u_control - 2 * np.pi * omega * t)


    t = 0.0
    kappas = []
    worm_positions = []
    while t < T:
        t += dt
        # update control
        control[:] = [alpha_forcing(t, j) for j in range(N)]

        # solve
        Cntrl = ControlsNumpy(alpha=control, beta=zeroN, gamma=zeroNm)
        ret = worm.update_solution(Cntrl.to_fenics(worm))

        # output variables as 'fenics functions
        x = ret.x
        vector_curvature = ret.kappa_expr

        # other variables computed
        tangent = ret.e0
        normal = ret.e1

        # scalar curvature
        alpha = ufl.dot(vector_curvature, normal)

        # using the variables to compute interesting quantities
        curvature_form = fem.form(0.5 * alpha**2 * ufl.dx)
        total_curvature = fem.assemble_scalar(curvature_form)
        #print(t, total_curvature)

        ret_np = ret.to_numpy()
        x_np = ret_np.x
        x_np_frame = x_np.T
        
        curvature_np = ret_np.alpha

        worm_positions.append(x_np_frame.copy())
        kappas.append(curvature_np.copy())
        #plot_curve(x_np)
    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T





if __name__ == "__main__":

    omega_vals = np.arange(0.1, 5, 0.1)

    wavelengths = np.empty(len(omega_vals), dtype=object)
    frequencies = np.empty(len(omega_vals), dtype=object)

    t_eval = np.arange(0,T,dt)

    for i, omega in enumerate(omega_vals):
        print("omega: ", omega)
        worm_positions, kappa = example2(omega)


        maxcurv = np.max(kappa)
        wave, freq = Hilbert_Transform(kappa, N, N_controls, t_eval)

        wavelengths[i] = wave
        frequencies[i] = freq

        print("Wavelength: ", np.round(wave, 2))
        print("Frequency Hz: ", np.round(freq, 2))
        print("Max curvature: ", np.round(maxcurv, 2))



    plt.figure()
    plt.plot(omega_vals, frequencies)
    plt.xlabel(r'Imposed frequency', size=12)
    plt.ylabel(r'Computed frequency', size=12)
    plt.title(r'Computed frequency against imposed frequency', size=16)
    plt.savefig(
              OUTPUT_DIR / f"{SCRIPT_NAME} - frequency comparison.png",
              dpi=300,
              bbox_inches="tight"
      )
    plt.close()
    
    plt.figure()
    plt.plot(omega_vals, np.abs(omega_vals - frequencies))
    plt.xlabel(r'Imposed frequency', size=12)
    plt.ylabel("Absolute error", size=12)
    plt.title(r'Absolute error of computed frequency', size=16)
    plt.savefig(
            OUTPUT_DIR / f"{SCRIPT_NAME} - frequency error.png",
            dpi=300,
            bbox_inches="tight"
    )
    plt.close()