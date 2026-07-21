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

from scripts.Error_Testing.Methods.Hilbert_Tests.hilbert_calc import Hilbert_Transform

# Parameters
N = 120  # Number of body points - recommend ~100

N_controls = 6
T = 1.0  # Final time - recommend several undulations
dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)


def plot_curve(x, filename="_tmp.png"):
    plt.figure(1)
    plt.plot(x[0], x[2])

    plt.axis("equal")
    plt.savefig(filename)




def example2(lam):
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
    omega = 1.0

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






if __name__ == "__main__":

    lam_vals = np.arange(0.2, 1.5, 0.1)

    wavelengths = np.empty(len(lam_vals), dtype=object)
    frequencies = np.empty(len(lam_vals), dtype=object)

    t_eval = np.arange(0,T,dt)

    for i, lam in enumerate(lam_vals):
        print("lam: ", lam)
        worm_positions, kappa = example2(lam)


        maxcurv = np.max(kappa)
        wave, freq = Hilbert_Transform(kappa, N, N_controls, t_eval)

        wavelengths[i] = wave
        frequencies[i] = freq

        print("Wavelength: ", np.round(wave, 2))
        print("Frequency Hz: ", np.round(freq, 2))
        print("Max curvature: ", np.round(maxcurv, 2))



    plt.figure()
    plt.plot(lam_vals, wavelengths)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'Hilbert')
    plt.show()

    plt.figure()
    plt.plot(lam_vals, np.abs(lam_vals - wavelengths))
    plt.xlabel(r'$\lambda$')
    plt.ylabel("E")
    plt.show()