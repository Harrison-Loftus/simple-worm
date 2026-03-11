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
from simple_worm.worm import Worm
from simple_worm.util import f2n, v2f

from kymograph import *

# Parameters
N = 100  # Number of body points - recommend ~100
T = 3.0  # Final time - recommend several undulations
dt = 1.0e-3  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)


def plot_curve(x, filename="_tmp.png"):
    plt.figure(1)
    plt.plot(x[0], x[2])

    plt.axis("equal")
    plt.savefig(filename)


def example1():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    # wave parameters
    A = 10.0
    lam = 1.5
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t):
        def alpha_forcing_t(u_):
            u = u_[0]  # convert 3d coordinate to 1d
            return A * np.sin(2.0 * np.pi * lam * u - 2 * np.pi * omega * t)

        return alpha_forcing_t

    def zero_forcing(u):
        return 0.0 * u[0]

    t = 0.0

    curvatures = []

    while t < T:
        t += dt

        # solve
        ret = worm.update_solution(
            ControlsFenics(
                alpha=v2f(alpha_forcing(t), fs=worm.V),
                beta=v2f(zero_forcing, fs=worm.V),
                gamma=v2f(zero_forcing, fs=worm.Q),
            )
        )

        # output variables as 'fenics functions
        x = ret.x
        vector_curvature = ret.kappa_expr

        # other variables computed
        tangent = ret.e0
        normal = ret.e1

        # scalar curvature
        #alpha = ufl.dot(vector_curvature, normal)

        # using the variables to compute interesting quantities
        #curvature_form = fem.form(0.5 * alpha**2 * ufl.dx)
        #total_curvature = fem.assemble_scalar(curvature_form)
        #print(t, total_curvature)

        ret_np = ret.to_numpy()
        x_np = ret_np.x
        curvature_np = ret_np.alpha

        # plot_curve(x_np)
        curvatures.append(curvature_np.copy())

    curvatures_np = np.array(curvatures)
    
    return curvatures_np.T


def example2():
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
    lam = 1.5
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t, j):
        # j is point in numpy array
        # j_control is the corresponding control point
        j_control = (j * N_controls) // N
        # u_control is center point of control region
        u_control = (j_control + 0.5) / N_controls
        return A * np.sin(2.0 * np.pi * lam * u_control - 2 * np.pi * omega * t)

    t = 0.0
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
        curvature_np = ret_np.alpha

        print(f"{x_np=}")
        print(f"{curvature_np=}")
        plot_curve(x_np)


if __name__ == "__main__":
    curvatures = example1()

    heights = [i/N for i in range(N)]
    t_eval = np.arange(0, T, dt)
   

    times, kappa_peaks = finding_peaks(curvatures, t_eval, N)
    wavelength, selection_time = lin_reg_wavelength(kappa_peaks, times, N)

    print("Wavelength via kymogram: ", wavelength)
    
    plt.figure(figsize=(8,4))
    plt.imshow(curvatures.T, aspect='auto', extent=[0, T, heights[0], heights[-1]], origin='lower', cmap='bwr')
    plt.colorbar(label='curvature')
    plt.xlabel('Time')
    plt.ylabel('Body length')
    plt.show()