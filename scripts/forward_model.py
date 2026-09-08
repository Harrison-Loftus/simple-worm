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

from velocity_calc import Average_Velocity
from hilbert_calc import Hilbert_Transform
from simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph

from kymograph import *
from parameters import *
from matplotlib_animate_worm import *

from pathlib import Path

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

import time

# Parameters

T = 30.0  # Final time - recommend several undulations
dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

def plot_curve(x, filename="_tmp.png"):
    plt.figure(1)
    plt.plot(x[0], x[2])

    plt.axis("equal")
    plt.savefig(filename)


# standard simple-worm examples with imposed travelling wave

def example1():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """

    global N, N_controls
    N = 120
    N_controls = 120
    
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    # wave parameters
    A = 10.0
    lam = 0.66
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t):
        def alpha_forcing_t(u_):
            u = u_[0]  # convert 3d coordinate to 1d
            return A * np.sin(2.0 * np.pi / lam * u - 2 * np.pi * omega * t)

        return alpha_forcing_t

    def zero_forcing(u):
        return 0.0 * u[0]

    t = 0.0
    kappas = []
    worm_positions = []
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


def example2():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
    global N, N_controls
    N = 120
    N_controls = 6
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    # controls holder but we will only use 7 different values
    N
    control = np.empty(N)

    # holder for other control directions
    zeroN = np.zeros(N)
    zeroNm = np.zeros(N - 1)

    # wave parameters
    A = 10.0
    lam = 1.0
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
        print(np.round(t, 2))
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


def example3():
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """
    # holders for 'worm', u and control
    worm = Worm(N, dt)

    # set material parameters
    MP = MaterialParameters(
        K=10.0,  # ratio of drag coefficients
        K_rot=1.0,  # rotational drag coefficient
        A=10.0,  # bending rigidity
        B=0.1,  # bending viscosity
        C=1.0,  # twisting rigidity
        D=0.1,  # twisting viscosity
    )
    worm.initialise(MP)

    # wave parameters
    Amp = 10.0
    lam = 0.66
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t):
        def alpha_forcing_t(u_):
            u = u_[0]  # convert 3d coordinate to 1d
            return Amp * np.sin(2.0 * np.pi / lam * u - 2 * np.pi * omega * t)

        return alpha_forcing_t

    def zero_forcing(u):
        return 0.0 * u[0]

    t = 0.0
    kappas = []
    worm_positions = []
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


def example4():
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """
    # holders for 'worm', u and control
    global N
    N = 96
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
    omega = 0.1

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
        print(np.round(t, 2))
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

    tick = time.time()

    t_eval = np.arange(0,T,dt)

    worm_positions, curvatures = example4()
    print(worm_positions.shape)

    maxcurv = np.max(curvatures)
    velocity = Average_Velocity(worm_positions, t_eval)
    wave, freq = Hilbert_Transform(curvatures, N, N_controls, t_eval)

    print("Wavelength: ", np.round(wave, 2))
    print("Frequency Hz: ", np.round(freq, 2))
    print("Max curvature: ", np.round(maxcurv, 2))
    print("Average velocity mm/s: ", np.round(velocity, 2))

    wavelength_k, freq_k = lin_reg_wavelength(curvatures, t_eval, N, N_controls)

    print("kymogram wavelength ", wavelength_k)
    print("kymo freq", freq_k)
    tock = time.time()

    print("time: ", np.round(tock - tick, 2))

    anim = create_worm_animation(worm_positions, dt)
    anim.save(OUTPUT_DIR / f"{SCRIPT_NAME} - worm.mp4", writer="ffmpeg", fps = int(round(1/dt)), dpi=150)

    #view_worm_pyqtgraph(worm_positions, dt)