from numpy.linalg import norm
from dolfinx import fem
from matplotlib import pyplot as plt
import numpy as np
import ufl
import time

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
    ControlSequenceFenics,
)
from simple_worm.worm import Worm
from simple_worm.util import f2n, v2f#
from scipy.integrate import solve_ivp



import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import importlib
module = importlib.import_module("scripts.hilbert_calc")

ODE_examples = ["ODEs.carter_odes_example_1", "ODEs.carter_odes_example_2"]
Example = importlib.import_module(ODE_examples[1])


start_time = time.time()

# Parameters
T = 10.0  # Final time - recommend several undulations
dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

N = Example.n # replace with N_1 / N_2 when performing respective example
N_controls = Example.N_controls

ODE_state = np.zeros(5*N)
ODE_state[3*N] = 1.0
ODE_state[4*N] = -1.0
ODE_time = 0.0
t_eval = np.linspace(0, T, n_timesteps + 1)

s = np.linspace(0.0,1.0,N)

def plot_curve(x, filename="_tmp.png"):
    plt.figure(1)
    plt.plot(x[0], x[2])

    plt.axis("equal")
    plt.savefig(filename)

def step_ode(dt):
    global ODE_state, ODE_time, kappa_current

    sol = solve_ivp(Example.ODEs,
                    (ODE_time, ODE_time + dt),
                    ODE_state,
                    method="RK23",
                    max_step=dt)

    ODE_state = sol.y[:, -1]
    ODE_time += dt

    # extract curvature for this timestep
    kappa_current = ODE_state[0:N]



def example1():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """


    def curvature_at_u(u):
        kappa_idx = [np.argmin(np.abs(u_val - s)) for u_val in u]
        kappa = ODE_state[kappa_idx]
        return kappa

    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()


    # specific forcing function
    def alpha_forcing(t):
        def alpha_forcing_t(u_):
            u = u_[0]      # parametric coordinate along worm
            kappas = curvature_at_u(u)
            return kappas
        return alpha_forcing_t

    def zero_forcing(u):
        return 0.0 * u[0]

    worm_positions = []
    curvatures = []
    curvatures.append(ODE_state[0:N].copy())
    t = 0.0

    for step in range(n_timesteps):
        t = step * dt
        step_ode(dt)
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
        x_np_frame = x_np.T
        curvature_np = ret_np.alpha
        
        curvatures.append(curvature_np.copy())
        worm_positions.append(x_np_frame.copy())

    curvatures = np.array(curvatures)
        #plot_curve(x_np)
    return worm_positions, curvatures.T


def example2():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """


    def curvature_at_u(u):
        kappa_idx = np.argmin(np.abs(u - s)) 
        kappa = ODE_state[kappa_idx]
        return kappa

    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    # controls holder but we will only use 7 different values
    control = np.empty(N)

    # holder for other control directions
    zeroN = np.zeros(N)
    zeroNm = np.zeros(N - 1)

    # wave parameters
    

    # specific forcing function
    def alpha_forcing(t, j):
        
        # j is point in numpy array
        # j_control is the corresponding control point
        j_control = (j * N_controls) // N
        
        # u_control is center point of control region
        u_control = (j_control + 0.5) / N_controls
        a = curvature_at_u(u_control)
        
        return a

    t = 0.0
    curvatures = []
    worm_positions = []
    curvatures.append(ODE_state[0:N].copy())
    
    for step in range(n_timesteps):
        t = step * dt
        step_ode(dt)
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
        
        #print(f"{x_np=}")
        #print(f"{curvature_np=}")
        #plot_curve(x_np)
        worm_positions.append(x_np_frame.copy())
        curvatures.append(curvature_np.copy())
    curvatures = np.array(curvatures)
    worm_positions = np.array(worm_positions)
    return worm_positions, curvatures.T


