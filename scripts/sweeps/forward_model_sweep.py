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

from scipy.integrate import solve_ivp
from simple_worm.worm import Worm
from simple_worm.util import f2n, v2f#


import importlib


# BEFORE CONDUCTING PARAMETER SWEEP CHANGE INDEX
Parameter_Sweeps = ["ODEs.amplitude", "ODEs.epsilon_g", "ODEs.epsilon_p", "scripts.sweeps.ODEs.mu_f", 
                    "ODEs.range", "ODEs.tau_m", "ODEs.tau_n"]

module_name = Parameter_Sweeps[3]
module = importlib.import_module(module_name)


N = module.N
N_controls = module.N_controls

# Parameters
T = 5.0  # Final time - recommend several undulations
dt = 1.0e-2 # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)



s = np.linspace(0.0,1.0,N)

def simulation(Argument):

    ODE_state = np.zeros(5*N)
    ODE_state[3*N] = 1.0
    ODE_state[4*N] = -1.0
    ODE_time = 0.0
    kappa_current = np.zeros(N)

    def plot_curve(x, filename="_tmp.png"):
        plt.figure(1)
        plt.plot(x[0], x[2])

        plt.axis("equal")
        plt.savefig(filename)

    def step_ode(dt):
        nonlocal ODE_state, ODE_time, kappa_current

        sol = solve_ivp(module.ODEs,
                        (ODE_time, ODE_time + dt),
                        ODE_state,
                        method="RK45",
                        max_step=dt,args=(Argument,))

        ODE_state = sol.y[:, -1]
        ODE_time += dt

        # extract curvature for this timestep
        kappa_current = ODE_state[0:N]


    
    def curvature_at_u(u):
        kappa_idx = np.argmin(np.abs(u - s))
        return kappa_current[kappa_idx]



        """
        This example shows how to call the simulator with a fenics function
        for forcing.
        """
        
        # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    # controls holder but we will only use 7 different values
    control = np.empty(N)

    # holder for other control directions
    zeroN = np.zeros(N)
    zeroNm = np.zeros(N - 1)

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
    kappas = []
    worm_positions = []
    while t < T:
        t += dt
        step_ode(dt)
        print(np.round(t,2))
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
        kappas.append(curvature_np.copy())
    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T
