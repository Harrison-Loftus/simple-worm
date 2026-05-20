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

from scipy.integrate import solve_ivp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.hilbert_calc import Hilbert_Transform
from scripts.velocity_calc import Average_Velocity

import time

import importlib


# BEFORE CONDUCTING PARAMETER SWEEP CHANGE INDEX
Parameter_Sweeps = ["scripts.sweeps.nondimensional.ODEs.amplitude", "scripts.sweeps.nondimensional.ODEs.a_0", 
                    "scripts.sweeps.nondimensional.ODEs.c_m", "scripts.sweeps.nondimensional.ODEs.c_s",
                    "scripts.sweeps.nondimensional.ODEs.epsilon_g", 
                    "scripts.sweeps.nondimensional.ODEs.epsilon_p",  
                    "scripts.sweeps.nondimensional.ODEs.range", "scripts.sweeps.nondimensional.ODEs.tau_m", 
                    "scripts.sweeps.nondimensional.ODEs.tau_n"]

module_name = Parameter_Sweeps[1]
module = importlib.import_module(module_name)


N = module.N
N_controls = module.N_controls



T = 5.0 / module.t_c # Final time - recommend several undulations
dt = 1.0e-2 / module.t_c # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

s = np.linspace(0.0,1.0,N)

def simulation(Argument):

    # Parameters
    

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
                        max_step=dt, args=(Argument,))

        ODE_state = sol.y[:, -1]
        ODE_time += dt

        # extract curvature for this timestep
        kappa_current = ODE_state[0:N]

    # Note: example 3 and 4 use nondimensionalised material parameters. Here we are importing Carter_ODES which has dimensionalised parameters.
    # A new set of ODEs will need to be made for examples 3 and 4. This is just a sense check of implementation for now.



    
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """


    def curvature_at_u(u):
        kappa_idx = np.argmin(np.abs(u - s)) 
        kappa = ODE_state[kappa_idx]
        return kappa

    worm = Worm(N, dt)

    # set material parameters
    MP = MaterialParameters(
        K=module.K_water,  # ratio of drag coefficients
        K_rot=1.0,  # rotational drag coefficient
        A=module.e,  # bending rigidity
        B=module.eta_tilde,  # bending viscosity
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


