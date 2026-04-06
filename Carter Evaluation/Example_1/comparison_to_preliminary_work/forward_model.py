import matplotlib
matplotlib.use("TkAgg")   

from numpy.linalg import norm
from dolfinx import fem
from matplotlib import pyplot as plt
import matplotlib.animation as animate
import numpy as np
import ufl

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
    ControlSequenceFenics,
)
from simple_worm.worm import Worm
from simple_worm.util import f2n, v2f

from Animate_Worm import *
from Animate_curvature import *
from Carter_ODEs import *
from kymograph import *
from hilbert_calc import *

# Parameters
T = 10.0  # Final time - recommend several undulations
dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

ODE_state = np.zeros(5*N)
ODE_state[3*N] = 1.0
ODE_state[4*N] = -1.0
ODE_time = 0.0
t_eval = t_eval = np.linspace(0, T, n_timesteps + 1)

s = np.linspace(0.0,1.0,N)

def plot_curve(x, filename="_tmp.png"):
    plt.figure(1)
    plt.plot(x[0], x[2])

    plt.axis("equal")
    plt.savefig(filename)

def step_ode(dt):
    global ODE_state, ODE_time, kappa_current

    sol = solve_ivp(ODEs,
                    (ODE_time, ODE_time + dt),
                    ODE_state,
                    method="RK23",
                    max_step=dt)

    ODE_state = sol.y[:, -1]
    ODE_time += dt

    # extract curvature for this timestep
    kappa_current = ODE_state[0:N]

def curvature_at_u(u):
    kappa_idx = [np.argmin(np.abs(u_val - s)) for u_val in u]
    kappa = ODE_state[kappa_idx]
    return kappa

def example1():
    """
    This example shows how to call the simulator with a fenics function
    for forcing.
    """
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
    return worm_positions, curvatures




if __name__ == "__main__":
    worm_positions, curvatures = example1()
    curvatures = curvatures.T

    state = np.zeros(5*N)
    state[3*N] = 1.0
    state[4*N] = -1.0
    sols = []

    sol = solve_ivp(ODEs, [0, T], state, t_eval=t_eval, method='RK45', atol=1e-6, rtol=1e-9, max_step=dt)
    sols.append(sol)

    kappa = sol.y[0:N, :]

    #--------wavelength comparison--------------
    wavelength_simple_worm = Hilbert_Transform(curvatures, N)
    wavelength_preliminary = Hilbert_Transform(kappa, N)

    print("Difference in wavelength estimates: ", np.abs(wavelength_simple_worm - wavelength_preliminary))

    #--------curvature comparison--------------
    plt.figure(figsize=(10,6))
    plt.plot(t_eval, kappa[N//2,:], label='preliminary values')
    plt.plot(t_eval, curvatures[N//2,:], label='simple worm values')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Curvature')
    plt.show()

    #--------curvature difference--------------
    plt.figure(figsize=(10,6))
    plt.plot(t_eval, np.abs(kappa[N//2,:] - curvatures[N//2,:]), label='difference in vals')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Curvature difference')
    plt.show()