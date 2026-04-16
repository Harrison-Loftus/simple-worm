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
from hilbert_calc import *
from Carter_ODEs import *
from Worm_velocity_calc import *

# Parameters
T = 5.0  # Final time - recommend several undulations
dt = 1.0e-2 # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

ODE_state = np.zeros(5*N)
ODE_state[3*N] = 1.0
ODE_state[4*N] = -1.0
ODE_time = 0.0

s = np.linspace(0.0,1.0,N)

def simulation(W_p_val):

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
                        method="RK45",
                        max_step=dt,args=(W_p_val,))

        ODE_state = sol.y[:, -1]
        ODE_time += dt

        # extract curvature for this timestep
        kappa_current = ODE_state[0:N]


    def curvature_at_u(u):
        kappa_idx = np.argmin(np.abs(u - s)) 
        kappa = ODE_state[kappa_idx]
        return kappa


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




if __name__ == "__main__":

    tick = time.time()

    t_eval = np.arange(0, T, dt)

    wavelengths = np.empty(len(range_vals), dtype=object)
    frequencies = np.empty(len(range_vals), dtype=object)
    velocities = np.empty(len(range_vals), dtype=object)

    for i, W_p in enumerate(W_p_vals):
        print(f'{i+1} out of {len(range_vals)}: {np.round((i)/len(range_vals *100),2)}% done')
        worm_positions, curvatures = simulation(W_p)
        

        #---------Hilber Transform----------
        wavelength, frequency = Hilbert_Transform(curvatures, N, N_controls, t_eval)
        wavelengths[i] = wavelength
        frequencies[i] = frequency
        print("Frequency: " , frequency)

        #---------Worm Velocity------------
        velocity_x = average_velocity(worm_positions, t_eval)
        velocities[i] = velocity_x
    
    tock = time.time()

    print("--- %s seconds ---" % (np.round(tock - tick, 2)))

    plt.figure()
    plt.plot(range_vals, wavelengths, 'ko')
    plt.xlabel("Proprioceptive strength " + r'$\varepsilon_p$')
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.title("Wavelength against proprioceptive strength")
    plt.show()


    plt.figure()
    plt.plot(range_vals, frequencies, 'ko')
    plt.xlabel("Proprioceptive strength " + r'$\varepsilon_p$')
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.title("Frequency against proprioceptive strength")
    plt.show()

    plt.figure()
    plt.plot(range_vals, velocities,'ko')
    plt.xlabel("Proprioceptive strength " + r'$\varepsilon_p$')
    plt.ylabel("Velocity " + r'$\mathrm{mm/s}$')
    plt.title("Velocity against proprioceptive strength")
    plt.show()
    
    