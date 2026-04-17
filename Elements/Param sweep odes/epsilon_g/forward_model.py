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



s = np.linspace(0.0,1.0,N)

def simulation(epsilon_g):

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

        sol = solve_ivp(ODEs,
                        (ODE_time, ODE_time + dt),
                        ODE_state,
                        method="RK45",
                        max_step=dt,args=(epsilon_g,))

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

    wavelengths = np.empty(len(epsilon_g_vals), dtype=object)
    frequencies = np.empty(len(epsilon_g_vals), dtype=object)
    velocities = np.empty(len(epsilon_g_vals), dtype=object)

    max_curvatures = np.empty(len(epsilon_g_vals), dtype=object)

    for i, epsilon_g in enumerate(epsilon_g_vals):
        print(f'{i+1} out of {len(epsilon_g_vals)}: {np.round((i)/len(epsilon_g_vals),2)*100}% done')
        worm_positions, curvatures = simulation(epsilon_g)
        print(worm_positions.shape)

        max_kappa = np.max(curvatures)
        max_curvatures[i] = max_kappa

        #---------Hilber Transform----------
        wavelength, frequency = Hilbert_Transform(curvatures, N, N_controls, t_eval)
        wavelengths[i] = wavelength
        frequencies[i] = frequency
        print("Wavelength: ", wavelength)
        print("Frequency: " , frequency)

        #---------Worm Velocity------------
        velocity_x = average_velocity(worm_positions, t_eval)
        print("Velocity: ", velocity_x)
        velocities[i] = velocity_x
    
    tock = time.time()

    print("--- %s seconds ---" % (np.round(tock - tick, 2)))

    plt.figure()
    plt.plot(epsilon_g_vals, wavelengths, 'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel("Normalised wavelength " + r'$\lambda / L$')
    plt.title("Wavelength against gap-junctional strength")
    plt.show()


    plt.figure()
    plt.plot(epsilon_g_vals, frequencies, 'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel(r'$\text{Frequency} \, \mathrm{Hz}$')
    plt.title("Frequency against gap-junctional strength")
    plt.show()

    plt.figure()
    plt.plot(epsilon_g_vals, velocities,'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel("velocity " + r'$\mathrm{mm/s}$')
    plt.title("Velocity against gap-junctional strength")
    plt.show()
    
    plt.figure()
    plt.plot(epsilon_g_vals, max_curvatures, 'ko')
    plt.xlabel("Gap-junctional strength " + r'$\varepsilon_g$')
    plt.ylabel("Curvature amplitude " + r'$\mathrm{mm^{-1}}$')
    plt.title("Curvature amplitude against gap-junctional strength")
    plt.show()
    