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
from simple_worm.util import f2n, v2f#
from bodyform_animation import *
from curvature_animation import *
from hilbert_calc import *
from kymograph import *
from Carter_ODEs import *
from simple_worm_viewer_pyqtgraph import *
from velocity_calc import *


# Parameters
T = 5.0  # Final time - recommend several undulations
dt = 1.0e-2 # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

ODE_state = np.zeros(5*N)
ODE_state[3*N] = 1.0
ODE_state[4*N] = -1.0
ODE_time = 0.0

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
                    method="RK45",
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
    # holders for 'worm', u and control
    worm = Worm(N, dt)
    worm.initialise()

    def curvature_at_u(u):
        kappa_idx = [np.argmin(np.abs(u_val - s)) for u_val in u]
        kappa = ODE_state[kappa_idx]
        return kappa
    

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

    t = 0.0
    while t < T:
        t += dt
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
    worm_positions = np.array(worm_positions)
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

    worm_positions, curvatures = example1()
    
    #----------Kymogram--------------
    heights = [i/N for i in range(N)]
    heights=np.array(heights)
    
    t_eval = np.arange(0, T, dt)


    """times, kappa_peaks = finding_peaks(curvatures, t_eval, N)
    wavelength, selection_time = lin_reg_wavelength(kappa_peaks, times, N)

    print("Wavelength via kymogram: ", wavelength)"""
    
    plt.figure(figsize=(8,4))
    plt.imshow(curvatures, aspect='auto', extent=[0, T, heights[0], heights[-1]], origin='lower', cmap='bwr')
    plt.colorbar(label='curvature')
    plt.xlabel('Time')
    plt.ylabel('Body length')
    plt.tight_layout()
    plt.savefig("kymogram.png")
    plt.close()

    max_curvatur = np.max(curvatures)
    print("max curvature: ", np.round(max_curvatur, 2))

    #---------Hilber Transform----------
    wavelength = Hilbert_Transform(curvatures, N, N_controls, t_eval)
    print("Wavelength via Hilbert transform: ", np.round(wavelength, 2))


    #---------Worm Velocity------------
    velocity_x = Average_Velocity(worm_positions, t_eval)

    print("Average velocity (mm/s): ", np.round(velocity_x, 2))

    """#----------Animation----------------
    anim = create_worm_animation(worm_positions, dt, plane='xz')
    anim.save("worm animation.mp4", writer="ffmpeg", fps=int(1/dt), dpi=150, extra_args=["-vcodec", "libx264"])
    plt.close()

    #----------Curvature Animation----------------
    curvature_anim = create_curvature_animation(N, L, curvatures)
    curvature_anim.save("curvature_anim.mp4", writer="ffmpeg", fps=int(1/dt), dpi=150, extra_args=["-vcodec", "libx264"])
    plt.close()"""

    view_worm_pyqtgraph(worm_positions, dt)
    