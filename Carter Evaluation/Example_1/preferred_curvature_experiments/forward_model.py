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
from worm_anim_controller import *




# Parameters
# Number of body points - recommend ~100
T = 5.0  # Final time - recommend several undulations
dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower
n_timesteps = int(T / dt)

ODE_state = np.zeros(5*N)
ODE_state[3*N] = 1.0
ODE_state[4*N] = -1.0
ODE_time = 0.0
t_eval = np.arange(0, T, dt)

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
        #plot_curve(x_np)
    return worm_positions, curvatures




if __name__ == "__main__":
    worm_positions, curvatures = example1()
    curvatures = curvatures.T

    heights = [i/N for i in range(N)]
    """times, kappa_peaks = finding_peaks(curvatures, t_eval, N)
    wavelength, selection_time = lin_reg_wavelength(kappa_peaks, times, N)
    print("Wavelength via kymogram: ", wavelength)"""
    
    plt.figure(figsize=(8,4))
    plt.imshow(curvatures, aspect='auto', extent=[t_eval[0], t_eval[-1], heights[0], heights[-1]], origin='lower', cmap='bwr')
    plt.colorbar(label='curvature')
    plt.xlabel('Time')
    plt.ylabel('Body length')
    plt.tight_layout()
    plt.savefig("kymogram.png")
    plt.close()


    curvature_anim = create_curvature_animation(N, L, curvatures)
    curvature_anim.save("curvature_anim.mp4", writer="ffmpeg", fps=int(1/dt), dpi=150, extra_args=["-vcodec", "libx264"])
    plt.close()

    anim = create_worm_animation(worm_positions, dt, plane='xz')
    anim.save("worm animation.mp4", writer="ffmpeg", fps=int(1/dt), dpi=150, extra_args=["-vcodec", "libx264"])
    plt.close()

    view_worm_pyqtgraph(worm_positions, dt)
    