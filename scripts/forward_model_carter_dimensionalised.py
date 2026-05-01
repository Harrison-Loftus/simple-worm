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

from Carter_ODEs_dimensionalised import *
#from simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from curvature_viewer import view_curvature_pyqtgraph


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

# Note: example 3 and 4 use nondimensionalised material parameters. Here we are importing Carter_ODES which has dimensionalised parameters.
# A new set of ODEs will need to be made for examples 3 and 4. This is just a sense check of implementation for now.

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


def example3():
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """
    # holders for 'worm', u and control
    worm = Worm(N, dt)

    def curvature_at_u(u):
        kappa_idx = [np.argmin(np.abs(u_val - s)) for u_val in u]
        kappa = ODE_state[kappa_idx]
        return kappa
    

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

    
    # specific forcing function
    def alpha_forcing(t):
        def alpha_forcing_t(u_):
            u = u_[0]      # parametric coordinate along worm
            kappas = curvature_at_u(u)
            return kappas
        return alpha_forcing_t

        

    def zero_forcing(u):
        return 0.0 * u[0]

    t = 0.0
    kappas = []
    worm_positions = []
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


    def curvature_at_u(u):
        kappa_idx = np.argmin(np.abs(u - s)) 
        kappa = ODE_state[kappa_idx]
        return kappa

    # holders for 'worm', u and control
    global N
    N = 120
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


if __name__ == "__main__":

    worm_positions, curvatures = example2()
    
    view_curvature_pyqtgraph(curvatures, dt)
    
    