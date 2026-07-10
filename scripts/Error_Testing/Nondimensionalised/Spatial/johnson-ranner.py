from numpy.linalg import norm
from dolfinx import fem
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import ufl

from simple_worm.controls import (
    ControlsFenics,
    ControlsNumpy,
    ControlSequenceFenics,
)
from simple_worm.material_parameters import MaterialParameters, MaterialParametersFenics
from simple_worm.worm import Worm
from simple_worm.util import f2n, v2f

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.hilbert_calc import Hilbert_Transform
from scripts.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.curvature_viewer import view_curvature_pyqtgraph

from scripts.Error_Testing.Nondimensionalised.Spatial.ODEs import *


from pathlib import Path

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

T = 10.0  # Final time - recommend several undulations
dt = 1.0e-2

def simulation(N):
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """
    # holders for 'worm', u and control
    # Parameters
    

    n_timesteps = int(T / dt)
    s = np.linspace(0.0,1.0,N)

    ODE_state = np.zeros(4*N)
    ODE_state[2*N:3*N] = 0.5
    ODE_state[3*N:4*N] = -0.5

    kappa_init = np.zeros(N)

    
    def pref_curvature_at_u(u, betas):
        beta_idx = np.argmin(np.abs(u - s))
        return betas[beta_idx]

    
    # holders for 'worm', u and control
    worm = Worm(N, dt)

    # set material parameters
    MP = MaterialParameters(
        K=C_N / C_T,  # ratio of drag coefficients
        K_rot=1.0,  # rotational drag coefficient
        A=e,  # bending rigidity
        B=eta_tilde,  # bending viscosity
        C=1.0,  # twisting rigidity
        D=0.1,  # twisting viscosity
    )
    worm.initialise(MP)

    
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
        a = pref_curvature_at_u(u_control, betas)
        
        return a

    t = 0.0
    kappas = []
    worm_positions = []

    while t < T:

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK45",
                        max_step=dt, args=(kappa_init, N,))

        ODE_state = sols.y[:, -1]

        A_V = ODE_state[0:N]
        A_D = ODE_state[N:2*N]


        t += dt

        betas = sigma(A_V) - sigma(A_D)        
        
        # update control
        control[:] = [alpha_forcing(t, j) for j in range(N)]

        # solve
        Cntrl = ControlsNumpy(alpha=control, beta=zeroN, gamma=zeroNm)
        ret = worm.update_solution(Cntrl.to_fenics(worm))

        # output variables as 'fenics functions
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
        kappa_init = curvature_np.copy()

        #print(np.max(betas), np.max(kappa_init))
        
    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T


if __name__ == "__main__":

    
    #final_worm_positions = np.empty(len(N_vals), dtype=object)
    lambdas = np.empty(len(N_vals), dtype=object)
    freqs = np.empty(len(N_vals), dtype=object)
    maxcurvs = np.empty(len(N_vals), dtype=object)
    

    for i, N in enumerate(N_vals):
        
        print("N: ", N)
        t_eval = np.arange(0,T,dt)

        worm_positions, curvatures = simulation(N)

        #f_worm_position = worm_positions[-1]
        #final_worm_positions[i] = f_worm_position

        maxcurv = np.max(curvatures)
        maxcurvs[i] = maxcurv
        wave, freq = Hilbert_Transform(curvatures, N, N_controls, t_eval)

        lambdas[i] = wave
        freqs[i] = freq

        print("Wavelength: ", np.round(wave, 2))
        print("Frequency Hz: ", np.round(freq, 2))
        print("Max curvature: ", np.round(maxcurv, 2))
    
    
    #errors = []
    freqs_diff = []
    lambdas_diff = []
    maxcurvs_diff = []

    for i in range(len(lambdas) - 1):
        #worm_coarse = final_worm_positions[i]
        #worm_fine = final_worm_positions[i+1]

        #point_distances = np.linalg.norm(worm_coarse - worm_fine, axis=1)

        #errors.append(np.mean(point_distances))
        freqs_diff.append(np.abs(freqs[i] - freqs[i+1]))
        lambdas_diff.append(np.abs(lambdas[i] - lambdas[i+1]))
        maxcurvs_diff.append(np.abs(maxcurvs[i] - maxcurvs[i+1]))



    #errors = np.array(errors)

    #print(errors)

    #--------------worm position errors---------------
    """plt.figure()
    plt.plot(N_vals[:-1], errors, 'ko')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Difference in average distance')
    
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - dt errors.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
"""
    #-------------wavelength errors-----------------
    plt.figure()
    plt.plot(N_vals[:-1], lambdas_diff, 'ko')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Difference in wavelength')
    
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - wavelength differences.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    #--------------frequency diffs-----------
    plt.figure()
    plt.plot(N_vals[:-1], freqs_diff, 'ko')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Difference in frequency')
    
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - frequency differences.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    #--------------curvature amplitude diffs------------
    plt.figure()
    plt.plot(N_vals[:-1], maxcurvs_diff, 'ko')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Difference in curvature amplitude')
    
    plt.savefig(
        OUTPUT_DIR / f"{SCRIPT_NAME} - curvature amplitude differences.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

