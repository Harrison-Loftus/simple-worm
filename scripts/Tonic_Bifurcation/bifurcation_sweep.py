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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kymograph import *
from scripts.hilbert_calc import *
from scripts.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.curvature_viewer import view_curvature_pyqtgraph

from scripts.Tonic_Bifurcation.ODEs import *

from pathlib import Path
import time

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

T = 30.0  # Final time - recommend several undulations

dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower

t_eval = np.arange(0,T,dt)


def simulation(I, epsilon_p):
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """
    # holders for 'worm', u and control
    # Parameters


    
    
    n_timesteps = int(T / dt)
    s = np.linspace(0.0,1.0,N)

    ODE_state = np.zeros(2*N_muscular + 2*N_controls)
    ODE_state[2*N_muscular: 2*N_muscular + N_controls] = 0.5
    ODE_state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls] = -0.5

    kappa_init = np.zeros(N)

    
    def pref_curvature_at_u(u, betas):
        beta_idx = np.argmin(np.abs(u - s))
        return betas[beta_idx]

    
    # holders for 'worm', u and control
    worm = Worm(N, dt)

    # set material parameters
    MP = MaterialParameters(
        K=K_agar,  # ratio of drag coefficients
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

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK23",
                        max_step=dt,args=(kappa_init,I,epsilon_p,))

        ODE_state = sols.y[:, -1]

        A_V = ODE_state[0:N_muscular]
        A_D = ODE_state[N_muscular:2*N_muscular]


        t += dt
        
        repeat_factor = N // N_muscular
        betas = A_V - A_D       
        betas = np.repeat(betas, repeat_factor)
        # print("betas: ", np.round(betas[N//2], 2), "A_V: ", np.round(A_V[N_muscular // 2], 2), "A_D", np.round(A_D[N_muscular // 2],2))

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
    tick = time.time()
    wavelengths = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)
    frequencies = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)
    max_curvatures = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)
    

    for i, epsilon_p in enumerate(epsilon_p_vals):
        print("eps: ", epsilon_p)
        for j, I in enumerate(I_vals):
            print("I: ", I)
            worm_positions, curvatures = simulation(I, epsilon_p)
            wave, freq = lin_reg_wavelength(curvatures[:,int(-20/dt):], t_eval[int(-20/dt):], N, N_controls)

            max_curv = np.max(curvatures)

            wavelengths[i,j] = wave
            frequencies[i,j] = freq
            max_curvatures[i,j] = max_curv

            wave_h, freq_h = Hilbert_Transform(curvatures, N, N_controls, t_eval)


            print("freq: ", freq)
            print("wave: ", wave)
            print("max curve: ", np.round(max_curv, 2))
            print("wave_h: ", wave_h)
            print("freq_h: ", freq_h)

    cmap = plt.cm.bwr.copy()
    cmap.set_bad('black')

    plt.figure()
    plt.imshow(wavelengths, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    cbar = plt.colorbar(label=r'Wavelength')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of wavelength for varying values of tonic input and proprioceptive strength', size=16)
    plt.show()

    cmap = plt.cm.bwr.copy()
    cmap.set_bad('black')

    plt.figure()
    plt.imshow(frequencies, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of frequency for varying values of tonic input and proprioceptive strength', size=16)
    plt.show()


    cmap = plt.cm.bwr.copy()
    cmap.set_bad('black')

    plt.figure()
    plt.imshow(max_curvatures, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of maximum curvature for varying values of tonic input and proprioceptive strength', size=16)
    plt.show()
