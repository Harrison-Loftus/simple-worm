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

from matplotlib.lines import Line2D


from pathlib import Path

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PAPER_DIR = Path(OUTPUT_DIR / "FOR_PAPER")
PAPER_DIR.mkdir(exist_ok=True)  

SCRIPT_NAME = Path(__file__).stem

from scripts.Archive.hilbert_calc import Hilbert_Transform
from scripts.Archive.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.Archive.curvature_viewer import view_curvature_pyqtgraph
from scripts.Archive.kymograph import *

from scripts.Archive.Viscosity_sweep_denham.ODEs import *
from scripts.Archive.parameters import *

T = 10.0  # Final time - recommend several undulations
T_max = 100.0

dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower

t_eval = np.arange(0,T,dt)


def simulation(k_value, epsilon_p):
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
        K=k_value,  # ratio of drag coefficients
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

    peak_times = []

    periods = []
    peak_heights = []


    consecutive_required = 3    # require 3 converged periods

    n_converged = 0

    extra_cycles = 3
    termination_time = None

    mid_idx = N // 2

    prev_curvatures = []
    prev_times = []

    times = []

    peak_indices = []
    
    while True:

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK45",
                        max_step=dt,args=(kappa_init, epsilon_p))

        ODE_state = sols.y[:, -1]

        A_V = ODE_state[0:N_muscular]
        A_D = ODE_state[N_muscular:2*N_muscular]
        V_V = ODE_state[2*N_muscular: 2*N_muscular + N_controls]
        V_D = ODE_state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls]

            


        t += dt
        # print(np.round(t,2))

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

        times.append(t)

        #print(np.max(betas), np.max(kappa_init))
        

        if t >= T:
             
            prev_curvatures.append(curvature_np[mid_idx])
            prev_times.append(t)

            if times[-1] >= 20.0 and len(peak_times) == 0:
                break
            


            if t >= T_max:
                print("Reached maximum time without convergence.")
                break

            if len(prev_curvatures) >= 3:

                y0 = prev_curvatures[-3]
                y1 = prev_curvatures[-2]
                y2 = prev_curvatures[-1]

                
                if y1 > y0 and y1 > y2 and y1 > 0.1:

                    peak_time = prev_times[-2]
                    peak_times.append(peak_time)
                    peak_heights.append(y1)
                    peak_indices.append(len(times) - 2)

                    
                    print(f"Peak at {peak_time:.3f}")

                    if len(peak_times) >= 2:

                        
                        period = peak_times[-1] - peak_times[-2]
                        periods.append(period)

                        print(f"Period = {period:.3f}")

                        if len(periods) >= consecutive_required:

                            recent_periods = np.array(periods[-consecutive_required:])

                            mean_period = np.mean(recent_periods)

                            period_error = (np.max(np.abs(recent_periods - mean_period)) / mean_period)

                            print(f"Period error = {period_error:.4f}")

                            if termination_time is None and period_error < 0.01:
                                termination_time = (t + extra_cycles * mean_period)

                                print(f"Converged. " f"Stopping at {termination_time:.3f}")
                                mean_period = np.mean(periods[-consecutive_required:])

                                termination_time = (t + extra_cycles * mean_period)

                                print(f"Converged. " f"Stopping at {termination_time:.3f}")

        if termination_time is not None and t >= termination_time:
            break



    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T, np.array(times), peak_heights, peak_times, peak_indices



if __name__ == "__main__":
    K_vals = [1.57, 7.5, 12, 18, 30, 45]
    
    wavelengths = np.full((len(K_vals), len(epsilon_p_vals)), np.nan)
    frequencies = np.full((len(K_vals), len(epsilon_p_vals)), np.nan)
    maxcurvs = np.full((len(K_vals), len(epsilon_p_vals)), np.nan)


    n = len(K_vals) * len(epsilon_p_vals)
    a = 0

    for i, K_val in enumerate(K_vals):
        for j, epsilon_p in enumerate(epsilon_p_vals):
            
            a += 1
            print(f"Running simulation {a} of {n}")
            worm_positions, curvatures, times, peak_heights, peak_times, peak_indices = simulation(K_val, epsilon_p)

            maxcurv = np.max(curvatures)
            maxcurvs[i, j] = maxcurv

            wave, freq = lin_reg_wavelength(curvatures, times, N, N_controls)        
            
            wavelengths[i, j] = wave
            frequencies[i, j] = freq

                
            
    with open(PAPER_DIR / f"wavelengths_{SCRIPT_NAME}.npy", "wb") as f:
        np.save(f, wavelengths)
    with open(PAPER_DIR / f"frequencies_{SCRIPT_NAME}.npy", "wb") as f:
        np.save(f, frequencies)
    with open(PAPER_DIR / f"maxcurvs_{SCRIPT_NAME}.npy", "wb") as f:
        np.save(f, maxcurvs)


    markers = {
        1.57: "x",
        7.5: "o",
        12: "s",
        18: "D",
        30: "^",
        45: "*"
    }

    

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap("Greens")
    norm = plt.Normalize(
        vmin=np.min(epsilon_p_vals),
        vmax=np.max(epsilon_p_vals)
    )

    for i, K_val in enumerate(K_vals):
        for j, eps in enumerate(epsilon_p_vals):

            freq = frequencies[i, j]
            wave = wavelengths[i, j]

            ax.scatter(
                freq,
                wave,
                marker=markers[K_val],
                c=[eps],
                cmap=cmap,
                norm=norm,
                s=65,
                edgecolors="none"
            )

    # Colourbar represents proprioceptive range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Proprioceptive strength", fontsize=14)

    # Separate legend for K marker shapes
    legend_elements = [
        Line2D(
            [0], [0],
            marker=markers[K],
            color="black",
            linestyle="None",
            markersize=8,
            label=fr"$K = {K}$"
        )
        for K in K_vals
    ]

    ax.legend(
        handles=legend_elements,
        title="Drag ratio",
        loc="best"
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Wavelength (mm)", fontsize=14)

    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()

    fig.savefig(
        PAPER_DIR / f"{SCRIPT_NAME}_strength_modulation.png",
        dpi=300,
        bbox_inches="tight"
    )
