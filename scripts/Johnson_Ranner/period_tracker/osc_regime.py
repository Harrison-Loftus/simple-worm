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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.kymograph import *
from scripts.hilbert_calc import *
from scripts.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.curvature_viewer import view_curvature_pyqtgraph
from scripts.matplotlib_animate_worm import *


from scripts.Johnson_Ranner.period_tracker.ODEs import *

from pathlib import Path
import time

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

BASE_DIR = OUTPUT_DIR / SCRIPT_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)

anim_dir = BASE_DIR / "animations"
anim_dir.mkdir(parents=True, exist_ok=True)

kappavdiff_dir = BASE_DIR / "neural_phase"
kappavdiff_dir.mkdir(parents=True, exist_ok=True)

kappadkappadt_dir = BASE_DIR / "kappa_phase"
kappadkappadt_dir.mkdir(parents=True, exist_ok=True)

kappatime_dir = BASE_DIR / "kappa_time"
kappatime_dir.mkdir(parents=True, exist_ok=True)

AVdAVdt_dir = BASE_DIR / "AV_phase"
AVdAVdt_dir.mkdir(parents=True, exist_ok=True)

AVtime_dir = BASE_DIR / "AV_time"
AVtime_dir.mkdir(parents=True, exist_ok=True)

VVdVVdt_dir = BASE_DIR / "VV_phase"
VVdVVdt_dir.mkdir(parents=True, exist_ok=True)

VVtime_dir = BASE_DIR / "VV_time"
VVtime_dir.mkdir(parents=True, exist_ok=True)

txt_dir = BASE_DIR / "txt"
txt_dir.mkdir(parents=True, exist_ok=True)

av_dir = txt_dir / "AV"
av_dir.mkdir(parents=True, exist_ok=True)

ad_dir = txt_dir / "AD"
ad_dir.mkdir(parents=True, exist_ok=True)

vv_dir = txt_dir / "VV"
vv_dir.mkdir(parents=True, exist_ok=True)

vd_dir = txt_dir / "VD"
vd_dir.mkdir(parents=True, exist_ok=True)

davdt_dir = txt_dir / "dAVdt"
davdt_dir.mkdir(parents=True, exist_ok=True)

daddt_dir = txt_dir / "dADdt"
daddt_dir.mkdir(parents=True, exist_ok=True)

dvvdt_dir = txt_dir / "dVVdt"
dvvdt_dir.mkdir(parents=True, exist_ok=True)

dvddt_dir = txt_dir / "dVDdt"
dvddt_dir.mkdir(parents=True, exist_ok=True)

kappa_dir = txt_dir / "kappa"
kappa_dir.mkdir(parents=True, exist_ok=True)

dkappadt_dir = txt_dir / "dkappadt"
dkappadt_dir.mkdir(parents=True, exist_ok=True)



T = 10.0  # Final time - recommend several undulations

dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower

#t_eval = np.arange(0,T,dt)


def example4():
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

    peak_times = []

    periods = []
    peak_heights = []

    convergence_tol = 0.05      # 5%
    consecutive_required = 3    # require 3 converged periods

    n_converged = 0

    extra_cycles = 3
    termination_time = None

    mid_idx = N // 2

    prev_curvatures = []
    prev_times = []

    times = []

    A_V_anterior_vals = []
    A_D_anterior_vals = []
    V_V_anterior_vals = []
    V_D_anterior_vals = []

    A_V_mid_vals = []
    A_D_mid_vals = []
    V_V_mid_vals = []
    V_D_mid_vals = []
    

    A_V_posterior_vals = []
    A_D_posterior_vals = []
    V_V_posterior_vals = []
    V_D_posterior_vals = []


    peak_indices = []
    
    while True:

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK45",
                        max_step=dt,args=(kappa_init,))

        ODE_state = sols.y[:, -1]

        A_V = ODE_state[0:N_muscular]
        A_D = ODE_state[N_muscular:2*N_muscular]
        V_V = ODE_state[2*N_muscular: 2*N_muscular + N_controls]
        V_D = ODE_state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls]

        A_V_anterior = A_V[N_muscular//4]
        A_D_anterior = A_D[N_muscular//4]
        V_V_anterior = V_V[N_controls//4]
        V_D_anterior = V_D[N_controls//4]

        A_V_mid = A_V[N_muscular//2]
        A_D_mid = A_D[N_muscular//2]
        V_V_mid = V_V[N_controls//2]
        V_D_mid = V_D[N_controls//2]


        A_V_posterior = A_V[-N_muscular//4]
        A_D_posterior = A_D[-N_muscular//4]
        V_V_posterior = V_V[-N_controls//4]
        V_D_posterior = V_D[-N_controls//4]


        A_V_mid_vals.append(A_V_mid)
        A_D_mid_vals.append(A_D_mid)
        V_V_mid_vals.append(V_V_mid)
        V_D_mid_vals.append(V_D_mid)


        A_V_anterior_vals.append(A_V_anterior)
        A_D_anterior_vals.append(A_D_anterior)
        V_V_anterior_vals.append(V_V_anterior)
        V_D_anterior_vals.append(V_D_anterior)

        A_V_posterior_vals.append(A_V_posterior)
        A_D_posterior_vals.append(A_D_posterior)
        V_V_posterior_vals.append(V_V_posterior)
        V_D_posterior_vals.append(V_D_posterior)
    


        t += dt
        print(np.round(t,2))

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

                                mean_period = np.mean(periods[-consecutive_required:])

                                termination_time = (t + extra_cycles * mean_period)

                                print(f"Converged. " f"Stopping at {termination_time:.3f}")

        if termination_time is not None and t >= termination_time:
            break

    A_V_vals = np.array([A_V_anterior_vals, A_V_mid_vals, A_V_posterior_vals])
    A_D_vals = np.array([A_D_anterior_vals, A_D_mid_vals, A_D_posterior_vals])
    V_V_vals = np.array([V_V_anterior_vals, V_V_mid_vals, V_V_posterior_vals])
    V_D_vals = np.array([V_D_anterior_vals, V_D_mid_vals, V_D_posterior_vals])

    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T, A_V_vals, A_D_vals,V_V_vals, V_D_vals, np.array(times), peak_heights, peak_times, peak_indices



if __name__ == "__main__":
    tick = time.time()

    worm_positions, curvatures, A_V, A_D, V_V, V_D, times, peak_heights, peak_times, peak_indices = example4()

    anim = create_worm_animation(worm_positions[:,:,:], dt)
    filename = (anim_dir / f"{SCRIPT_NAME}_eps{epsilon_p:.2f}_I{I:.4f}.mp4")
    anim.save(filename, writer="ffmpeg", fps = int(round(1/dt)), dpi=150)
    plt.close("all")


    k_anterior = curvatures[N//4, :]
    k_mid = curvatures[N//2, :]
    k_posterior = curvatures[-N//4, :]

    
    with open(
        anim_dir / f"worm_positions_eps{epsilon_p:.2f}_I{I:.4f}.txt",
        "w"
    ) as f:

        for t_idx in range(len(worm_positions)):

            f.write(f"timestep {t_idx}\n")

            for node_idx in range(N):
                x, y, z = worm_positions[t_idx, node_idx]

                f.write(
                    f"{node_idx}, {x}, {y}, {z}\n"
                )

            f.write("\n")

    with open(kappa_dir / f"kappa_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("k_anterior\n")
        for val in k_anterior:
            f.write(f"{val}\n")
        f.write("\nk_mid\n")
        for val in k_mid:
            f.write(f"{val}\n")
        f.write("\nk_posterior\n")
        for val in k_posterior:
            f.write(f"{val}\n")
    
    dkdt_anterior = np.gradient(k_anterior, dt)        
    dkdt_mid = np.gradient(k_mid, dt)
    dkdt_posterior = np.gradient(k_posterior, dt)

    with open(dkappadt_dir / f"dkappa_dt_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("dkdt_anterior\n")
        for val in dkdt_anterior:
            f.write(f"{val}\n")
        f.write("\ndkdt_mid\n")
        for val in dkdt_mid:
            f.write(f"{val}\n")
        f.write("\ndkdt_posterior\n")
        for val in dkdt_posterior:
            f.write(f"{val}\n")
    
    A_V_anterior = A_V[0]
    A_V_mid = A_V[1]
    A_V_posterior = A_V[2]

    with open(av_dir / f"AV_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("A_V_anterior\n")
        for val in A_V_anterior:
            f.write(f"{val}\n")
        f.write("\nA_V_mid\n")
        for val in A_V_mid:
            f.write(f"{val}\n")
        f.write("\nA_V_posterior\n")
        for val in A_V_posterior:
            f.write(f"{val}\n")

    A_D_anterior = A_D[0]
    A_D_mid = A_D[1]
    A_D_posterior = A_D[2]

    with open(ad_dir / f"AD_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("A_D_anterior\n")
        for val in A_D_anterior:
            f.write(f"{val}\n")
        f.write("\nA_D_mid\n")
        for val in A_D_mid:
            f.write(f"{val}\n")
        f.write("\nA_D_posterior\n")
        for val in A_D_posterior:
            f.write(f"{val}\n")
    
    V_V_anterior = V_V[0]
    V_V_mid = V_V[1]
    V_V_posterior = V_V[2]

    with open(vv_dir / f"VV_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("V_V_anterior\n")
        for val in V_V_anterior:
            f.write(f"{val}\n")
        f.write("\nV_V_mid\n")
        for val in V_V_mid:
            f.write(f"{val}\n")
        f.write("\nV_V_posterior\n")
        for val in V_V_posterior:
            f.write(f"{val}\n")
    
    V_D_anterior = V_D[0]
    V_D_mid = V_D[1]
    V_D_posterior = V_D[2]

    with open(vd_dir / f"VD_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("V_D_anterior\n")
        for val in V_D_anterior:
            f.write(f"{val}\n")
        f.write("\nV_D_mid\n")
        for val in V_D_mid:
            f.write(f"{val}\n")
        f.write("\nV_D_posterior\n")
        for val in V_D_posterior:
            f.write(f"{val}\n")


    davdt_anterior = np.gradient(A_V_anterior, dt)
    davdt_mid = np.gradient(A_V_mid, dt)
    davdt_posterior = np.gradient(A_V_posterior, dt)

    with open(davdt_dir / f"dAVdt_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("dA_Vdt_anterior\n")
        for val in davdt_anterior:
            f.write(f"{val}\n")
        f.write("\ndA_Vdt_mid\n")
        for val in davdt_mid:
            f.write(f"{val}\n")
        f.write("\ndA_Vdt_posterior\n")
        for val in davdt_posterior:
            f.write(f"{val}\n")

    daddt_anterior = np.gradient(A_D_anterior, dt)
    daddt_mid = np.gradient(A_D_mid, dt)
    daddt_posterior = np.gradient(A_D_posterior, dt)

    with open(daddt_dir / f"dADdt_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("dA_Ddt_anterior\n")
        for val in daddt_anterior:
            f.write(f"{val}\n")
        f.write("\ndA_Ddt_mid\n")
        for val in daddt_mid:
            f.write(f"{val}\n")
        f.write("\ndA_Ddt_posterior\n")
        for val in daddt_posterior:
            f.write(f"{val}\n")

    dVvdt_anterior = np.gradient(V_V_anterior, dt)
    dVvdt_mid = np.gradient(V_V_mid, dt)
    dVvdt_posterior = np.gradient(V_V_posterior, dt)

    with open(dvvdt_dir / f"dVVdt_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("dV_Vdt_anterior\n")
        for val in dVvdt_anterior:
            f.write(f"{val}\n")
        f.write("\ndV_Vdt_mid\n")
        for val in dVvdt_mid:
            f.write(f"{val}\n")
        f.write("\ndV_Vdt_posterior\n")
        for val in dVvdt_posterior:
            f.write(f"{val}\n")

    dvddt_anterior = np.gradient(V_D_anterior, dt)
    dvddt_mid = np.gradient(V_D_mid, dt)
    dvddt_posterior = np.gradient(V_D_posterior, dt)

    with open(dvddt_dir / f"dVDdt_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("dV_Ddt_anterior\n")
        for val in dvddt_anterior:
            f.write(f"{val}\n")
        f.write("\ndV_Ddt_mid\n")
        for val in dvddt_mid:
            f.write(f"{val}\n")
        f.write("\ndV_Ddt_posterior\n")
        for val in dvddt_posterior:
            f.write(f"{val}\n")

    V_diff_anterior = V_V_anterior - V_D_anterior
    V_diff_mid = V_V_mid - V_D_mid
    V_diff_posterior = V_V_posterior - V_D_posterior

    #---------Kappa vs V diff--------------
    plt.figure()
    plt.plot(V_diff_anterior, k_anterior, '--', label='anterior', color='blue', )
    plt.plot(V_diff_mid, k_mid, '--',label='Middle', color='green')
    plt.plot(V_diff_posterior, k_posterior, '--',label='posterior', color='red')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$V_\text{V} - V_\text{D}$', size=12)
    plt.ylabel(r'$\kappa$', size=12)
    plt.title(f'Phase portrait of kappa vs V diff for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(kappavdiff_dir / f"vdiff_kappa_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

    with open(kappavdiff_dir / f"vdiff_kappa_eps{epsilon_p:.2f}_I{I:.4f}.txt", "w") as f:
        f.write("V_diff_anterior, k_anterior\n")
        for v_diff, k in zip(V_diff_anterior, k_anterior):
            f.write(f"{v_diff}, {k}\n")
        f.write("\nV_diff_mid, k_mid\n")
        for v_diff, k in zip(V_diff_mid, k_mid):
            f.write(f"{v_diff}, {k}\n")
        f.write("\nV_diff_posterior, k_posterior\n")
        for v_diff, k in zip(V_diff_posterior, k_posterior):
            f.write(f"{v_diff}, {k}\n")

    #---------Kappa vs dKappa/dt--------------
    plt.figure()
    plt.plot(k_anterior, dkdt_anterior, '--', label='anterior', color='blue')
    plt.plot(k_mid, dkdt_mid, '--', label='Middle', color='green')
    plt.plot(k_posterior, dkdt_posterior, '--', label='posterior', color='red')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$\kappa$', size=12)
    plt.ylabel(r'$\frac{d\kappa}{dt}$', size=12)
    plt.title(f'Phase portrait of kappa vs dkappa/dt for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(kappadkappadt_dir / f"kappa_dkappa_dt_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

    

    #--------Kappa vs time--------------
    plt.figure()
    plt.plot(times, k_anterior, '--', label='anterior', color='blue')
    plt.plot(times, k_mid, '--', label='Middle', color='green')
    plt.plot(times, k_posterior, '--', label='posterior', color='red')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$t$', size=12) 
    plt.ylabel(r'$\kappa$', size=12)
    plt.title(f'Kappa vs time for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(kappatime_dir / f"kappa_time_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

    
    #---------A_V vs dA_V/dt--------------
    plt.figure()
    plt.plot(A_V_anterior, davdt_anterior, '--', label='anterior', color='blue')
    plt.plot(A_V_mid, davdt_mid, '--', label='Middle', color='green')
    plt.plot(A_V_posterior, davdt_posterior, '--', label='posterior', color='red')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$A_V$', size=12)
    plt.ylabel(r'$\frac{dA_V}{dt}$', size=12)
    plt.title(f'Phase portrait of A_V vs dA_V/dt for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(AVdAVdt_dir / f"AV_dAV_dt_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

    
    #---------A_V vs time--------------
    plt.figure()
    plt.plot(times, A_V_anterior, '--', label='anterior', color='blue')
    plt.plot(times, A_V_mid, '--', label='Middle', color='green')
    plt.plot(times, A_V_posterior, '--', label='posterior', color='red')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$t$', size=12)
    plt.ylabel(r'$A_V$', size=12)
    plt.title(f'A_V vs time for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(AVtime_dir / f"AV_time_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

    
    #---------V_V vs dV_V/dt--------------
    plt.figure()
    plt.plot(V_V_anterior, dVvdt_anterior, '--', label='anterior', color='blue')
    plt.plot(V_V_mid, dVvdt_mid, '--', label='Middle', color='green')
    plt.plot(V_V_posterior, dVvdt_posterior, '--', label='posterior', color='red')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$V_V$', size=12)
    plt.ylabel(r'$\frac{dV_V}{dt}$', size=12)
    plt.title(f'Phase portrait of V_V vs dV_V/dt for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(VVdVVdt_dir / f"VV_dVV_dt_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

    
    #---------V_V vs time--------------
    plt.figure()
    plt.plot(times, V_V_anterior, '--', label='anterior', color='blue')
    plt.plot(times, V_V_mid, '--', label='Middle', color='green')
    plt.plot(times, V_V_posterior, '--', label='posterior', color='red')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$t$', size=12)
    plt.ylabel(r'$V_V$', size=12)
    plt.title(f'V_V vs time for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(VVtime_dir / f"VV_time_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)
    plt.close("all")

    if len(peak_indices)==0:
        sys.exit()
    else:


        final_peaks = peak_indices[-3:]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, idx in zip(axes, final_peaks):

            posture = worm_positions[idx]

            x = posture[:, 0]
            y = posture[:, 2]

            ax.plot(x, y, 'k', lw=2)

            ax.set_aspect('equal')
            ax.set_title(f't = {times[idx]:.2f}s', size=12)

        plt.tight_layout()
        plt.savefig(BASE_DIR / f"final_three_peaks_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)
        plt.close("all")

        plt.figure(figsize=(6,6))

        colours = ['r', 'g', 'b']

        for idx, c in zip(final_peaks, colours):

            posture = worm_positions[idx]

            x = posture[:,0]
            y = posture[:,2]

            plt.plot(x, y, color=c, lw=2, label=f't={times[idx]:.2f}s')

        plt.axis('equal')
        plt.legend()
        plt.xlabel(r'$x$', size=12)
        plt.ylabel(r'$y$', size=12)
        plt.title('Final three peak postures', size=12)
        plt.savefig(BASE_DIR / f"final_three_peaks_combined_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)
        plt.close("all")

        last3_amp = peak_heights[-3:]
        last3_periods = np.diff(peak_times[-4:])
        last3_freqs = 1 / last3_periods
        
        print("Last 3 peak amplitudes = ", np.round(last3_amp, 4))
        print("Amplitude mean =", np.mean(last3_amp))
        print("Amplitude std =", np.std(last3_amp))
        print("Amplitude CV (%) =", 100*np.std(last3_amp)/np.mean(last3_amp))

        print("Last 3 frequencies = ", np.round(last3_freqs, 4))
        print("Amplitude mean =", np.mean(last3_freqs))
        print("Amplitude std =", np.std(last3_freqs))
        print("Amplitude CV (%) =", 100*np.std(last3_freqs)/np.mean(last3_freqs))

        plt.figure(figsize=(6,6))

        for idx, c in zip(final_peaks, colours):
            posture = worm_positions[idx].copy()
            posture -= posture[0]
            plt.plot(posture[:,0], posture[:,2], color=c, lw=2, label=f't={times[idx]:.2f}s')

        plt.legend()
        plt.axis('equal')
        plt.title('Aligned peak postures', size=12)
        plt.savefig(BASE_DIR / f"aligned_final_three_peaks_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)
        plt.close("all")

        last3_waves = []

        for i in range(-6,-3):
            t0 = peak_times[i]
            t1 = peak_times[i+3]
            print("t0: ", t0)
            print("t1: ", t1)

            idx0 = np.searchsorted(times, t0)
            idx1 = np.searchsorted(times, t1)

            print(times[idx0])
            print(times[idx1])

            wave_i, _ = lin_reg_wavelength(curvatures[:, idx0:idx1], times[idx0:idx1], N, N_controls)

            last3_waves.append(wave_i)

        last3_waves = np.array(last3_waves)

        print("Last 3 wavelengths = ", np.round(last3_waves, 4))
        print("Amplitude mean =", np.mean(last3_waves))
        print("Amplitude std =", np.std(last3_waves))
        print("Amplitude CV (%) =", 100*np.std(last3_waves)/np.mean(last3_waves))



        maxcurv = np.max(curvatures)

        wave, freq = lin_reg_wavelength(curvatures[:,int(-20/dt):], times[int(-20/dt):], N, N_controls)
        wave_h, freq_h = Hilbert_Transform(curvatures, N, N_controls, times)
        
        print("Wavelength: ", np.round(wave, 2))
        print("Frequency Hz: ", np.round(freq, 2))
        print("Max curvature: ", np.round(maxcurv, 2))

        print("wave_H: ", wave_h)
        print("freq_H: ", freq_h)

        
    
        tock = time.time()
        print("time: ", np.round(tock - tick, 2))
        

        #view_worm_pyqtgraph(worm_positions, dt)
        #view_curvature_pyqtgraph(curvatures, dt)
