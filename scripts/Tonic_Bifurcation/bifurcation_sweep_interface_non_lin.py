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
from scripts.matplotlib_animate_worm import *

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

    
    A_V_head_vals = []
    V_V_head_vals = []
    V_D_head_vals = []

    A_V_mid_vals = []
    V_V_mid_vals = []
    V_D_mid_vals = []
    

    A_V_tail_vals = []
    V_V_tail_vals = []
    V_D_tail_vals = []

    while t < T:

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK23", rtol=1e-6, atol=1e-9,
                        max_step=dt,args=(kappa_init,I,epsilon_p,))

        ODE_state = sols.y[:, -1]

        A_V = ODE_state[0:N_muscular]
        A_D = ODE_state[N_muscular:2*N_muscular]
        V_V = ODE_state[2*N_muscular: 2*N_muscular + N_controls]
        V_D = ODE_state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls]

    
        A_V_head = A_V[0]
        V_V_head = V_V[0]
        V_D_head = V_D[0]

        A_V_mid = A_V[N_muscular//2]
        V_V_mid = V_V[N_controls//2]
        V_D_mid = V_D[N_controls//2]


        A_V_tail = A_V[-1]
        V_V_tail = V_V[-1]
        V_D_tail = V_D[-1]


        A_V_mid_vals.append(A_V_mid)
        V_V_mid_vals.append(V_V_mid)
        V_D_mid_vals.append(V_D_mid)


        A_V_head_vals.append(A_V_head)
        V_V_head_vals.append(V_V_head)
        V_D_head_vals.append(V_D_head)

        A_V_tail_vals.append(A_V_tail)
        V_V_tail_vals.append(V_V_tail)
        V_D_tail_vals.append(V_D_tail)

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
        
    A_V_vals = np.array([A_V_head_vals, A_V_mid_vals, A_V_tail_vals])
    V_V_vals = np.array([V_V_head_vals, V_V_mid_vals, V_V_tail_vals])
    V_D_vals = np.array([V_D_head_vals, V_D_mid_vals, V_D_tail_vals])
    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T, A_V_vals, V_V_vals, V_D_vals



if __name__ == "__main__":
    tick = time.time()
    epsilon_p_vals = np.arange(0.09, 1.19, 0.1)
    print("epsilon_p_vals: ", epsilon_p_vals)
    print(len(epsilon_p_vals))
    I_vals = np.logspace(-3, 0, 10)
    print("I_vals: ", I_vals)
    

    wavelengths = np.full((len(epsilon_p_vals), len(I_arrays[-1])), np.nan, dtype=float)
    frequencies = np.full((len(epsilon_p_vals), len(I_arrays[-1])), np.nan, dtype=float)
    max_curvatures = np.full((len(epsilon_p_vals), len(I_arrays[-1])), np.nan, dtype=float)

    for i, epsilon_p in enumerate(epsilon_p_vals):
        print("eps: ", epsilon_p)
            
        for j, I in enumerate(I_vals):
            print("I: ", I)
            worm_positions, curvatures, A_V, V_V, V_D = simulation(I, epsilon_p)
            wave, freq = lin_reg_wavelength(curvatures[:,int(-20/dt):], t_eval[int(-20/dt):], N, N_controls)

            max_curv = np.max(curvatures)

            wavelengths[i,j] = wave
            frequencies[i,j] = freq
            max_curvatures[i,j] = max_curv

            wave_h, freq_h = Hilbert_Transform(curvatures, N, N_controls, t_eval)

            anim = create_worm_animation(worm_positions[int(-10/dt):,:,:], dt)
            filename = (anim_dir / f"{SCRIPT_NAME}_eps{epsilon_p:.2f}_I{I:.4f}.mp4")
            anim.save(filename, writer="ffmpeg", fps = int(round(1/dt)), dpi=150)
            plt.close("all")


            print("freq: ", freq)
            print("wave: ", wave)
            print("max curve: ", np.round(max_curv, 2))
            print("wave_h: ", wave_h)
            print("freq_h: ", freq_h)

            k_head = curvatures[0, int(-20/dt):]
            k_mid = curvatures[N//2, int(-20/dt):]
            k_tail = curvatures[-1, int(-20/dt):]

            dkdt_head = np.gradient(k_head, dt)        
            dkdt_mid = np.gradient(k_mid, dt)
            dkdt_tail = np.gradient(k_tail, dt)


            A_V_head = A_V[0]
            A_V_mid = A_V[1]
            A_V_tail = A_V[2]

            V_V_head = V_V[0]
            V_V_mid = V_V[1]
            V_V_tail = V_V[2]

            V_D_head = V_D[0]
            V_D_mid = V_D[1]
            V_D_tail = V_D[2]

            davdt_head = np.gradient(A_V_head[int(-20/dt):], dt)
            davdt_mid = np.gradient(A_V_mid[int(-20/dt):], dt)
            davdt_tail = np.gradient(A_V_tail[int(-20/dt):], dt)

            dVvdt_head = np.gradient(V_V_head[int(-20/dt):], dt)
            dVvdt_mid = np.gradient(V_V_mid[int(-20/dt):], dt)
            dVvdt_tail = np.gradient(V_V_tail[int(-20/dt):], dt)

            V_diff_head = V_V_head[int(-20/dt):] - V_D_head[int(-20/dt):]
            V_diff_mid = V_V_mid[int(-20/dt):] - V_D_mid[int(-20/dt):]
            V_diff_tail = V_V_tail[int(-20/dt):] - V_D_tail[int(-20/dt):]

            #---------Kappa vs V diff--------------
            plt.figure()
            plt.plot(V_diff_head, k_head, '--', label='Head', color='blue', )
            plt.plot(V_diff_mid, k_mid, '--',label='Middle', color='green')
            plt.plot(V_diff_tail, k_tail, '--',label='Tail', color='red')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xlabel(r'$V_\text{V} - V_\text{D}$', size=12)
            plt.ylabel(r'$\kappa$', size=12)
            plt.title(f'Phase portrait of kappa vs V diff for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
            plt.tight_layout()
            plt.savefig(kappavdiff_dir / f"vdiff_kappa_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

            #---------Kappa vs dKappa/dt--------------
            plt.figure()
            plt.plot(k_head, dkdt_head, '--', label='Head', color='blue')
            plt.plot(k_mid, dkdt_mid, '--', label='Middle', color='green')
            plt.plot(k_tail, dkdt_tail, '--', label='Tail', color='red')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xlabel(r'$\kappa$', size=12)
            plt.ylabel(r'$\frac{d\kappa}{dt}$', size=12)
            plt.title(f'Phase portrait of kappa vs dkappa/dt for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
            plt.tight_layout()
            plt.savefig(kappadkappadt_dir / f"kappa_dkappa_dt_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)


            #--------Kappa vs time--------------
            plt.figure()
            plt.plot(t_eval[int(-20/dt):], k_head, '--', label='Head', color='blue')
            plt.plot(t_eval[int(-20/dt):], k_mid, '--', label='Middle', color='green')
            plt.plot(t_eval[int(-20/dt):], k_tail, '--', label='Tail', color='red')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xlabel(r'$t$', size=12) 
            plt.ylabel(r'$\kappa$', size=12)
            plt.title(f'Kappa vs time for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
            plt.tight_layout()
            plt.savefig(kappatime_dir / f"kappa_time_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

            #---------A_V vs dA_V/dt--------------
            plt.figure()
            plt.plot(A_V_head[int(-20/dt):], davdt_head, '--', label='Head', color='blue')
            plt.plot(A_V_mid[int(-20/dt):], davdt_mid, '--', label='Middle', color='green')
            plt.plot(A_V_tail[int(-20/dt):], davdt_tail, '--', label='Tail', color='red')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xlabel(r'$A_V$', size=12)
            plt.ylabel(r'$\frac{dA_V}{dt}$', size=12)
            plt.title(f'Phase portrait of A_V vs dA_V/dt for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
            plt.tight_layout()
            plt.savefig(AVdAVdt_dir / f"AV_dAV_dt_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

            #---------A_V vs time--------------
            plt.figure()
            plt.plot(t_eval[int(-20/dt):], A_V_head[int(-20/dt):], '--', label='Head', color='blue')
            plt.plot(t_eval[int(-20/dt):], A_V_mid[int(-20/dt):], '--', label='Middle', color='green')
            plt.plot(t_eval[int(-20/dt):], A_V_tail[int(-20/dt):], '--', label='Tail', color='red')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xlabel(r'$t$', size=12)
            plt.ylabel(r'$A_V$', size=12)
            plt.title(f'A_V vs time for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
            plt.tight_layout()
            plt.savefig(AVtime_dir / f"AV_time_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)

            #---------V_V vs dV_V/dt--------------
            plt.figure()
            plt.plot(V_V_head[int(-20/dt):], dVvdt_head, '--', label='Head', color='blue')
            plt.plot(V_V_mid[int(-20/dt):], dVvdt_mid, '--', label='Middle', color='green')
            plt.plot(V_V_tail[int(-20/dt):], dVvdt_tail, '--', label='Tail', color='red')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xlabel(r'$V_V$', size=12)
            plt.ylabel(r'$\frac{dV_V}{dt}$', size=12)
            plt.title(f'Phase portrait of V_V vs dV_V/dt for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
            plt.tight_layout()
            plt.savefig(VVdVVdt_dir / f"VV_dVV_dt_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)
            

            #---------V_V vs time--------------
            plt.figure()
            plt.plot(t_eval[int(-20/dt):], V_V_head[int(-20/dt):], '--', label='Head', color='blue')
            plt.plot(t_eval[int(-20/dt):], V_V_mid[int(-20/dt):], '--', label='Middle', color='green')
            plt.plot(t_eval[int(-20/dt):], V_V_tail[int(-20/dt):], '--', label='Tail', color='red')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xlabel(r'$t$', size=12)
            plt.ylabel(r'$V_V$', size=12)
            plt.title(f'V_V vs time for eps={epsilon_p:.2f} and I={I:.4f}', size=12)
            plt.tight_layout()
            plt.savefig(VVtime_dir / f"VV_time_eps{epsilon_p:.2f}_I{I:.4f}.png", dpi=150)
            plt.close("all")



    #-----------------Heat Map plots-------------
    cmap = plt.cm.bwr.copy()
    cmap.set_bad('black')

    plt.figure()
    plt.imshow(wavelengths, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    cbar = plt.colorbar(label=r'Wavelength')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of wavelength for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"wavelengths_heatmap.png", dpi=150)

    cmap = plt.cm.bwr.copy()
    cmap.set_bad('black')

    plt.figure()
    plt.imshow(frequencies, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of frequency for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"frequencies_heatmap.png", dpi=150)


    cmap = plt.cm.bwr.copy()
    cmap.set_bad('black')

    plt.figure()
    plt.imshow(max_curvatures, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of maximum curvature for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"max_curvatures_heatmap.png", dpi=150)