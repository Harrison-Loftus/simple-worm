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

from scripts.Archive.kymograph import *
from scripts.Archive.hilbert_calc import *
from scripts.Archive.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.Archive.curvature_viewer import view_curvature_pyqtgraph

from scripts.Archive.Tonic_Bifurcation.ODEs import *
from scripts.Archive.matplotlib_animate_worm import *

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

    while t < T:

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK23", rtol=1e-6, atol=1e-9,
                        max_step=dt,args=(kappa_init,I,epsilon_p,))

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
        
    A_V_vals = np.array([A_V_anterior_vals, A_V_mid_vals, A_V_posterior_vals])
    A_D_vals = np.array([A_D_anterior_vals, A_D_mid_vals, A_D_posterior_vals])
    V_V_vals = np.array([V_V_anterior_vals, V_V_mid_vals, V_V_posterior_vals])
    V_D_vals = np.array([V_D_anterior_vals, V_D_mid_vals, V_D_posterior_vals])
    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T, A_V_vals, A_D_vals,V_V_vals, V_D_vals



if __name__ == "__main__":
    tick = time.time()
    epsilon_p_vals = np.array([1.0])


    print("epsilon_p_vals: ", epsilon_p_vals)


    # I_vals = np.arange(0.25,0.354, 0.004)
    I_vals = np.arange(0.0, 0.4, 0.01)


    print(len(epsilon_p_vals))


    print("I_vals: ", I_vals)
    

    wavelengths = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)
    frequencies = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)
    max_curvatures = np.full((len(epsilon_p_vals), len(I_vals)), np.nan, dtype=float)

    for i, epsilon_p in enumerate(epsilon_p_vals):
        print("eps: ", epsilon_p)
            
        for j, I in enumerate(I_vals):
            print("I: ", I)
            worm_positions, curvatures, A_V, A_D, V_V, V_D = simulation(I, epsilon_p)
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

            k_anterior = curvatures[N//4, :]
            k_mid = curvatures[N//2, :]
            k_posterior = curvatures[-N//4, :]

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
            plt.plot(t_eval, k_anterior, '--', label='anterior', color='blue')
            plt.plot(t_eval, k_mid, '--', label='Middle', color='green')
            plt.plot(t_eval, k_posterior, '--', label='posterior', color='red')
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
            plt.plot(t_eval, A_V_anterior, '--', label='anterior', color='blue')
            plt.plot(t_eval, A_V_mid, '--', label='Middle', color='green')
            plt.plot(t_eval, A_V_posterior, '--', label='posterior', color='red')
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
            plt.plot(t_eval, V_V_anterior, '--', label='anterior', color='blue')
            plt.plot(t_eval, V_V_mid, '--', label='Middle', color='green')
            plt.plot(t_eval, V_V_posterior, '--', label='posterior', color='red')
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

        #-----------------wavelength normal variation--------------
    plt.figure()
    plt.imshow(wavelengths, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    cbar = plt.colorbar(label=r'Wavelength')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of wavelength for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"wavelengths_heatmap.png", dpi=150)

        #-----------------wavelength logx variation--------------
    plt.figure()
    plt.imshow(wavelengths, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    cbar = plt.colorbar(label=r'Wavelength')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of wavelength for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"wavelengths_heatmap_logx.png", dpi=150)

        #------------------wavelength logxlogy variation--------------
    plt.figure()
    plt.imshow(wavelengths, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar(label=r'Wavelength')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of wavelength for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"wavelengths_heatmap_logxlogy.png", dpi=150)


        #-----------------frequency normal variation--------------
    plt.figure()
    plt.imshow(frequencies, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of frequency for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"frequencies_heatmap.png", dpi=150)

        #-----------------frequency logx variation--------------
    plt.figure()
    plt.imshow(frequencies, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of frequency for varying values\n'
                'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"frequencies_heatmap_logx.png", dpi=150)

    
        #-----------------frequency logxlogy variation--------------
    plt.figure()
    plt.imshow(frequencies, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of frequency for varying values\n'
                'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"frequencies_heatmap_logxlogy.png", dpi=150)



        #-----------------max curvature normal variation--------------
    plt.figure()
    plt.imshow(max_curvatures, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of maximum curvature for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"max_curvatures_heatmap.png", dpi=150)


        #-----------------max curvature logx variation--------------
    plt.figure()
    plt.imshow(max_curvatures, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of maximum curvature for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"max_curvatures_heatmap_logx.png", dpi=150)

        #-----------------max curvature logxlogy variation--------------
    plt.figure()
    plt.imshow(max_curvatures, aspect='auto', extent=[I_vals[0], I_vals[-1], epsilon_p_vals[0] , epsilon_p_vals[-1]], origin='lower', cmap=cmap)
    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar(label=r'Frequency Hz')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel(r'Tonic input $I$', size=12)
    plt.ylabel(r'Proprioceptive strength $\varepsilon_p$', size=12)
    plt.title('Kymograph of maximum curvature for varying values\n'
               'of tonic input and proprioceptive strength', size=16)
    plt.savefig(BASE_DIR / f"max_curvatures_heatmap_logxlogy.png", dpi=150)


    with open(BASE_DIR / f"frequencies.txt", "w") as f:
        for row in frequencies:
            f.write(",".join(map(str, row)) + "\n")

    with open(BASE_DIR / f"max_curvatures.txt", "w") as f:
        for row in max_curvatures:
            f.write(",".join(map(str, row)) + "\n")

    with open(BASE_DIR / f"wavelengths.txt", "w") as f:
        for row in wavelengths:
            f.write(",".join(map(str, row)) + "\n")