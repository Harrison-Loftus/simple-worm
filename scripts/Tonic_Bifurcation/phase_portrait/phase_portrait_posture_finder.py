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


from pathlib import Path
import time

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

BASE_DIR = OUTPUT_DIR / SCRIPT_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)


from scripts.Johnson_Ranner.ODEs import *

T = 30.0  # Final time - recommend several undulations

dt = 1.0e-2  # Time step - recommend ~1.0e-2 or lower

t_eval = np.arange(0,T,dt)


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
        print(np.round(t,4))

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK45",
                        max_step=dt,args=(kappa_init,))

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

    worm_positions, curvatures, A_V, V_V, V_D = example4()

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

    


    AV = A_V_mid[int(-20/dt):]
    dAV = davdt_mid

    VV_mid = V_V_mid[int(-20/dt):]

    phase = np.column_stack((AV, dAV))

    corners = np.array([
        [0.0,   0.0],
        [2.5,  80.0],
        [8.0, -80.0],
        [10.0,  0.0]
    ])

    print("AV range:", AV.min(), AV.max())
    print("dAV range:", dAV.min(), dAV.max())

    corner_inds = []

    for c in corners:
        dist = np.linalg.norm(phase - c, axis=1)
        corner_inds.append(np.argmin(dist))

    print(corner_inds)

    for c, idx in zip(corners, corner_inds):
        print(
        "target =", c,
        " found =",
        (AV[idx], dAV[idx]),
        " distance =",
        np.linalg.norm([AV[idx]-c[0], dAV[idx]-c[1]])
        )


    offset = len(t_eval) - len(AV)

    corner_inds_full = [i + offset for i in corner_inds]

    corner_worms = [worm_positions[i] for i in corner_inds_full]

    fig, axs = plt.subplots(2, 2, figsize=(8,8))

    labels = [
        r'$(A_V,\dot A_V)\approx(0,0)$',
        r'$(A_V,\dot A_V)\approx(2.5,80)$',
        r'$(A_V,\dot A_V)\approx(8,-80)$',
        r'$(A_V,\dot A_V)\approx(10,0)$'
    ]

    for ax, worm, lab in zip(axs.ravel(), corner_worms, labels):

        ax.plot(worm[:,0], worm[:,2], 'k', lw=2)
        ax.set_aspect('equal')
        ax.set_title(lab)

    plt.tight_layout()
    plt.show()


    #---------Kappa vs V diff--------------
    plt.figure()
    plt.plot(V_diff_head, k_head, label='Head', color='blue', )
    plt.plot(V_diff_mid, k_mid,label='Middle', color='green')
    plt.plot(V_diff_tail, k_tail, '--',label='Tail', color='red')
    plt.plot(V_diff_mid[corner_inds], k_mid[corner_inds], 'o', color='yellow')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$V_\text{V} - V_\text{D}$', size=12)
    plt.ylabel(r'$\kappa$', size=12)
    plt.title(f'Phase portrait of kappa vs V diff for dt={dt:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"vdiff_kappa_dt{dt:.4f}.png", dpi=150)

    #---------Kappa vs dKappa/dt--------------
    plt.figure()
    plt.plot(k_head, dkdt_head, '--', label='Head', color='blue')
    plt.plot(k_mid, dkdt_mid, label='Middle', color='green')
    plt.plot(k_tail, dkdt_tail, '--', label='Tail', color='red')
    plt.plot(k_mid[corner_inds], dkdt_mid[corner_inds], 'o', color='yellow')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$\kappa$', size=12)
    plt.ylabel(r'$\frac{d\kappa}{dt}$', size=12)
    plt.title(f'Phase portrait of kappa vs dkappa/dt for dt={dt:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"kappa_dkappa_dt_dt{dt:.4f}.png", dpi=150)


    #--------Kappa vs time--------------
    plt.figure()
    plt.plot(t_eval[int(-20/dt):], k_head, label='Head', color='blue')
    plt.plot(t_eval[int(-20/dt):], k_mid, label='Middle', color='green')
    plt.plot(t_eval[int(-20/dt):], k_tail, '--', label='Tail', color='red')
    plt.plot(t_eval[int(-20/dt):][corner_inds], k_mid[corner_inds], 'o', color='yellow')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$t$', size=12) 
    plt.ylabel(r'$\kappa$', size=12)
    plt.title(f'Kappa vs time for dt={dt:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"kappa_time_dt{dt:.4f}.png", dpi=150)


    print(AV[corner_inds], dAV[corner_inds])
    #---------A_V vs dA_V/dt--------------
    plt.figure()
    plt.plot(A_V_head[int(-20/dt):], davdt_head, label='Head', color='blue')
    plt.plot(A_V_mid[int(-20/dt):], davdt_mid, label='Middle', color='green')
    plt.plot(A_V_tail[int(-20/dt):], davdt_tail, '--', label='Tail', color='red')
    plt.plot(AV[corner_inds], dAV[corner_inds], 'o', color='yellow')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$A_V$', size=12)
    plt.ylabel(r'$\frac{dA_V}{dt}$', size=12)
    plt.title(f'Phase portrait of A_V vs dA_V/dt for dt={dt:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"AV_dAV_dt_dt{dt:.4f}.png", dpi=150)

    #---------A_V vs time--------------
    plt.figure()
    plt.plot(t_eval[int(-20/dt):], A_V_head[int(-20/dt):], label='Head', color='blue')
    plt.plot(t_eval[int(-20/dt):], A_V_mid[int(-20/dt):], label='Middle', color='green')
    plt.plot(t_eval[int(-20/dt):], A_V_tail[int(-20/dt):], '--', label='Tail', color='red')
    plt.plot(t_eval[int(-20/dt):][corner_inds], AV[corner_inds], 'o', color='yellow')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$t$', size=12)
    plt.ylabel(r'$A_V$', size=12)
    plt.title(f'A_V vs time for dt={dt:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"AV_time_dt{dt:.4f}.png", dpi=150)

    #---------V_V vs dV_V/dt--------------
    plt.figure()
    plt.plot(V_V_head[int(-20/dt):], dVvdt_head, label='Head', color='blue')
    plt.plot(V_V_mid[int(-20/dt):], dVvdt_mid, label='Middle', color='green')
    plt.plot(V_V_tail[int(-20/dt):], dVvdt_tail, '--', label='Tail', color='red')
    plt.plot(VV_mid[corner_inds], dVvdt_mid[corner_inds], 'o', color='yellow')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$V_V$', size=12)
    plt.ylabel(r'$\frac{dV_V}{dt}$', size=12)
    plt.title(f'Phase portrait of V_V vs dV_V/dt for dt={dt:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"VV_dVV_dt_dt{dt:.4f}.png", dpi=150)
    

    #---------V_V vs time--------------
    plt.figure()
    plt.plot(t_eval[int(-20/dt):], V_V_head[int(-20/dt):], label='Head', color='blue')
    plt.plot(t_eval[int(-20/dt):], V_V_mid[int(-20/dt):], label='Middle', color='green')
    plt.plot(t_eval[int(-20/dt):], V_V_tail[int(-20/dt):], '--', label='Tail', color='red')
    plt.plot(t_eval[int(-20/dt):][corner_inds], VV_mid[corner_inds], 'o', color='yellow')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.xlabel(r'$t$', size=12)
    plt.ylabel(r'$V_V$', size=12)
    plt.title(f'V_V vs time for dt={dt:.4f}', size=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"VV_time_dt{dt:.4f}.png", dpi=150)
    plt.close("all")
