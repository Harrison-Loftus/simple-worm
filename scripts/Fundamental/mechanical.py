from numpy.linalg import norm
from dolfinx import fem
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

from pathlib import Path

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PAPER_DIR = Path(OUTPUT_DIR, "FOR_PAPER")
PAPER_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem



def example4(config, ODEs):
    """
    This example shows how to call the simulator with a fenics function
    for forcing with a different set of material parameters
    """
    # holders for 'worm', u and control
    # Parameters

    N = config.N
    N_muscular = config.N_muscular
    N_controls = config.N_controls

    K_agar = config.K_agar
    e = config.e
    eta_tilde = config.eta_tilde


    T = config.T
    dt = config.dt
    max_time = config.max_time
    
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

    convergence_tol = config.convergence_tol      # 5%
    consecutive_required = config.consecutive_required    # require 3 converged periods

    n_converged = 0

    extra_cycles = config.extra_cycles
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
                        max_step=dt,args=(kappa_init,config))

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

        betas = A_V - A_D

        muscle_indices = np.floor(np.arange(N) * N_muscular / N).astype(int)

        betas = betas[muscle_indices]
                
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

            if t > max_time:
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

                    
                    # print(f"Peak at {peak_time:.3f}")

                    if len(peak_times) >= 2:

                        
                        period = peak_times[-1] - peak_times[-2]
                        periods.append(period)

                        # print(f"Period = {period:.3f}")

                        if len(periods) >= consecutive_required:

                            recent_periods = np.array(periods[-consecutive_required:])

                            mean_period = np.mean(recent_periods)

                            period_error = (np.max(np.abs(recent_periods - mean_period)) / mean_period)

                            # print(f"Period error = {period_error:.4f}")

                            if termination_time is None and period_error < 0.01:
                                termination_time = (t + extra_cycles * mean_period)

                                # print(f"Converged. " f"Stopping at {termination_time:.3f}")
                                mean_period = np.mean(periods[-consecutive_required:])

                                termination_time = (t + extra_cycles * mean_period)

                                # print(f"Converged. " f"Stopping at {termination_time:.3f}")

        if termination_time is not None and t >= termination_time:
            break

    A_V_vals = np.array([A_V_anterior_vals, A_V_mid_vals, A_V_posterior_vals])
    A_D_vals = np.array([A_D_anterior_vals, A_D_mid_vals, A_D_posterior_vals])
    V_V_vals = np.array([V_V_anterior_vals, V_V_mid_vals, V_V_posterior_vals])
    V_D_vals = np.array([V_D_anterior_vals, V_D_mid_vals, V_D_posterior_vals])



    kappas = np.array(kappas)
    worm_positions = np.array(worm_positions)
    return worm_positions, kappas.T, np.array(times), peak_heights, peak_times, peak_indices, A_V_vals, A_D_vals, V_V_vals, V_D_vals
