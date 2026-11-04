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

from scripts import matplotlib_animate_worm
from scripts.kymograph import *
from scripts.hilbert_calc import *
from scripts.simple_worm_viewer_pyqtgraph import view_worm_pyqtgraph
from scripts.curvature_viewer import view_curvature_pyqtgraph

from scripts.Johnson_Ranner.ODEs import *

from pathlib import Path
import time

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PAPER_DIR = Path(OUTPUT_DIR, "FOR_PAPER")
PAPER_DIR.mkdir(exist_ok=True)

SCRIPT_NAME = Path(__file__).stem

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

    peak_indices = []
    
    while True:

        sols = solve_ivp(ODEs, (t, t+dt), ODE_state, method="RK45",
                        max_step=dt,args=(kappa_init,))

        ODE_state = sols.y[:, -1]

        A_V = ODE_state[0:N_muscular]
        A_D = ODE_state[N_muscular:2*N_muscular]
        V_V = ODE_state[2*N_muscular: 2*N_muscular + N_controls]
        V_D = ODE_state[2*N_muscular + N_controls: 2*N_muscular + 2*N_controls]

            


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
    tick = time.time()

    worm_positions, curvatures, times, peak_heights, peak_times, peak_indices = example4()

    com_positions = np.mean(worm_positions[:, :, [0, 2]], axis=1)

    start_idx = peak_indices[-4]
    end_idx = peak_indices[-1]

    displacement = (com_positions[end_idx] - com_positions[start_idx])

    delta_t = times[end_idx] - times[start_idx]

    mean_velocity = np.linalg.norm(displacement) / delta_t

    print("Mean forward speed =", mean_velocity, "mm/s")

    print("Mean forward speed =", mean_velocity / L, "body lengths/s")


    final_peaks = peak_indices[-3:]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))


    #-------------Final three postures subplots----------------
    for ax, idx in zip(axes, final_peaks):

        posture = worm_positions[idx]

        x = posture[:, 0]
        y = posture[:, 2]

        ax.plot(x, y, 'k', lw=2)

        ax.set_aspect('equal')
        ax.set_title(f't = {times[idx]:.2f}s', size=12)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / f"{SCRIPT_NAME}_final_postures.png", dpi=300)


    #-------------Final three postures overlay----------------
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
    plt.title('Final three undulations', size=16)
    plt.savefig(PAPER_DIR / f"{SCRIPT_NAME}_final_postures_overlay.png", dpi=300)



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


    #-------------Final three postures aligned----------------
    plt.figure(figsize=(6,6))

    for idx, c in zip(final_peaks, colours):
        posture = worm_positions[idx].copy()
        posture -= posture[0]
        plt.plot(posture[:,0], posture[:,2], color=c, lw=2, label=f't={times[idx]:.2f}s')

    plt.legend()
    plt.axis('equal')
    plt.xlabel(r'$x$', size=12)
    plt.ylabel(r'$y$', size=12)
    plt.title('Final three undulations (aligned)', size=16)
    plt.savefig(PAPER_DIR / f"{SCRIPT_NAME}_final_postures_aligned.png", dpi=300)



    kappa1 = curvatures[:, final_peaks[0]]
    kappa2 = curvatures[:, final_peaks[1]]
    kappa3 = curvatures[:, final_peaks[2]]

    E12 = np.linalg.norm(kappa1 - kappa2) / np.linalg.norm(kappa1)
    E13 = np.linalg.norm(kappa1 - kappa3) / np.linalg.norm(kappa1)
    E23 = np.linalg.norm(kappa2 - kappa3) / np.linalg.norm(kappa2)

    print("Relative error between final 3 postures:")
    print(f"E12 = {E12:.4f}")
    print(f"E13 = {E13:.4f}")
    print(f"E23 = {E23:.4f}")


    #-------------Final undulation, different phases of cycle----------------
    t_start = peak_times[-2]
    t_end = peak_times[-1]

    phases = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    phase_times = t_start + phases * (t_end - t_start)

    phase_indices = [np.argmin(np.abs(times - phase_time)) for phase_time in phase_times]

    colours = ['r', 'g', 'b', 'c', 'm']


    fig, ax = plt.subplots(figsize=(8, 5))

    for phase, idx, colour in zip(phases, phase_indices, colours):

        posture = worm_positions[idx]

        x = posture[:, 0]
        y = posture[:, 2]

        ax.plot(
            x, y,
            color=colour,
            linewidth=2,
            label=fr'$\phi = {phase:.2f}$'
        )

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
    ax.set_title('Final undulation cycle')

    plt.tight_layout()
    plt.savefig(
        PAPER_DIR / f"{SCRIPT_NAME}_final_cycle_phases.png",
        dpi=300,
        bbox_inches='tight'
    )
    

    #-------------Kymograph----------------
    plt.figure()
    plt.imshow(curvatures, aspect='auto', extent=[0,times[-1], 0 , L], origin='lower', cmap='bwr')
    cbar = plt.colorbar(label=r'Curvature (mm$^{-1}$)')
    cbar.ax.yaxis.label.set_size(12)
    plt.xlabel('Time (s)', size=12)
    plt.ylabel('Body length (mm)', size=12)
    plt.title('Kymogram of a simulated $\\mathit{C.\\ elegans}$ \n healthy forward locomotion in Agar', size=16)
    plt.savefig(
        PAPER_DIR / f"{SCRIPT_NAME}_kymograph.png",
        dpi=300,
        bbox_inches="tight"
    )
    



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
    print("Wavelength mean =", np.mean(last3_waves))
    print("Wavelength std =", np.std(last3_waves))
    print("Wavelength CV (%) =", 100*np.std(last3_waves)/np.mean(last3_waves))



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
    
    #-------------Animation----------------
    anim = matplotlib_animate_worm.create_worm_animation(worm_positions, dt)
    anim.save(PAPER_DIR / f"{SCRIPT_NAME}_worm_animation.mp4", writer="ffmpeg", fps=100, dpi=300)