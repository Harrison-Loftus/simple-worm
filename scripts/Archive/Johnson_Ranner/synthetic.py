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
import random

from scipy.signal import find_peaks

from pathlib import Path
import time

# Outputs directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    global N
    N = 96
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
    
    # wave parameters
    Amp = 10.0
    lam = 0.66
    omega = 1.0

    # specific forcing function
    def alpha_forcing(t, j):
        
        # j is point in numpy array
        # j_control is the corresponding control point
        j_control = (j * N_controls) // N
        
        # u_control is center point of control region
        u_control = (j_control + 0.5) / N_controls
        
        return Amp * np.sin(2.0 * np.pi / lam * u_control - 2 * np.pi * omega * t)

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
        t += dt
        print(np.round(t,2))

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

        prominence = 0.5
        

        if t >= T:

            prev_curvatures.append(curvature_np[mid_idx])
            prev_times.append(t)

            if times[-1] >= 20.0 and len(peak_times) == 0:
                break


            if len(prev_curvatures) >= 15:

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    

    view_worm_pyqtgraph(worm_positions, dt)
    #view_curvature_pyqtgraph(curvatures, dt)
