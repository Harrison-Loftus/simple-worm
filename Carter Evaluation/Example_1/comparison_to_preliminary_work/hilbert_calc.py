import numpy as np
from scipy.signal import hilbert

def Hilbert_Transform(kappa, n):
    phase_segments = []
    for j in range(n):
        analytic_signal = hilbert(kappa[j,:])
        instantaneous_phase = np.angle(analytic_signal)
        phase_segments.append(instantaneous_phase)

    phi_j_list = []
    for j in range(n-1):
        phase_diff = np.unwrap(phase_segments[j+1])- np.unwrap(phase_segments[j])
        phi_j = np.mean(phase_diff) % (2 * np.pi) / (2 * np.pi)
        phi_j_list.append(phi_j)

    phi_arr = np.array(phi_j_list)
    lam_over_L = 1.0 / (n * np.mean(1.0 - phi_arr))
    wavelengths_H = (lam_over_L)

    return wavelengths_H