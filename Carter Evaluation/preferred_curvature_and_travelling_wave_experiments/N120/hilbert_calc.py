import numpy as np
from scipy.signal import hilbert


def Hilbert_Transform(kappa, n, N_controls):
    phase_segments = []
    for j in range(N_controls):
        analytic_signal = hilbert(kappa[int(j / N_controls * n),:])
        instantaneous_phase = np.angle(analytic_signal)
        phase_segments.append(instantaneous_phase)

    phi_j_list = []
    for j in range(N_controls-1):
        phase_diff = np.unwrap(phase_segments[j+1])- np.unwrap(phase_segments[j])
        phi_j = (phase_diff) % (2 * np.pi) / (2 * np.pi)
        phi_j = np.mean(phi_j)
        phi_j_list.append(phi_j)

    phi_arr = np.array(phi_j_list)
    lam_over_L = (N_controls - 1) / (N_controls * np.sum(1 - phi_arr))
    wavelengths_H = (lam_over_L)

    return wavelengths_H