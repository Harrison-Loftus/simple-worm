import numpy as np
from scipy.signal import hilbert
from scipy.signal import find_peaks


def Hilbert_Transform(kappa, n, N_controls, t_eval):
   #-------------------Wavelength Calc------------------
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
    lam_over_L = 1.0 / (N_controls * np.mean(1 - phi_arr))
    wavelengths_H = (lam_over_L)

    #-------------------Frequency Calc------------------
    freqs = []
    for i in range(kappa.shape[0]):
        peaks, _ = find_peaks(kappa[i, :])
        peak_times = t_eval[peaks]
        periods = np.diff(peak_times)

        freqs.append(1.0 / np.mean(periods))
        frequency_H = np.mean(freqs)

    return wavelengths_H, frequency_H