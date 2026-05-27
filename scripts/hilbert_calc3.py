import numpy as np
from scipy.signal import hilbert
from scipy.signal import find_peaks


def Hilbert_Transform(kappa, n, N_controls, t_eval):
   #-------------------Wavelength Calc------------------

    l = 1 / N_controls   # module spacing
    indices = [
        int((j + 0.5) * n / N_controls)
        for j in range(N_controls)
        ]
    phase_segments = []
    for idx in indices:
        analytic_signal = hilbert(kappa[idx, :])
        phase = np.unwrap(np.angle(analytic_signal)) / (2*np.pi)
        phase_segments.append(phase)

    phi_j_list = []

    for j in range(N_controls - 1):
        diff = phase_segments[j+1] - phase_segments[j]

        
        diff = (diff + 0.5) % 1 - 0.5
        # average over time
        phi_j = np.mean(diff)
        
        phi_j_list.append(phi_j)

    phi_star = np.mean(phi_j_list)

    wavelengths_H = l / (1 - phi_star)    
    
    
    #-------------------Frequency Calc------------------
    freqs = []
    for i in range(kappa.shape[0]):
        peaks, _ = find_peaks(kappa[i, :])
        peak_times = t_eval[peaks]
        periods = np.diff(peak_times)
        freqs.append(1.0 / np.mean(periods))
    
    frequency_H = np.mean(freqs)

    return wavelengths_H, frequency_H