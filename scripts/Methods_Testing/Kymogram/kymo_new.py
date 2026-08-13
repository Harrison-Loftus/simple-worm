import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def lin_reg_wavelength(kappa, t_eval, n, n_controls):

    control_indices = np.arange(0, n, n // n_controls)

    kappa_reduced = kappa[control_indices, :]   

    def finding_peaks(kappa_reduced, t_eval, n_controls):
        time_peaks = []
        kappa_peaks = []
        times = []
        
        for i in range(n_controls):
            peaks = find_peaks(kappa_reduced[i, :], height=0.1)
            peaks_indx = peaks[0]
            time_peaks.append(peaks_indx)
            a = []
            b = []
            for j in range(len(peaks_indx)):
                a.append(kappa_reduced[i, peaks_indx[j]])
                b.append(t_eval[peaks_indx[j]])
            kappa_peaks.append(a)
            times.append(b)
        return times, kappa_peaks

    times, kappa_peaks = finding_peaks(kappa_reduced, t_eval, n_controls)
    
    print(len(times[0]))

    freqs = []
    for i in range(kappa_reduced.shape[0]):
        peaks, _ = find_peaks(kappa_reduced[i, :], height=0.1)
        peak_times = t_eval[peaks]
        periods = np.diff(peak_times)
        freqs.append(1.0 / np.mean(periods))
    
    freq = np.mean(freqs)

    heights = [(i + 0.5) / n_controls for i in range(n_controls)]
    wavelengths = []
    tracks = []
    
    if np.isnan(freq):
        return np.nan, np.nan

    n_cycles = int(freq * (t_eval[-1] - t_eval[0]))
    print("N cycles: ", n_cycles)
    n_peaks = max(0, n_cycles // 3)
    print(n_peaks)
    print(len(times[0][:n_cycles]))

    for start_peak in times[0][:n_peaks]:
        selection_time = [start_peak]    
        selection_curves = [] 

        for i in range(1, n_controls):
            curv_list = np.array(kappa_peaks[i])
            t_list = np.array(times[i])

            
            diffs = np.abs(t_list - selection_time[-1])

            valid = np.where(t_list >= selection_time[-1])[0]

            if len(valid) == 0:
                break

            idx = valid[0]

            
            
            selection_curves.append(curv_list[idx])
            selection_time.append(t_list[idx])

        x = selection_time
        y = heights

        m, c = np.polyfit(x,y,1)

        wavelength = abs(m) / freq
        wavelengths.append(wavelength)

        plt.figure()
        plt.imshow(kappa, aspect='auto', extent=[t_eval[0], t_eval[-1], 0 , 1], origin='lower', cmap='bwr')
        cbar = plt.colorbar(label=r'Curvature (mm$^{-1}$)')
        cbar.ax.yaxis.label.set_size(12)
        plt.xlabel('Time (s)', size=12)
        plt.ylabel('Body length (mm)', size=12)
        plt.title('Kymograph of a Recovered C. elegans in Agar', size=16)
        plt.plot(x, y, 'bo')
        plt.show()
                
    norm_wave = np.mean(wavelengths)
    return norm_wave, freq