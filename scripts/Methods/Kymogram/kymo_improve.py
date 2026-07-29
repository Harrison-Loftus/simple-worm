import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt



def lin_reg_wavelength(kappa, t_eval, n):
    def finding_peaks(kappa, t_eval, n):
        time_peaks = []
        kappa_peaks = []
        times = []
        
        for i in range(n):
            peaks = find_peaks(kappa[i, :], height=0.1)
            peaks_indx = peaks[0]
            time_peaks.append(peaks_indx)
            a = []
            b = []
            for j in range(len(peaks_indx)):
                a.append(kappa[i, peaks_indx[j]])
                b.append(t_eval[peaks_indx[j]])
            kappa_peaks.append(a)
            times.append(b)
        return times, kappa_peaks

    times, kappa_peaks = finding_peaks(kappa, t_eval, n)

    def frequency_calc(times, n):
        freqs = []
        for i in range(n):
            for j in range(len(times[i])-1):
                    a = times[i][j+1] - times[i][j]
                    freqs.append(1/a)
        return np.mean(freqs)

    heights = [(i + 0.5) / n for i in range(n)]
    wavelengths = []
    tracks = []
    freq = frequency_calc(times, n)
    for start_peak in times[0][:int((t_eval[0] - t_eval[-1])/(freq * 2))]:
        selection_time = [start_peak]    
        selection_curves = [] 

        for i in range(1, n):
            curv_list = np.array(kappa_peaks[i])
            t_list = np.array(times[i])

            
            

            diffs = np.abs(t_list - selection_time[-1])
            idx = np.argmin(diffs)

            # stop tracking if nearest peak is too far away
            

            selection_curves.append(curv_list[idx])
            selection_time.append(t_list[idx])

        
    

        x = selection_time
        y = heights

        plt.figure()
        plt.imshow(kappa, aspect='auto', extent=[t_eval[0], t_eval[-1], 0 , 1], origin='lower', cmap='bwr')
        cbar = plt.colorbar(label=r'Curvature (mm$^{-1}$)')
        cbar.ax.yaxis.label.set_size(12)
        plt.xlabel('Time (s)', size=12)
        plt.ylabel('Body length (mm)', size=12)
        plt.title('Kymograph of a Recovered C. elegans in Agar', size=16)
        plt.plot(x, y, 'bo')
        plt.show()
        

        m, c = np.polyfit(x,y,1)

        wavelength = abs(m) / freq
        wavelengths.append(wavelength)

    norm_wave = np.mean(wavelengths)
    return norm_wave, freq