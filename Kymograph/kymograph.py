import numpy as np
from scipy.signal import find_peaks

def finding_peaks(kappa, t_eval, n):
    time_peaks = []
    kappa_peaks = []
    times = []
    
    for i in range(n):
        peaks = find_peaks(kappa[i, :], height=0.1, distance=20.0/(n/6.0))
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


def lin_reg_wavelength(kappa_peaks, times, n):
    selection_curves = [kappa_peaks[0][0]]
    heights = [(i + 0.5) / n for i in range(n)]
    selection_time = [times[0][0]]

    for i in range(1, n):
        curv_list = np.array(kappa_peaks[i])
        t_list = np.array(times[i])

        diffs = np.abs(t_list - selection_time[-1])
        idx = np.argmin(diffs)
        
        selection_curves.append(curv_list[idx])
        selection_time.append(t_list[idx])

    x = selection_time
    y = heights
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_less_mean = [i - x_mean for i in x]
    y_less_mean = [i - y_mean for i in y]
    x_less_mean_sqrd = [i**2 for i in x_less_mean]
    y_less_mean_mult_x_less_mean = []
    for i in range(len(y_less_mean)):
        a = y_less_mean[i] * x_less_mean[i]
        y_less_mean_mult_x_less_mean.append(a)

    Beta1 = np.sum(y_less_mean_mult_x_less_mean) / np.sum(x_less_mean_sqrd)
    Beta0 = y_mean - (Beta1 * x_mean)

    original_height = 0.5 / n
    wavelength_time = times[0][1]
    wavelength_height = Beta0 + Beta1 * wavelength_time

    wavelength = np.abs(wavelength_height - original_height)
    return wavelength, selection_time