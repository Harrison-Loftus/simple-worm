import numpy as np
from scipy.signal import find_peaks

def kinematic_values(kappa, t_eval, worm_positions, peak_indices, peak_times, config):
    n = config.N
    n_controls = config.N_controls

    if len(peak_indices) < 4:
        print("Not enough peaks detected for kinematic analysis.")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    def lin_reg_wavelength():

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
        
        mid = n_controls // 2

        peaks, _ = find_peaks(kappa_reduced[mid, :], height=0.1)

        peak_times = t_eval[peaks]

        print("Peak times:")
        print(peak_times)

        print("Intervals:")
        print(np.diff(peak_times))



        peaks, _ = find_peaks(kappa_reduced[mid, :], height=0.1)

        peak_times = t_eval[peaks]

        freq = 1.0 / np.mean(np.diff(peak_times[-4:]))  

        heights = [(i + 0.5) / n_controls for i in range(n_controls)]
        wavelengths = []
        
        if np.isnan(freq):
            return np.nan, np.nan

                
        start_time = peak_times[-4]
        end_time = peak_times[-1]

        for start_peak in times[0]:
            if not (start_time <= start_peak <= end_time):
                continue
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
            y = heights[:len(x)]

            m, c = np.polyfit(x,y,1)

            wavelength = abs(m) / freq
            wavelengths.append(wavelength)

        norm_wave = np.mean(wavelengths)
        return norm_wave, freq

    norm_wave, freq = lin_reg_wavelength()

    #--------velocity------------
    com_positions = np.mean(worm_positions[:, :, [0, 2]], axis=1)
    
    start_idx = peak_indices[-4]
    end_idx = peak_indices[-1]

    displacement = (com_positions[end_idx] - com_positions[start_idx])

    delta_t = t_eval[end_idx] - t_eval[start_idx]

    mean_velocity = np.linalg.norm(displacement) / delta_t

    start_idx = peak_indices[-4]
    end_idx = peak_indices[-1]

    max_curvature = np.max(
        np.abs(kappa[:, start_idx:end_idx + 1])
    )
    period = np.mean(np.diff(peak_times[-4:]))

    return norm_wave, freq, period, max_curvature, mean_velocity