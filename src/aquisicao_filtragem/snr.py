from scipy.stats import kurtosis, skew
import numpy as np

class SNR:
    @classmethod
    def calc_snr(cls,raw_signal,filtered_signal):
        noise = raw_signal - filtered_signal

        var_filtered_signal = np.var(filtered_signal)
        var_noise = np.var(noise)

        if var_noise == 0:
            return np.inf
        
        snr = 10* np.log10(var_filtered_signal/var_noise) #db

        return snr