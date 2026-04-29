import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

class pca:
    def process(self, data, fs):
        ecg_df = self.differentiate(data)
        ecg_ma = self.movingavarage(ecg_df)
        peaks = self.qrspeaks(ecg_ma, fs)
        janela = self.window(data, peaks, fs, 0.2, 0.4)
        matrix, x = self.obersvartion_matrix(janela)  
        pca = PCA(n_components=5)
        z = pca.fit_transform(matrix)
        return z, pca, x
#After the filtering process, the ECG signal was differentiated. 
# The derivative enhance the steepness or slope of the QRS complex,
#  because differentiating a curve enhances rapid changes and sharp edges.

#The signal was squared after the derivative stage to ensure 
# that all signal components exhibited positive values
    def differentiate(self, data):
        ECG_df = np.diff(data)
        ECG_sq = np.power(ECG_df, 2)
        return np.insert(ECG_sq, 0, ECG_sq[0]) #manter o tamanho do array
#A moving average window was applied to the signal to merge the peaks 
#from the previous stage together and cancel sharp edges to facilitate QRS detection
    def movingavarage(self, data, n=30):
        window = np.ones((1, n))/n
        ecg_ma = np.convolve(np.squeeze(data), np.squeeze(window), mode="same")
        return ecg_ma
    
#Detection of peaks
    def qrspeaks(self, data, fs):
        peaks, _ = find_peaks(data, height=np.mean(data), distance=round(fs*0.200))
        return peaks
# extraction of a window around each peak R to ensure that all segments have the same length.
    def window(self, data, peaks, fs,  n_before, n_after):
        n_before = int(n_before*fs)
        n_after = int(n_after*fs)
        segments = []
        for r in peaks:
            start = r -n_before
            end = r+n_after
            if start >= 0 and end<=len(data):
                beat = data[start:end]
                segments.append(beat)
        return np.array(segments)

    def obersvartion_matrix(self, janela):
        print(janela.shape)
        X = np.vstack(janela)
        print(X.shape)
        X_centered = X - X.mean(axis=0)  
        return X_centered, X

