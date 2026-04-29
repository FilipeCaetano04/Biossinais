import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import neurokit2 as nk
from snr import SNR
from kurtosis_skewness import calculate_kurtosis_skewness as cks
import neurokit2 as nk
from pca import pca

def aplicar_filtros(sinal, fs):
    # Filtro Passa-Banda Butterworth - 0.5 Hz a 40 Hz
    # Remove baixas frequências (< 0.5 Hz) e ruídos de alta frequência (> 40 Hz)
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    sinal_bandpass = signal.filtfilt(b, a, sinal)
    
    # Filtro Rejeita-Faixa para Ruído de Linha
    f0 = 50.0  # Frequência a ser removida
    Q = 30.0   # Fator de qualidade
    b_notch, a_notch = signal.iirnotch(f0, Q, fs)
    sinal_filtrado = signal.filtfilt(b_notch, a_notch, sinal_bandpass)
    
    return sinal_filtrado

kurtosis_filtrado = []
skew_filtrado = []
snr_db = []
quality = []
entropy = []
info = []
segmento = []
tempo = []
ecg_raw = []
ecg_filtrado = []
pca_scores = [] 
pca_models = []

for i in range(5):
    segmento.append(i)

    # Exemplo de registro do PTB-XL
    path = r'C:\Users\leona\Biosinais\data500\00001_hr'#Tive que alterar aqui, a maneira que tava nao pegava a pasta fora de 'aquisicao-filtragem'

    record = wfdb.rdrecord(path)
    fs = record.fs  # Frequência de amostragem (esperado 500 Hz)

    # Extraindo a Derivação II
    # O PTB-XL está organizado em (I, II, III, AVR, AVL, AVF, V1-V6)
    lead_names = record.sig_name
    lead_ii_idx = lead_names.index('II')
    ecg_raw.append(record.p_signal[:, lead_ii_idx])

    # Vetor de tempo em segundos
    tempo.append(np.arange(len(ecg_raw[i])) / fs)

    ecg_filtrado.append(aplicar_filtros(ecg_raw[i], fs))
    modelo_pca = pca()
    z, pca_model, matrix = modelo_pca.process(ecg_filtrado[i], fs)
    pca_scores.append(z)
    pca_models.append(pca_model)

    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], alpha=0.7, edgecolors='k')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Mapa dos Batimentos no Espaço PCA')
    plt.grid(True)
    plt.show()

    k=2
    z_reduzido = z.copy()
    z_reduzido[:, k:] = 0
    x_reconstruido = pca_model.inverse_transform(z_reduzido)  
    tempo = np.linspace(-0.2, 0.4, matrix.shape[1])

    fig, axes = plt.subplots(3, 2, figsize=(12, 2.5 * 3))

    for idx in range(3):
        axes[idx, 0].plot(tempo, x_reconstruido[idx] + matrix.mean(axis=0), color='orange')
        axes[idx, 0].axvline(0, color='gray', linestyle=':', alpha=0.5)
        axes[idx, 0].set_title(f'Batimento {idx} — Reconstruído')
        axes[idx, 0].grid(True)

        # Original bruto
        axes[idx, 1].plot(tempo, matrix[idx], color='blue')
        axes[idx, 1].axvline(0, color='gray', linestyle=':', alpha=0.5)
        axes[idx, 1].set_title(f'Batimento {idx} — Original')
        axes[idx, 1].grid(True)
    plt.suptitle('Reconstruído (K=2) vs Original', fontsize=14)
    plt.tight_layout()
    plt.show()
    #FOR LOOP PARA QUE PEGUE VARIOS QUADROS DE 10S

    kurtosis_filtrado.append(cks.calc_kurtosis(ecg_filtrado[i]))
    skew_filtrado.append(cks.calc_skew(ecg_filtrado[i]))
    snr_db.append(SNR.calc_snr(ecg_raw[i], ecg_filtrado[i]))
    #NK:
    quality.append(nk.ecg_quality(ecg_filtrado[i], sampling_rate=500, method="zhao2018"))
    entropy_x, info_x = nk.entropy_spectral(ecg_filtrado[i],show=False)
    entropy.append(entropy_x)
    info.append(info_x)

    # 5. Gerando os Gráficos
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 1, 1)
    plt.plot(tempo[i], ecg_raw[i], color='lightgray', label='Sinal Bruto')
    plt.title(f'Sinal de ECG Bruto - Derivação II ({fs} Hz)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True)

    plt.plot(tempo[i], ecg_filtrado[i], color='red', label='Sinal Filtrado')
    plt.title('Sinal Após Filtragem')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print(f"Kurtosis: {kurtosis_filtrado}")
print(f"Skewness: {skew_filtrado}")
print(f"snr: {snr_db}")
print(f"Quality: {quality}")

#SALVANDO NUM CSV PARA ANALISE, QUADROS DE 10S:

df_stats = pd.DataFrame({'segmento': segmento, 'Kurtosis': kurtosis_filtrado, 'Skewness': skew_filtrado,
                        'snr': snr_db, 
                        'Quality': quality,
                        'Entropy': entropy })
df_stats.to_csv("../data/stats_signal.csv",index=False)
# Salvando num CSV para que a próxima etapa
for i in range(5):
    df_saida = pd.DataFrame({'Tempo_s': tempo[i], 'ECG_Bruto': ecg_raw[i], 'ECG_Filtrado': ecg_filtrado[i]})
    df_saida.to_csv(f'../data/sinal_filtrado{i+1}.csv', index=False)
    print(f"Arquivo sinal_filtrado{i+1}.csv salvo com sucesso para o Passo 2!")