import pandas as pd
import numpy as np
import pywt
import re
from scipy.signal import find_peaks

def limpar_ecg_wavelet(sinal: np.ndarray, wavelet='db4', level=4) -> np.ndarray:
    """Limpa uma única derivação (vetor 1D) usando DWT."""
    coeffs = pywt.wavedec(sinal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    if sigma == 0: return sinal # Proteção contra linha reta (flatline)
        
    limiar = sigma * np.sqrt(2 * np.log(len(sinal)))
    coeffs_limpos = [coeffs[0]] + [pywt.threshold(c, value=limiar, mode='soft') for c in coeffs[1:]]
    sinal_limpo = pywt.waverec(coeffs_limpos, wavelet)
    
    return sinal_limpo[:len(sinal)]

# =====================================================================
# EXTRAIR PICOS R E INTERVALOS
# =====================================================================
def extrair_features_intervalos(matriz_limpa: np.ndarray, leads: list, freq=500):
    """
    matriz_limpa: Array numpy (1000, 12) já limpo pela Wavelet.
    Extrai o Ritmo (RR) da derivação II e a Morfologia das demais.
    """
    features = {}
    idx_lead_II = leads.index('II') if 'II' in leads else 1
    sinal_ritmo = matriz_limpa[:, idx_lead_II]
    

    altura_minima = np.mean(sinal_ritmo) + 0.5 * np.std(sinal_ritmo)
    picos_R, _ = find_peaks(sinal_ritmo, distance=int(freq * 0.4), height=altura_minima)
    
    features['Batimentos_Segmento'] = len(picos_R)
    
    if len(picos_R) > 1:
        intervalos_rr = np.diff(picos_R) / freq
        features['RR_Medio'] = np.mean(intervalos_rr)
        features['RR_Std'] = np.std(intervalos_rr)  
        features['RR_Max'] = np.max(intervalos_rr)
        features['RR_Min'] = np.min(intervalos_rr)
    else:
        # Paciente com bradicardia extrema ou artefato massivo (menos de 2 batimentos em 2s)
        features['RR_Medio'] = 0; features['RR_Std'] = 0
        features['RR_Max'] = 0; features['RR_Min'] = 0

    # ---------------------------------------------------------
    # EXTRAÇÃO DE MORFOLOGIA
    # ---------------------------------------------------------
    for i, lead in enumerate(leads):
        sinal = matriz_limpa[:, i]
        # PtP (Pico a Pico) - Mede a largura vertical do QRS Limpo
        features[f'PtP_Limpo_{lead}'] = np.ptp(sinal)
        # RMS - Força real do sinal elétrico na derivação
        features[f'RMS_Limpo_{lead}'] = np.sqrt(np.mean(sinal ** 2))
        
    return features

# =====================================================================
# GERAR O DATASET FINAL COM AS FEATURES
# =====================================================================
def gerar_features_clinicas(df_raw: pd.DataFrame, df_not_bad_data: pd.DataFrame, freq=500):
    """Usa os dado brutos, limpa e extrai features apenas dos válidos."""
    amostras_por_segmento = int(2 * freq) 
    LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    print("🧠 Agrupando o Gigante (df_raw) na memória...")
    grouped_raw = df_raw.groupby('ecg_id')
    col_label = 'label_clinico' if 'label_clinico' in df_not_bad_data.columns else 'label'
    mapa_aprovados = df_not_bad_data[['ecg_id', 'segment_id', col_label]].drop_duplicates()
    
    lista_features = []
    
    for _, row in mapa_aprovados.iterrows():
        eid, sid, diag = row['ecg_id'], row['segment_id'], row[col_label]
        
        if eid in grouped_raw.groups:
            sinal_completo = grouped_raw.get_group(eid)
            
            # Matemática para o seu formato "seg_0a2s"
            num = re.findall(r'\d+', str(sid))
            if num:
                segundo_inicial = int(num[0])
                inicio = segundo_inicial * freq
                fim = inicio + amostras_por_segmento
                
                # 1. RECORTA AS 1000 LINHAS BRUTAS
                fatia_suja = sinal_completo.iloc[inicio:fim][LEADS].values
                
                if len(fatia_suja) == amostras_por_segmento:
                    # 2. DENOISING COM WAVELET 
                    fatia_limpa = np.zeros_like(fatia_suja)
                    for i in range(12):
                        fatia_limpa[:, i] = limpar_ecg_wavelet(fatia_suja[:, i])
                    
                    # 3. EXTRAI INTERVALOS E MORFOLOGIA DO SINAL LIMPO
                    feats = extrair_features_intervalos(fatia_limpa, LEADS, freq)
                    
                    # Salva referências
                    feats['ecg_id'] = eid
                    feats['segment_id'] = sid
                    feats['label'] = diag
                    
                    lista_features.append(feats)

    # Monta a matriz de IA
    df_final = pd.DataFrame(lista_features)
    if df_final.empty:
        return pd.DataFrame(columns=['ecg_id', 'segment_id', 'label'])
    cols_ordem = ['ecg_id', 'segment_id', 'label'] + [c for c in df_final.columns if c not in ['ecg_id', 'segment_id', 'label']]
    print(f"\nConcluído - DWT! Matriz gerada. Shape: {df_final.shape}")
    return df_final[cols_ordem]
