import pandas as pd
import numpy as np
import re
from scipy.stats import kurtosis, skew
import pywt

def extract_features(matriz_sinal: np.ndarray, leads: list):
    """Calcula as features matemáticas de 1 segmento perfeito (1000 linhas x 12 leads)."""
    features = {}
    for i, lead in enumerate(leads):
        sinal = matriz_sinal[:, i]
        #features[f'Energia_{lead}'] = np.sum(sinal ** 2)
        features[f'RMS_{lead}'] = np.sqrt(np.mean(sinal ** 2))
        features[f'PtP_{lead}'] = np.ptp(sinal) 
        features[f'Kurtosis_{lead}'] = kurtosis(sinal, fisher=True, bias=False)
        features[f'Skewness_{lead}'] = skew(sinal, bias=False)
        features[f'Var_{lead}'] = np.var(sinal)
        coeffs = pywt.wavedec(sinal, 'db4', level=3)
        
        # cA3: Coeficientes de Aproximação (Frequências muito baixas, ex: Onda T e P)
        # cD3: Detalhes nível 3 (Frequências médias, ex: Complexo QRS)
        # cD1: Detalhes nível 1 (Frequências altas, espículas ou ruído)
        cA3, cD3, cD2, cD1 = coeffs
        
        # guardar a energia das ondas
        features[f'Wav_Baixa_OndaTP_{lead}'] = np.sum(cA3 ** 2)
        features[f'Wav_Media_QRS_{lead}'] = np.sum(cD3 ** 2)
        features[f'Wav_Alta_Freq_{lead}'] = np.sum(cD1 ** 2)
    return features

def validation_extraction(df_raw: pd.DataFrame, df_not_bad_data: pd.DataFrame, freq=500):
    """
    Usa o df_not_bad_data para recortar o sinal do df_raw .
    Adaptado para o formato de segmento 'seg_XaYs'.
    """
    amostras_por_segmento = int(2 * freq) # 1000 linhas = 2 segundos
    LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    grouped_raw = df_raw.groupby('ecg_id')
    
    col_label = 'label_clinico' if 'label_clinico' in df_not_bad_data.columns else 'label'
    mapa_aprovados = df_not_bad_data[['ecg_id', 'segment_id', col_label]].drop_duplicates()
    
    lista_elite = []
    
    for _, row in mapa_aprovados.iterrows():
        eid = row['ecg_id']
        sid = row['segment_id']
        diag = row[col_label]
        
        if eid in grouped_raw.groups:
            sinal_completo = grouped_raw.get_group(eid)
            

            num = re.findall(r'\d+', str(sid))
            
            if num:
                segundo_inicial = int(num[0])
                
                inicio = segundo_inicial * freq
                fim = inicio + amostras_por_segmento
                
                fatia = sinal_completo.iloc[inicio:fim][LEADS].values
                
                if len(fatia) == amostras_por_segmento:
                    feats = extract_features(fatia, LEADS)
                    
                    feats['ecg_id'] = eid
                    feats['segment_id'] = sid
                    feats['label'] = diag
                    
                    lista_elite.append(feats)

    df_final = pd.DataFrame(lista_elite)
    if df_final.empty:
        return pd.DataFrame(columns=['ecg_id', 'segment_id', 'label'])
    
    # organiza
    cols_ordem = ['ecg_id', 'segment_id', 'label'] + [c for c in df_final.columns if c not in ['ecg_id', 'segment_id', 'label']]
    return df_final[cols_ordem]
