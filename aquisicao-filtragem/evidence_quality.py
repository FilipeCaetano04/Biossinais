import pandas as pd
import numpy as np
import wfdb
import ast
import neurokit2 as nk
import os
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class CreateDataRaw:
    """
    Evidence the database, gathering information from 3 diferent archives to match a complete raw_database
    """
    
    @staticmethod
    def _extract_label(scp_dict_str, diag_map):
        """Maps the class to a superclass based on diag_map"""
        try:
            dct = ast.literal_eval(scp_dict_str)
            for code in dct.keys():
                if code in diag_map:
                    return diag_map[code]
            return 'OTHER'
        except:
            return 'UNKNOWN'

    @classmethod
    def create_dataframe(cls, number_of_pacients: int, data_path: str = "../ignored_data/00000/", 
                         data_label: str = "../data500/ptbxl_database.csv", 
                         scp_path: str = "../data500/scp_statements.csv",
                         shuffle: bool = False,
                         to_csv:bool=False) -> pd.DataFrame:
        
        db = pd.read_csv(data_label, index_col='ecg_id')
        scp_st = pd.read_csv(scp_path, index_col=0)
        diag_map = scp_st[scp_st.diagnostic == 1]['diagnostic_class'].to_dict()

        # indices selection (ecg_id) 1 to 21837
        available_indices = db.index.values
        if shuffle:
            selected_ids = np.random.choice(available_indices, number_of_pacients, replace=False)
        else:
            selected_ids = available_indices[:number_of_pacients]
        
        all_records = []

        for ecg_id in selected_ids:
            row = db.loc[ecg_id]

            file_path = os.path.join(data_path, str(ecg_id).zfill(5) + '_hr')
            
            try:
                record = wfdb.rdrecord(file_path)
                df_temp = record.to_dataframe()
                
                # index and time
                df_temp = df_temp.reset_index()
                df_temp.columns = ['TEMPO'] + list(record.sig_name)
                df_temp['TEMPO'] = df_temp['TEMPO'].dt.total_seconds()
                
                # data insertion
                df_temp['age'] = row['age']
                df_temp['ecg_id'] = ecg_id
                df_temp['sex'] = row['sex']
                df_temp['weight'] = row['weight']
                df_temp['label'] = cls._extract_label(row['scp_codes'], diag_map)
                df_temp['ecg_id'] = ecg_id
                
                all_records.append(df_temp)
            except FileNotFoundError:
                print(f"{file_path} not found, going to next one")

        # final format
        df_final = pd.concat(all_records).reset_index()
        df_final.rename(columns={'index': 'INDEX'}, inplace=True)

        # Reordering
        cols_order = ['INDEX', 'TEMPO', 'ecg_id', 'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'age', 'sex','weight', 'label']
        

        df_final = df_final[[c for c in cols_order if c in df_final.columns]]

        print(f"Loaded registry of {number_of_pacients} pacients, {len(df_final)} registers")
        print(df_final.head(10))
        
        if to_csv: 
            df_final.to_csv("../data/raw_data.csv")

        return df_final

class SignalQualityEvaluator:
    """
    Quality avaluation of the data_raw
    """

    @staticmethod
    def _calculate_snr(signal, fs=500):
        """
        Calculates the snr by the assumption that the only noise is from electrical grid - 50hz.
        This may lead to stable snr but unacceptable for zhao
        """
        freqs, psd = welch(signal, fs, nperseg=1000)
        
        # defining freq masks
        mask_signal = (freqs >= 0.5) & (freqs <= 40.0)
        mask_noise  = (freqs >= 49.0) & (freqs <= 51.0)
        
        
        # heart bandwidth
        p_signal = np.trapezoid(psd[mask_signal], freqs[mask_signal])
        
        # electrical bandwidth
        p_noise = np.trapezoid(psd[mask_noise], freqs[mask_noise])
        

        if p_noise <= 1e-10:
            p_noise = 1e-10
            
        if p_signal <= 1e-10:
             return 0.0 
        

        snr_db = 10 * np.log10(p_signal / p_noise)
        
        return snr_db

    @classmethod
    def evaluate_quality(cls, df: pd.DataFrame, fs: int = 500, window_sec: int = 2) -> pd.DataFrame:
        derivacoes = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        derivacoes_criticas = ['I', 'II', 'V2'] 
        
        all_sqi_records = []
        window_samples = window_sec * fs

        grouped = df.groupby('ecg_id')

        for ecg_id, group in grouped:
            label_paciente = group['label'].iloc[0]
            patient_records = []

            for d in derivacoes:
                full_signal = group[d].values 
                total_samples = len(full_signal)
                
                for i in range(0, total_samples, window_samples):
                    segmento = full_signal[i:i+window_samples]
                    
                    if len(segmento) < window_samples:
                        continue
                        
                    seg_id = f"seg_{i//fs}a{(i+window_samples)//fs}s"
                    
                    try:
                        quality_status = nk.ecg_quality(segmento, sampling_rate=fs, method="zhao2018")
                        discard_seg = True if quality_status == "Unacceptable" else False
                        
                        snr_val = cls._calculate_snr(segmento, fs)
                        kurt = kurtosis(segmento, fisher=True)
                        sk = skew(segmento)
                        entropy, _ = nk.entropy_spectral(segmento)

                        patient_records.append({
                            'ecg_id': ecg_id,
                            'segment_id': seg_id,
                            'derivation': d,
                            'label_clinico': label_paciente,
                            'snr_db': round(snr_val, 2),
                            'kurtosis': round(kurt, 2),
                            'skewness': round(sk, 2),
                            'spectral_entropy': round(entropy, 2),
                            'quality_status': quality_status,
                            'discard_segment': discard_seg
                        })
                    except Exception as e:
                        pass
            
            if not patient_records:
                continue
                
            df_patient = pd.DataFrame(patient_records)
            

            derivas_ruins = df_patient[df_patient['discard_segment'] == True]['derivation'].unique()
            
            if len(derivas_ruins) > 2 or any(d in derivas_ruins for d in derivacoes_criticas):
                df_patient['discard_patient'] = True
            else:
                df_patient['discard_patient'] = False
                
            all_sqi_records.append(df_patient)

        return pd.concat(all_sqi_records, ignore_index=True) if all_sqi_records else pd.DataFrame()




class Visualizer:

    @staticmethod
    def plot_class_distribution(df_raw):
        """Class distribution based on df_raw"""
        plt.figure(figsize=(8, 5))
        counts = df_raw['label'].value_counts() / 5000
        df_counts = counts.reset_index()
        df_counts.columns = ['label', 'pacients_count']

        sns.barplot(data=df_counts, y='pacients_count', x='label', palette='viridis',hue='label')
        plt.title('Class distribution based on df_raw', fontsize=14)
        plt.xlabel('Clinic diagnostic', fontsize=12)
        plt.ylabel('Pacients number', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_raw_signal(df_raw, ecg_id, derivacao='II', fs=500):
        """10s raw data plot"""
        linha = df_raw[df_raw['ecg_id'] == ecg_id]
        sinal = linha[derivacao]
        tempo = np.arange(len(sinal)) / fs
        
        plt.figure(figsize=(12, 4))
        plt.plot(tempo, sinal, color='black', lw=1)
        plt.title(f'Raw data 10s plot (Pacient: {ecg_id} | Derivation: {derivacao} | Label: {linha["label"].iloc[0]})')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude (mV / V)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 10)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_snr_boxplot(df_sqi):
        """
        Compare SNR accepted by zhao vs SNR non accepted by zhao
        """
        plt.figure(figsize=(8, 5))
        
        df_plot = df_sqi.copy()
        df_plot['Status'] = df_plot['discard_segment'].map({False: 'Accepted (Excellent)', True: 'Rejected (Unacceptable)'})
        
        sns.boxplot(data=df_plot, x='Status', y='snr_db', palette=['#2ecc71', '#e74c3c'], hue='Status')
        plt.title('SNR distribution by Quality', fontsize=14)
        plt.ylabel('SNR (dB)', fontsize=12)
        plt.xlabel('')
        plt.axhline(y=10, color='r', linestyle='--', label='Critic border (10 dB)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_segmented_quality(df_raw, df_sqi, ecg_id, derivacao='II', fs=500, window_sec=2):
        linha_sinal = df_raw[df_raw['ecg_id'] == ecg_id]
        sinal = linha_sinal[derivacao]
        tempo = np.arange(len(sinal)) / fs
        
        sqi_paciente = df_sqi[(df_sqi['ecg_id'] == ecg_id) & (df_sqi['derivation'] == derivacao)]
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(tempo, sinal, color='black', lw=1, zorder=2)
        
        for idx, row in sqi_paciente.iterrows():

            t_str = row['segment_id'].replace('seg_', '').replace('s', '').split('a')
            t_inicio = int(t_str[0])
            t_fim = int(t_str[1])
            
            if row['discard_segment']:
                cor = '#ff9999' 
                label = 'Unacceptable'
            else:
                cor = '#99ff99' 
                label = 'Excellent'
                
            rect = patches.Rectangle((t_inicio, ax.get_ylim()[0]), t_fim - t_inicio, 
                                     ax.get_ylim()[1] - ax.get_ylim()[0], 
                                     linewidth=0, facecolor=cor, alpha=0.4, zorder=1)
            ax.add_patch(rect)
            
            ax.text((t_inicio + t_fim)/2, ax.get_ylim()[1]*0.9, label, 
                    horizontalalignment='center', fontsize=9, fontweight='bold', color='darkgreen' if not row['discard_segment'] else 'darkred')

        plt.title(f'Segment Quality ({window_sec}s) - Pacient {ecg_id} | derivation {derivacao}', fontsize=14)
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 10)
        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    print("\n--- EVIDENCE ---")
    df_raw = CreateDataRaw.create_dataframe(
        number_of_pacients=10, 
        data_path="../ignored_data/00000/", 
        data_label="../data500/ptbxl_database.csv",
        shuffle=False,
        to_csv=False #saves to csv or not
    )
    df_sqi = SignalQualityEvaluator.evaluate_quality(df_raw, fs=500)
    df_sqi.to_csv('../data/quality_data_raw.csv')

    
    print("\n--- QUALITY ---")
    print(df_sqi.head(60))

    print('\n--- PLOTS ---')
    Visualizer.plot_class_distribution(df_raw)
    Visualizer.plot_raw_signal(df_raw, ecg_id=2)
    Visualizer.plot_snr_boxplot(df_sqi)

    Visualizer.plot_segmented_quality(df_raw, df_sqi, ecg_id=2, derivacao='II')
    
