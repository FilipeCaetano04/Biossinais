import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal

from pca_pico import LEADS, PCAPico


class PCAFiltradoPicoPipeline:
    """
    Pipeline completa:
    Carrega os pacientes do PTB-XL.
    Aplica filtro passa-banda 0.5-40 Hz nas 12 derivacoes.
    Aplica notch em 50 Hz nas 12 derivacoes.
    etecta pico R na derivacao II filtrada.
    Corta janelas ao redor de cada pico R.
    Roda PCA e plota PC1 x PC2 com hue=label.
    """

    def __init__(
        self,
        number_of_pacients=1000,
        data_path="../ignored_data/00000/",
        data_label="../data500/ptbxl_database.csv",
        scp_path="../data500/scp_statements.csv",
        fs=500,
        rpeak_lead="II",
        pre_sec=0.2,
        post_sec=0.4,
        n_components=6,
        output_dir="../data/pca_filtrado_pico_outputs",
    ):
        self.number_of_pacients = number_of_pacients
        self.data_path = self._resolve_path(data_path)
        self.data_label = self._resolve_path(data_label)
        self.scp_path = self._resolve_path(scp_path)
        self.fs = fs
        self.rpeak_lead = rpeak_lead
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.n_components = n_components
        self.output_dir = self._resolve_path(output_dir)

    @staticmethod
    def _script_dir():
        return Path(__file__).resolve().parent

    def _resolve_path(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        return (self._script_dir() / path).resolve()

    @staticmethod
    def _extract_label(scp_dict_str, diag_map):
        try:
            dct = ast.literal_eval(scp_dict_str)
            for code in dct.keys():
                if code in diag_map:
                    return diag_map[code]
            return "OTHER"
        except Exception:
            return "UNKNOWN"

    def create_dataframe(self, shuffle=False):
        db = pd.read_csv(self.data_label, index_col="ecg_id")
        scp_st = pd.read_csv(self.scp_path, index_col=0)
        diag_map = scp_st[scp_st.diagnostic == 1]["diagnostic_class"].to_dict()

        available_indices = db.index.values
        if shuffle:
            selected_ids = np.random.choice(available_indices, self.number_of_pacients, replace=False)
        else:
            selected_ids = available_indices[:self.number_of_pacients]

        all_records = []

        for ecg_id in selected_ids:
            row = db.loc[ecg_id]
            file_path = self.data_path / f"{str(ecg_id).zfill(5)}_hr"

            try:
                record = wfdb.rdrecord(str(file_path))
                df_temp = record.to_dataframe()
                df_temp = df_temp.reset_index()
                df_temp.columns = ["TEMPO"] + list(record.sig_name)
                df_temp["TEMPO"] = df_temp["TEMPO"].dt.total_seconds()

                df_temp["age"] = row["age"]
                df_temp["ecg_id"] = ecg_id
                df_temp["sex"] = row["sex"]
                df_temp["weight"] = row["weight"]
                df_temp["label"] = self._extract_label(row["scp_codes"], diag_map)

                all_records.append(df_temp)
            except FileNotFoundError:
                print(f"{file_path} not found, going to next one")

        if not all_records:
            raise ValueError("Nenhum registro foi carregado. Confira data_path e data_label.")

        df_final = pd.concat(all_records).reset_index()
        df_final.rename(columns={"index": "INDEX"}, inplace=True)

        cols_order = ["INDEX", "TEMPO", "ecg_id"] + LEADS + ["age", "sex", "weight", "label"]
        df_final = df_final[[col for col in cols_order if col in df_final.columns]]

        print(f"Loaded registry of {self.number_of_pacients} pacients, {len(df_final)} registers")
        return df_final

    def aplicar_filtros(self, sinal, fs):
        # Filtro Passa-Banda Butterworth - 0.5 Hz a 40 Hz
        # Remove baixas frequencias (< 0.5 Hz) e ruidos de alta frequencia (> 40 Hz)
        nyquist = 0.5 * fs
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
        sinal_bandpass = signal.filtfilt(b, a, sinal)

        # Filtro Rejeita-Faixa para Ruido de Linha
        f0 = 50.0  # Frequencia a ser removida
        Q = 30.0   # Fator de qualidade
        b_notch, a_notch = signal.iirnotch(f0, Q, fs)
        sinal_filtrado = signal.filtfilt(b_notch, a_notch, sinal_bandpass)

        return sinal_filtrado

    def filtrar_dataframe(self, df_raw):
        df_filtrado = df_raw.copy()

        for ecg_id, group in df_raw.groupby("ecg_id"):
            idx = group.sort_values("TEMPO").index

            for lead in LEADS:
                sinal = df_raw.loc[idx, lead].to_numpy(dtype=float)
                df_filtrado.loc[idx, lead] = self.aplicar_filtros(sinal, self.fs)

        return df_filtrado

    def run(self, shuffle=False, plot=True, save_filtered=False):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n--- Carregando dados ---")
        df_raw = self.create_dataframe(shuffle=shuffle)

        print("\n--- Filtrando as 12 derivacoes ---")
        df_filtrado = self.filtrar_dataframe(df_raw)

        if save_filtered:
            df_filtrado.to_csv(self.output_dir / "df_filtrado_pico.csv", index=False)

        print("\n--- PCA por pico R e janelamento ---")
        pca = PCAPico(
            leads=LEADS,
            rpeak_lead=self.rpeak_lead,
            fs=self.fs,
            pre_sec=self.pre_sec,
            post_sec=self.post_sec,
            n_components=self.n_components,
        )
        pca_df, pca_model, scaler, imputer = pca.process(
            df_filtrado,
            plot=plot,
            output_dir=self.output_dir,
            save_scores=True,
        )

        return {
            "df_raw": df_raw,
            "df_filtrado": df_filtrado,
            "pca_df": pca_df,
            "pca_model": pca_model,
            "scaler": scaler,
            "imputer": imputer,
        }


if __name__ == "__main__":
    pipeline = PCAFiltradoPicoPipeline(
        number_of_pacients=1000,
        data_path="../ignored_data/00000/",
        data_label="../data500/ptbxl_database.csv",
        scp_path="../data500/scp_statements.csv",
    )
    pipeline.run()
