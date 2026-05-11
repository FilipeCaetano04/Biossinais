import pandas as pd
import numpy as np
import wfdb
import ast
import neurokit2 as nk
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
from typing import Mapping, Sequence
from .signal_cleaning_validation import ECGSignalCleaner, plotar_comparacao_filtros
from .ica import plotar_ica_estatico, ICA, remove_outliers, PCA_SIMPLE
from .feature_extraction_fft import validation_extraction
from .wavelet import gerar_features_clinicas


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
            return "OTHER"
        except:
            return "UNKNOWN"

    @staticmethod
    def build_record_index(data_path: str) -> dict[int, str]:
        data_root = Path(data_path)
        record_files = sorted(data_root.rglob("*_hr.hea"))
        record_path_by_id: dict[int, str] = {}

        for record_file in record_files:
            stem = record_file.stem
            if not stem.endswith("_hr"):
                continue

            ecg_id_str = stem[:-3]
            try:
                ecg_id = int(ecg_id_str)
            except ValueError:
                continue

            record_path_by_id[ecg_id] = str(record_file.with_suffix(""))

        return record_path_by_id

    @classmethod
    def list_available_record_ids(
        cls,
        data_path: str,
        data_label: str,
        record_index: Mapping[int, str] | None = None,
    ) -> list[int]:
        db = pd.read_csv(data_label, index_col="ecg_id")
        available_indices = {int(idx) for idx in db.index.values}
        index_map = dict(record_index) if record_index is not None else cls.build_record_index(data_path)
        return sorted(ecg_id for ecg_id in index_map.keys() if ecg_id in available_indices)

    @classmethod
    def create_dataframe(
        cls,
        number_of_pacients: int,
        process_all: bool,
        data_path: str,
        data_label: str,
        scp_path: str,
        shuffle: bool,
        to_csv: bool,
        selected_ids: Sequence[int] | None = None,
        record_index: Mapping[int, str] | None = None,
    ) -> pd.DataFrame:

        db = pd.read_csv(data_label, index_col="ecg_id")
        scp_st = pd.read_csv(scp_path, index_col=0)
        diag_map = scp_st[scp_st.diagnostic == 1]["diagnostic_class"].to_dict()

        record_path_by_id = (
            dict(record_index)
            if record_index is not None
            else cls.build_record_index(data_path)
        )

        available_indices = {int(idx) for idx in db.index.values}
        available_record_ids = sorted(
            ecg_id for ecg_id in record_path_by_id.keys() if ecg_id in available_indices
        )

        if not available_record_ids:
            raise ValueError(
                f"No records found in '{data_path}' that match '{data_label}'."
            )

        if selected_ids is not None:
            selected_ids = [int(ecg_id) for ecg_id in selected_ids if int(ecg_id) in record_path_by_id]
            selected_ids = [ecg_id for ecg_id in selected_ids if ecg_id in available_indices]
            if not selected_ids:
                raise ValueError("No selected IDs matched available records and metadata.")
        elif process_all:
            selected_ids = available_record_ids
        else:
            if number_of_pacients <= 0:
                raise ValueError("number_of_pacients must be greater than zero.")

            sample_size = min(number_of_pacients, len(available_record_ids))
            if shuffle:
                selected_ids = np.random.choice(
                    available_record_ids, sample_size, replace=False
                ).tolist()
            else:
                selected_ids = available_record_ids[:sample_size]

        print(
            "Discovered "
            f"{len(record_path_by_id)} records in {data_path}; "
            f"{len(available_record_ids)} match label metadata; "
            f"processing {len(selected_ids)} records."
        )

        all_records = []
        total_ids = len(selected_ids)
        progress_every = 250

        for idx, ecg_id in enumerate(selected_ids, start=1):
            row = db.loc[ecg_id]

            file_path = record_path_by_id.get(int(ecg_id))
            if file_path is None:
                continue

            try:
                record = wfdb.rdrecord(file_path)
                df_temp = record.to_dataframe()

                # index and time
                df_temp = df_temp.reset_index()
                df_temp.columns = ["TEMPO"] + list(record.sig_name)
                df_temp["TEMPO"] = df_temp["TEMPO"].dt.total_seconds()

                # data insertion
                df_temp["age"] = row["age"]
                df_temp["ecg_id"] = ecg_id
                df_temp["sex"] = row["sex"]
                df_temp["weight"] = row["weight"]
                df_temp["label"] = cls._extract_label(row["scp_codes"], diag_map)
                df_temp["ecg_id"] = ecg_id

                all_records.append(df_temp)
            except FileNotFoundError:
                print(f"{file_path} not found, going to next one")

            if idx % progress_every == 0 or idx == total_ids:
                print(
                    f"[CreateDataRaw] Loaded {idx}/{total_ids} records "
                    f"({(100 * idx / total_ids):.1f}%)"
                )

        if not all_records:
            raise ValueError(
                "No records were loaded. Check dataset paths and metadata alignment."
            )

        # final format
        df_final = pd.concat(all_records).reset_index()
        df_final.rename(columns={"index": "INDEX"}, inplace=True)

        # Reordering
        cols_order = [
            "INDEX",
            "TEMPO",
            "ecg_id",
            "I",
            "II",
            "III",
            "AVR",
            "AVL",
            "AVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "age",
            "sex",
            "weight",
            "label",
        ]

        df_final = df_final[[c for c in cols_order if c in df_final.columns]]

        print(
            f"Loaded registry of {len(selected_ids)} pacients, {len(df_final)} registers"
        )
        print(df_final.head(10))
        df_final = df_final.drop(df_final[df_final["label"] == "OTHER"].index)

        if to_csv:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(data_dir / "raw_data.csv", index=False)

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
        mask_noise = (freqs >= 49.0) & (freqs <= 51.0)

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
    def evaluate_quality(
        cls, df: pd.DataFrame, fs: int = 500, window_sec: int = 2
    ) -> pd.DataFrame:
        derivacoes = [
            "I",
            "II",
            "III",
            "AVR",
            "AVL",
            "AVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        derivacoes_criticas = ["I", "II", "V2"]

        all_sqi_records = []
        window_samples = window_sec * fs

        grouped = df.groupby("ecg_id")

        for ecg_id, group in grouped:
            label_paciente = group["label"].iloc[0]
            patient_records = []

            for d in derivacoes:
                full_signal = group[d].values
                total_samples = len(full_signal)

                for i in range(0, total_samples, window_samples):
                    segmento = full_signal[i : i + window_samples]

                    if len(segmento) < window_samples:
                        continue

                    seg_id = f"seg_{i // fs}a{(i + window_samples) // fs}s"

                    try:
                        quality_status = nk.ecg_quality(
                            segmento, sampling_rate=fs, method="zhao2018"
                        )
                        discard_seg = (
                            True if quality_status == "Unacceptable" else False
                        )

                        snr_val = cls._calculate_snr(segmento, fs)
                        kurt = kurtosis(segmento, fisher=True)
                        sk = skew(segmento)
                        entropy, _ = nk.entropy_spectral(segmento)

                        patient_records.append(
                            {
                                "ecg_id": ecg_id,
                                "segment_id": seg_id,
                                "derivation": d,
                                "label_clinico": label_paciente,
                                "snr_db": round(snr_val, 2),
                                "kurtosis": round(kurt, 2),
                                "skewness": round(sk, 2),
                                "spectral_entropy": round(entropy, 2),
                                "quality_status": quality_status,
                                "discard_segment": discard_seg,
                            }
                        )
                    except Exception:
                        pass

            if not patient_records:
                continue

            df_patient = pd.DataFrame(patient_records)

            derivas_ruins = df_patient[df_patient["discard_segment"] == True][
                "derivation"
            ].unique()

            if len(derivas_ruins) > 2 or any(
                d in derivas_ruins for d in derivacoes_criticas
            ):
                df_patient["discard_patient"] = True
            else:
                df_patient["discard_patient"] = False

            all_sqi_records.append(df_patient)

        return (
            pd.concat(all_sqi_records, ignore_index=True)
            if all_sqi_records
            else pd.DataFrame()
        )

    @classmethod
    def remove_bad_data(cls, df_quality: pd.DataFrame):
        df_quality["ecg_id"] = df_quality["ecg_id"].astype(int)

        # Convertemos para numérico antes de filtrar pra garantir
        df_quality["discard_segment"] = pd.to_numeric(
            df_quality["discard_segment"], errors="coerce"
        )
        df_quality["discard_patient"] = pd.to_numeric(
            df_quality["discard_patient"], errors="coerce"
        )

        df_validos: pd.DataFrame = df_quality[
            (df_quality["discard_segment"] == 0) & (df_quality["discard_patient"] == 0)
        ].copy()

        print(f"📊 Linhas após filtro de qualidade: {len(df_validos)}")
        if len(df_validos) == 0:
            print(
                "❌ ERRO: O filtro de qualidade removeu TUDO. Verifique os valores de discard_segment/patient."
            )
            return pd.DataFrame()

        print("Estatisticas dos df_validos:")
        print(
            f"Quantidade de dados:\nNovo->{df_validos.shape}\n->antigo:{df_quality.shape}"
        )
        print(
            f"Quantidade de pacientes unicos:\nNovo->{df_validos['ecg_id'].nunique()}\nAntigo->{df_quality['ecg_id'].nunique()}"
        )

        return df_validos

    @classmethod
    def balancear_classes_undersampling(
        cls, df: pd.DataFrame, label_col="label", max_ratio=3
    ):
        """
        Reduz as classes majoritárias (Undersampling) para que nenhuma classe tenha
        mais do que `max_ratio` vezes o tamanho da menor classe.
        """
        print("📊 Distribuição ANTES do balanceamento:")
        contagem_antes = df[label_col].value_counts()
        print(contagem_antes.to_string())

        # 1. Encontra qual é a classe com menos dados e define o teto
        min_count = contagem_antes.min()
        max_permitido = min_count * max_ratio

        print(f"\n🎯 A menor classe tem {min_count} amostras.")
        print(
            f"✂️ Limite máximo definido para {max_permitido} amostras por classe (Ratio: {max_ratio}x).\n"
        )

        lista_dfs = []

        # 2. Avalia cada classe separadamente
        for classe in contagem_antes.index:
            df_classe = df[df[label_col] == classe]
            qtd_atual = len(df_classe)

            if qtd_atual > max_permitido:
                # Sorteia aleatoriamente as linhas para reduzir até o limite
                # random_state=42 garante que você pode rodar 10x e ele corta os mesmos pacientes
                df_classe_reduzida = df_classe.sample(n=max_permitido, random_state=42)
                print(
                    f"  🔻 Classe {classe}: Reduzida de {qtd_atual} para {max_permitido}."
                )
                lista_dfs.append(df_classe_reduzida)
            else:
                print(f"  ✔️ Classe {classe}: Mantida intacta com {qtd_atual} amostras.")
                lista_dfs.append(df_classe)

        # 3. Junta tudo em um único DataFrame
        df_balanceado = pd.concat(lista_dfs)

        # 4. EMBARALHAMENTO (Crucial para Machine Learning)
        # O sample(frac=1) embaralha 100% das linhas. Se não fizermos isso,
        # o algoritmo vai ler todos os normais, depois todos os infartos... e isso pode viciar o modelo.
        df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        print("\n✅ Balanceamento concluído! Nova distribuição:")
        print(df_balanceado[label_col].value_counts().to_string())

        return df_balanceado


class Visualizer:
    @staticmethod
    def plot_class_distribution(df: pd.DataFrame, title_suffix=""):
        """Class distribution based on df_raw"""
        plt.figure(figsize=(8, 5))
        label_col = [c for c in df.columns if c.startswith("label")]
        counts = (
            df.groupby(label_col[0])["ecg_id"].nunique().sort_values(ascending=False)
        )

        # Transformar em DataFrame para o Seaborn
        df_counts = counts.reset_index()
        df_counts.columns = ["Diagnostic", "Patient_Count"]

        # 3. Plotagem
        sns.barplot(
            data=df_counts,
            x="Diagnostic",
            y="Patient_Count",
            palette="viridis",
            hue="Diagnostic",
            legend=False,
        )

        # Adicionar os números exatos no topo das barras
        for i, v in enumerate(df_counts["Patient_Count"]):
            plt.text(i, v + (v * 0.01), str(v), ha="center", fontweight="bold")

        plt.title(
            f"Distribuição de Pacientes Únicos por Classe {title_suffix}", fontsize=14
        )
        plt.xlabel("Diagnóstico Clínico", fontsize=12)
        plt.ylabel("Número de Pacientes", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_raw_signal(df_raw, ecg_id, derivacao="II", fs=500):
        """10s raw data plot"""
        linha = df_raw[df_raw["ecg_id"] == ecg_id]
        sinal = linha[derivacao]
        tempo = np.arange(len(sinal)) / fs

        plt.figure(figsize=(12, 4))
        plt.plot(tempo, sinal, color="black", lw=1)
        plt.title(
            f"Raw data 10s plot (Pacient: {ecg_id} | Derivation: {derivacao} | Label: {linha['label'].iloc[0]})"
        )
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Amplitude (mV / V)")
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
        df_plot["Status"] = df_plot["discard_segment"].map(
            {False: "Accepted (Excellent)", True: "Rejected (Unacceptable)"}
        )

        sns.boxplot(
            data=df_plot,
            x="Status",
            y="snr_db",
            palette=["#2ecc71", "#e74c3c"],
            hue="Status",
        )
        plt.title("SNR distribution by Quality", fontsize=14)
        plt.ylabel("SNR (dB)", fontsize=12)
        plt.xlabel("")
        plt.axhline(y=10, color="r", linestyle="--", label="Critic border (10 dB)")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_segmented_quality(
        df_raw, df_sqi, ecg_id, derivacao="II", fs=500, window_sec=2
    ):
        linha_sinal = df_raw[df_raw["ecg_id"] == ecg_id]
        sinal = linha_sinal[derivacao]
        tempo = np.arange(len(sinal)) / fs

        sqi_paciente = df_sqi[
            (df_sqi["ecg_id"] == ecg_id) & (df_sqi["derivation"] == derivacao)
        ]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(tempo, sinal, color="black", lw=1, zorder=2)

        for idx, row in sqi_paciente.iterrows():
            t_str = row["segment_id"].replace("seg_", "").replace("s", "").split("a")
            t_inicio = int(t_str[0])
            t_fim = int(t_str[1])

            if row["discard_segment"]:
                cor = "#ff9999"
                label = "Unacceptable"
            else:
                cor = "#99ff99"
                label = "Excellent"

            rect = patches.Rectangle(
                (t_inicio, ax.get_ylim()[0]),
                t_fim - t_inicio,
                ax.get_ylim()[1] - ax.get_ylim()[0],
                linewidth=0,
                facecolor=cor,
                alpha=0.4,
                zorder=1,
            )
            ax.add_patch(rect)

            ax.text(
                (t_inicio + t_fim) / 2,
                ax.get_ylim()[1] * 0.9,
                label,
                horizontalalignment="center",
                fontsize=9,
                fontweight="bold",
                color="darkgreen" if not row["discard_segment"] else "darkred",
            )

        plt.title(
            f"Segment Quality ({window_sec}s) - Pacient {ecg_id} | derivation {derivacao}",
            fontsize=14,
        )
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Amplitude")
        plt.xlim(0, 10)
        plt.tight_layout()
        plt.show()

    @classmethod
    def plotar_amostras_aleatorias(cls, df_raw, lead="II", freq=500):
        """
        Sorteia um paciente de cada classe clínica e plota o sinal da derivação II.
        """
        # 1. Busca inteligente da coluna de label (mesma lógica robusta de antes)
        label_col = [c for c in df_raw.columns if c.startswith("label")][0]

        # 2. Identifica todas as classes clínicas únicas (você mencionou 6)
        classes_unicas = df_raw[label_col].dropna().unique()
        n_classes = len(classes_unicas)

        # 3. Prepara a Figura (N linhas x 1 coluna)
        # A altura ajusta dinamicamente dependendo da quantidade de classes
        fig, axes = plt.subplots(
            nrows=n_classes, ncols=1, figsize=(10, 8), sharex=True, sharey=True
        )

        # Se por acaso tiver apenas 1 classe, axes não será uma lista, então forçamos ser
        if n_classes == 1:
            axes = [axes]

        # 4. Loop por cada diagnóstico clínico
        for ax, classe in zip(axes, classes_unicas):
            # Filtra os dados para pegar apenas pacientes dessa doença
            df_classe = df_raw[df_raw[label_col] == classe]

            # Lista os pacientes (ecg_id) únicos que têm esse diagnóstico
            pacientes = df_classe["ecg_id"].unique()

            # Sorteia UM paciente dessa lista
            paciente_sorteado = random.choice(pacientes)

            # Pega todas as linhas de sinal (o tempo todo) desse paciente sorteado
            sinal_paciente = df_raw[df_raw["ecg_id"] == paciente_sorteado]

            # Verifica se a coluna 'II' existe, caso esteja nomeada diferente
            coluna_derivação = (
                f"{lead}" if f"{lead}" in sinal_paciente.columns else f"lead_{lead}"
            )

            # Extrai os valores do sinal
            sinal_II = sinal_paciente[coluna_derivação].values

            # Cria o eixo do tempo (amostras / frequência = segundos)
            tempo = np.arange(len(sinal_II)) / freq

            # Plota no subplot
            ax.plot(tempo, sinal_II, color="#2ca02c", linewidth=1.2)

            # Configurações visuais do subplot
            ax.set_title(
                f"Diagnóstico: {classe} | Sorteado: Paciente {paciente_sorteado}",
                fontsize=11,
                fontweight="bold",
                loc="left",
            )
            ax.set_ylabel("Amplitude (mV)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Remove as bordas superior e direita para ficar mais "clean" (estilo artigo)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # 5. Ajustes Finais do Gráfico Inteiro
        plt.xlabel("Tempo (segundos)", fontsize=12, fontweight="bold")
        plt.suptitle(
            f"Comparação da Derivação {lead} entre Diferentes Patologias (Amostras Aleatórias)",
            fontsize=15,
            y=1,
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("\n--- EVIDENCE ---")
    df_raw = CreateDataRaw.create_dataframe(
        number_of_pacients=1000,
        process_all=False,
        data_path="./ignored_data",
        data_label="./data500/ptbxl_database.csv",
        scp_path="./data500/scp_statements.csv",
        shuffle=False,
        to_csv=False,  # saves to csv or not
    )
    print("\n--- QUALITY ---")
    df_sqi = SignalQualityEvaluator.evaluate_quality(df_raw, fs=500)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    quality_path = data_dir / "quality_data_raw.csv"
    df_sqi.to_csv(quality_path, index=False)
    df_sqi = pd.read_csv(quality_path)

    df_not_bad_data = SignalQualityEvaluator.remove_bad_data(df_quality=df_sqi)

    df_raw_filtrado_sqi = df_raw[
        df_raw["ecg_id"].isin(df_not_bad_data["ecg_id"].unique())
    ].copy()
    print(
        f"unicos em:{df_raw['label'].unique()}\n{df_not_bad_data['label_clinico'].unique()}"
    )

    print(df_sqi.head(60))

    print("\n-------FILTERED-------")

    df_filtered = ECGSignalCleaner.clean_signals(
        df_raw_filtrado_sqi,
        ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    )

    print(df_filtered.head())
    # EXTRAÇÃO DE FEATURES:
    df_features = validation_extraction(df_filtered, df_not_bad_data)
    print(df_features.shape)
    print(df_features.head(10))
    df_features = remove_outliers(df_features)
    df_features_balanced = SignalQualityEvaluator.balancear_classes_undersampling(
        df_features, max_ratio=2
    )

    print("\n------ PCA & ICA -------")
    df_ica_features_stats = ICA(
        df_features_balanced,
        feature_cols=[
            col
            for col in df_features.columns
            if col not in ["ecg_id", "segment_id", "label"]
        ],
    )
    df_pca_features_stats = PCA_SIMPLE(df_features_balanced, n_components=5)

    print("----- WAVELET FEATURE EXTRACTION ------")
    df_features_novas = gerar_features_clinicas(df_raw, df_not_bad_data)
    df_ica = ICA(
        df_features_novas,
        feature_cols=[
            col
            for col in df_features_novas.columns
            if col not in ["ecg_id", "segment_id", "label"]
        ],
    )
    df_pca = PCA_SIMPLE(df_features_novas, n_components=5)

    print("\n--- PLOTS ---")

    def plotter():
        Visualizer.plot_class_distribution(df_raw)
        Visualizer.plot_raw_signal(df_raw, ecg_id=2)
        Visualizer.plot_snr_boxplot(df_sqi)
        Visualizer.plotar_amostras_aleatorias(df_raw_filtrado_sqi)
        plotar_comparacao_filtros(df_raw_filtrado_sqi, df_filtered)
        Visualizer.plot_class_distribution(df_not_bad_data)
        plotar_ica_estatico(df_ica_features_stats, "ICA FEATURE STATS")
        plotar_ica_estatico(df_pca_features_stats, "PCA FEATURE STATS")
        plotar_ica_estatico(df_ica, "ICA FEATURES WAVELET")
        plotar_ica_estatico(df_pca, "PCA FEATURES WAVELET")

    plotter()
