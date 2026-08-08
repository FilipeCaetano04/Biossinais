from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class PCAPico:
    """
    detecta os picos R, corta janelas ao redor
    de cada pico e roda PCA nas janelas achatadas para todas as derivacoes.
    """

    def __init__(
        self,
        leads=None,
        rpeak_lead="II",
        fs=500,
        pre_sec=0.2,
        post_sec=0.4,
        n_components=2,
    ):
        self.leads = LEADS if leads is None else list(leads)
        self.rpeak_lead = rpeak_lead
        self.fs = fs
        self.pre_sec = pre_sec
        self.post_sec = post_sec
        self.n_components = n_components

    def _validate_columns(self, df):
        required = ["ecg_id", "label", self.rpeak_lead] + self.leads
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no dataframe: {missing}")

    def create_peak_window_matrix(self, df):
        self._validate_columns(df)

        pre_samples = int(self.pre_sec * self.fs)
        post_samples = int(self.post_sec * self.fs)
        window_samples = pre_samples + post_samples

        rows = []
        metadata = []

        for ecg_id, group in df.groupby("ecg_id"):
            group = group.sort_values("TEMPO") if "TEMPO" in group.columns else group.sort_index()

            rpeak_signal = group[self.rpeak_lead].to_numpy(dtype=float)
            lead_matrix = group[self.leads].to_numpy(dtype=float)

            try:
                _, info = nk.ecg_peaks(rpeak_signal, sampling_rate=self.fs)
                rpeaks = info["ECG_R_Peaks"]
            except Exception as exc:
                print(f"Falha ao detectar pico R no ecg_id={ecg_id}: {exc}")
                continue

            for beat_index, rpeak in enumerate(rpeaks):
                start = int(rpeak) - pre_samples
                end = int(rpeak) + post_samples

                if start < 0 or end > len(lead_matrix):
                    continue

                window = lead_matrix[start:end, :]
                if window.shape[0] != window_samples:
                    continue

                rows.append(window.reshape(-1))
                metadata.append({
                    "ecg_id": ecg_id,
                    "label": group["label"].iloc[0],
                    "beat_index": beat_index,
                    "rpeak_index": int(rpeak),
                    "rpeak_time_sec": int(rpeak) / self.fs,
                    "window_start_sec": start / self.fs,
                    "window_end_sec": end / self.fs,
                    "n_leads": len(self.leads),
                    "window_samples": window_samples,
                })

        if not rows:
            raise ValueError("Nenhuma janela valida foi criada a partir dos picos R.")

        X = np.vstack(rows)
        metadata_df = pd.DataFrame(metadata)

        print("Shape da matriz pico R antes da PCA:", X.shape)
        return X, metadata_df

    def run_pca(self, X, metadata):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        max_components = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
        pca_model = PCA(n_components=max_components)
        scores = pca_model.fit_transform(X_scaled)

        pca_df = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(max_components)])
        pca_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)

        print("Shape usado na PCA:", X_scaled.shape)
        print("Variancia explicada:", pca_model.explained_variance_ratio_)
        print("Variancia acumulada:", pca_model.explained_variance_ratio_.cumsum())

        return pca_df, pca_model, scaler, imputer

    def plot_pc1_pc2(self, pca_df, output_path=None):
        if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
            raise ValueError("A PCA precisa ter PC1 e PC2 para plotar.")

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(9, 7))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="label",
            alpha=0.72,
            s=32,
            edgecolor="none",
        )
        plt.title("PCA por pico R e janelamento")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=220, bbox_inches="tight")

        plt.show()

    def plot_pc1_pc2_pc3(self, pca_df, output_path=None):
        required = ["PC1", "PC2", "PC3"]
        missing = [c for c in required if c not in pca_df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes para plot 3D: {missing}")

        labels = pca_df["label"].unique()
        palette = sns.color_palette("tab10", n_colors=len(labels))
        color_map = dict(zip(labels, palette))

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        for label in labels:
            subset = pca_df[pca_df["label"] == label]
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                subset["PC3"],
                label=label,
                color=color_map[label],
                alpha=0.75,
                s=30,
                edgecolors="none",
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA por pico R e janelamento - 3D")
        ax.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=220, bbox_inches="tight")

        plt.show()

    def process(self, df, plot=True, output_dir=None, save_scores=True):
        X, metadata = self.create_peak_window_matrix(df)
        pca_df, pca_model, scaler, imputer = self.run_pca(X, metadata)

        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        if save_scores and output_path is not None:
            pca_df.to_csv(output_path / "pca_pico_scores.csv", index=False)

        if plot:
            plot_path = output_path / "pca_pico_pc1_pc2.png" if output_path is not None else None
            self.plot_pc1_pc2(pca_df, output_path=plot_path)
            self.plot_pc1_pc2_pc3(pca_df)

        return pca_df, pca_model, scaler, imputer
