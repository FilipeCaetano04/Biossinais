from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FEATURE_COLS = [f"mean_{lead}" for lead in LEADS]


def project_root():
    return Path(__file__).resolve().parents[1]


def load_mean_statistics(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["ecg_id", "segment_id", "label"] + FEATURE_COLS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    return df.dropna(subset=["label"]).copy()


def run_mean_pca(df, n_components=6):
    metadata = df[["ecg_id", "segment_id", "label"]].reset_index(drop=True)
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    max_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca_model = PCA(n_components=max_components)
    scores = pca_model.fit_transform(X_scaled)

    pca_df = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(max_components)])
    pca_df = pd.concat([pca_df, metadata], axis=1)

    print("\n--- PCA usando somente medias por derivacao ---")
    print("Features usadas:", FEATURE_COLS)
    print("Shape usado na PCA:", X_scaled.shape)
    print("Variancia explicada:", pca_model.explained_variance_ratio_)
    print("Variancia acumulada:", pca_model.explained_variance_ratio_.cumsum())

    return pca_df, pca_model, scaler, imputer


def plot_pc1_pc2(pca_df, output_path=None):
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
        raise ValueError("A PCA precisa ter PC1 e PC2 para gerar o plot.")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="label",
        alpha=0.75,
        s=45,
        edgecolor="none",
    )
    plt.title("PCA - medias das 12 derivacoes por segmento")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=220, bbox_inches="tight")

    plt.show()


def main(
    input_csv=None,
    output_dir=None,
    save_outputs=True,
):
    root = project_root()
    if input_csv is None:
        input_csv = root / "data" / "statistical_analysis_outputs" / "descriptive_statistics_segmented.csv"
    if output_dir is None:
        output_dir = root / "data" / "statistical_analysis_outputs" / "pca"

    df = load_mean_statistics(input_csv)
    pca_df, pca_model, scaler, imputer = run_mean_pca(df, n_components=6)

    output_path = Path(output_dir)
    if save_outputs:
        output_path.mkdir(parents=True, exist_ok=True)
        pca_df.to_csv(output_path / "mean_only_pca_scores.csv", index=False)
        plot_pc1_pc2(pca_df, output_path=output_path / "mean_only_pc1_pc2.png")
    else:
        plot_pc1_pc2(pca_df)

    return pca_df, pca_model, scaler, imputer


if __name__ == "__main__":
    main()
