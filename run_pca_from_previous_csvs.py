from pathlib import Path

import pandas as pd

from src.aquisicao_filtragem.ica import PCA_SIMPLE, plotar_ica_estatico
from src.dimensionality_reduction.pca_from_statistical_analysis import (
    main as run_pca_statistic,
)


def build_variance_table(pca_model) -> pd.DataFrame:
    explained_ratio = pd.Series(pca_model.explained_variance_ratio_, dtype=float)
    cumulative_ratio = explained_ratio.cumsum()
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(explained_ratio))],
            "explained_variance_ratio": explained_ratio.values,
            "explained_variance_pct": explained_ratio.values * 100.0,
            "cumulative_variance_ratio": cumulative_ratio.values,
            "cumulative_variance_pct": cumulative_ratio.values * 100.0,
        }
    )


def main() -> None:
    output_dir = Path("data/pca_previous_run_outputs")
    pca_output_dir = output_dir / "pca"
    ica_output_dir = output_dir / "ica"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = Path(
        "data/statistical_analysis_outputs/descriptive_statistics_segmented.csv"
    )
    features_stats_csv = Path("data/batch_outputs/features_stats_all.csv")
    features_wavelet_csv = Path("data/batch_outputs/features_wavelet_all.csv")

    if not stats_csv.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {stats_csv}")
    if not features_stats_csv.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {features_stats_csv}")
    if not features_wavelet_csv.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {features_wavelet_csv}")

    _, pca_stat_model, _, _ = run_pca_statistic(
        input_csv=stats_csv,
        output_dir=pca_output_dir,
        save_outputs=True,
    )
    build_variance_table(pca_stat_model).to_csv(
        output_dir / "pca_variance_statistical.csv", index=False
    )

    df_features_stats = pd.read_csv(features_stats_csv)
    df_pca_features_stats, pca_feature_stats_model, _, _ = PCA_SIMPLE(
        df_features_stats, n_components=5, return_metadata=True
    )
    plotar_ica_estatico(
        df_pca_features_stats,
        "PCA FEATURE STATS",
        output_dir=ica_output_dir,
        file_prefix="pca_feature_stats",
        show_plot=False,
    )
    build_variance_table(pca_feature_stats_model).to_csv(
        output_dir / "pca_variance_feature_stats.csv", index=False
    )

    df_features_wavelet = pd.read_csv(features_wavelet_csv)
    df_pca_features_wavelet, pca_feature_wavelet_model, _, _ = PCA_SIMPLE(
        df_features_wavelet, n_components=5, return_metadata=True
    )
    plotar_ica_estatico(
        df_pca_features_wavelet,
        "PCA FEATURES WAVELET",
        output_dir=ica_output_dir,
        file_prefix="pca_features_wavelet",
        show_plot=False,
    )
    build_variance_table(pca_feature_wavelet_model).to_csv(
        output_dir / "pca_variance_feature_wavelet.csv", index=False
    )

    print("PCA executado com sucesso usando CSVs da ultima run.")
    print(f"Entrada estatistica: {stats_csv}")
    print(f"Entrada feature stats: {features_stats_csv}")
    print(f"Entrada feature wavelet: {features_wavelet_csv}")
    print(f"Saidas em: {output_dir}")


if __name__ == "__main__":
    main()
