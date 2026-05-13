from pathlib import Path

import pandas as pd

from src.scripts.train_autoencoder import run_autoencoder_from_dataframe


def main() -> None:
    features_stats_csv = Path(
        "data/batch_outputs_without_outliers/features_stats_all.csv"
    )
    features_wavelet_csv = Path(
        "data/batch_outputs_without_outliers/features_wavelet_all.csv"
    )
    output_root = Path("data/autoencoder_previous_run_outputs_without_outliers")

    if not features_stats_csv.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {features_stats_csv}")
    if not features_wavelet_csv.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {features_wavelet_csv}")

    df_features_stats = pd.read_csv(features_stats_csv)
    df_features_wavelet = pd.read_csv(features_wavelet_csv)

    result_stats = run_autoencoder_from_dataframe(
        df=df_features_stats,
        output_dir=output_root / "autoencoder_feature_stats",
        latent_dim=15,
        num_epochs=200,
        n_pca_components=3,
        save_loss_artifacts=True,
    )

    result_wavelet = run_autoencoder_from_dataframe(
        df=df_features_wavelet,
        output_dir=output_root / "autoencoder_wavelet",
        latent_dim=15,
        num_epochs=200,
        n_pca_components=3,
        save_loss_artifacts=True,
    )

    print("Autoencoder (CSVs anteriores) concluido com sucesso.")
    print(
        "Feature stats - curva de perda:",
        result_stats["loss_plot_path"],
    )
    print(
        "Feature wavelet - curva de perda:",
        result_wavelet["loss_plot_path"],
    )


if __name__ == "__main__":
    main()
