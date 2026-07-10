from pathlib import Path
import gc

import pandas as pd

from src.aquisicao_filtragem.evidence_quality import (
    CreateDataRaw,
    SignalQualityEvaluator,
)
from src.aquisicao_filtragem.statistical_analysis import (
    LEADS,
    descriptive_statistics_segmented,
    run_statistical_analysis,
)
from src.aquisicao_filtragem.signal_cleaning_validation import ECGSignalCleaner
from src.aquisicao_filtragem.feature_extraction_fft import validation_extraction
from src.aquisicao_filtragem.ica import *
from src.aquisicao_filtragem.wavelet import gerar_features_clinicas
from src.dimensionality_reduction.pca_from_statistical_analysis import (
    main as run_pca_statistic,
)
#from src.scripts.train_autoencoder import run_autoencoder_from_dataframe


def _append_csv(df: pd.DataFrame | None, output_path: Path) -> None:
    if df is None or df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False)


def _chunk_list(values: list[int], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def run_pipeline(
    number_of_pacients: int = 1000,
    process_all: bool = False,
    data_path: str = "./ignored_data",
    data_label: str = "./data500/ptbxl_database.csv",
    scp_path: str = "./data500/scp_statements.csv",
    shuffle: bool = False,
    batch_size: int = 5000,
    fs: int = 500,
    window_sec: int = 2,
    autoencoder_latent_dim: int = 15,
    autoencoder_epochs: int = 200,
) -> None:
    data_dir = Path("./data")
    stats_output_dir = data_dir / "statistical_analysis_outputs_without_outliers"
    features_fft_csv = data_dir/"fft_extracted_features.csv"
    pca_output_dir = stats_output_dir / "pca"
    ica_output_dir = stats_output_dir / "ica"
    autoencoder_stats_output_dir = stats_output_dir / "autoencoder_feature_stats"
    autoencoder_wavelet_output_dir = stats_output_dir / "autoencoder_wavelet"
    batch_output_dir = data_dir / "batch_outputs_without_outliers"
    raw_csv = data_dir / "raw_data_without_outliers.csv"
    quality_csv = data_dir / "quality_data_raw_without_outliers.csv"
    quality_all_csv = batch_output_dir / "quality_data_raw_all.csv"
    features_stats_csv = batch_output_dir / "features_stats_all.csv"
    features_wavelet_csv = batch_output_dir / "features_wavelet_all.csv"
    segmented_stats_csv = stats_output_dir / "descriptive_statistics_segmented.csv"

    data_dir.mkdir(parents=True, exist_ok=True)

    if process_all:
        record_index = CreateDataRaw.build_record_index(data_path)
        available_ids = CreateDataRaw.list_available_record_ids(
            data_path=data_path,
            data_label=data_label,
            record_index=record_index,
        )
        if not available_ids:
            raise ValueError(
                "No available ECG records found for full-dataset processing."
            )
        if shuffle:
            available_ids = (
                pd.Series(available_ids)
                .sample(frac=1, random_state=42)
                .astype(int)
                .tolist()
            )

        total_batches = (len(available_ids) + batch_size - 1) // batch_size
        print(
            f"[ALL RECORDS] Processing {len(available_ids)} records in {total_batches} "
            f"batch(es) of up to {batch_size}."
        )

        for output_file in [
            quality_all_csv,
            features_stats_csv,
            features_wavelet_csv,
            segmented_stats_csv,
        ]:
            if output_file.exists():
                output_file.unlink()

        for batch_idx, batch_ids in enumerate(
            _chunk_list(available_ids, batch_size), start=1
        ):
            print(
                f"[BATCH {batch_idx}/{total_batches}] "
                f"Loading {len(batch_ids)} records ({batch_ids[0]}..{batch_ids[-1]})"
            )

            df_raw = CreateDataRaw.create_dataframe(
                number_of_pacients=len(batch_ids),
                process_all=False,
                data_path=data_path,
                data_label=data_label,
                scp_path=scp_path,
                shuffle=False,
                to_csv=False,
                selected_ids=batch_ids,
                record_index=record_index,
            )
            print(f"[BATCH {batch_idx}/{total_batches}] data loading finished")

            df_raw_filtered = ECGSignalCleaner.clean_signals(
                df_raw,
                [
                    "I",
                    "II",
                    "III",
                    "V3",
                    "V6",
                ],
                fs=fs,
            )
            print(f"[BATCH {batch_idx}/{total_batches}] signal cleaning finished")

            df_quality = SignalQualityEvaluator.evaluate_quality(
                df_raw_filtered, fs=fs, window_sec=window_sec
            )
            _append_csv(df_quality, quality_all_csv)
            print(f"[BATCH {batch_idx}/{total_batches}] quality evaluation finished")

            if df_quality.empty:
                print(
                    f"[BATCH {batch_idx}/{total_batches}] quality output is empty. "
                    "Skipping feature extraction for this batch."
                )
                del df_raw
                del df_raw_filtered
                del df_quality
                gc.collect()
                continue

            df_filtered_sqi = SignalQualityEvaluator.remove_bad_data(df_quality)
            if df_filtered_sqi.empty:
                df_features = pd.DataFrame()
                df_features_novas = pd.DataFrame()
            else:
                df_features = validation_extraction(
                    df_raw_filtered, df_filtered_sqi, freq=fs
                )
                print(
                    f"[BATCH {batch_idx}/{total_batches}] FFT feature extraction finished"
                )

                df_features_novas = gerar_features_clinicas(
                    df_raw, df_filtered_sqi, freq=fs
                )
                print(
                    f"[BATCH {batch_idx}/{total_batches}] wavelet feature extraction finished"
                )

                if not df_features.empty:
                    print(
                        f"[BATCH {batch_idx}/{total_batches}] removing outliers (FFT features)..."
                    )
                    df_features = remove_outliers(df_features, threshold=0.001)
                    print(
                        f"[BATCH {batch_idx}/{total_batches}] FFT outlier removal finished"
                    )

                if not df_features_novas.empty:
                    print(
                        f"[BATCH {batch_idx}/{total_batches}] removing outliers (wavelet features)..."
                    )
                    df_features_novas = remove_outliers(
                        df_features_novas, threshold=0.001
                    )
                    print(
                        f"[BATCH {batch_idx}/{total_batches}] wavelet outlier removal finished"
                    )

            _append_csv(df_features, features_stats_csv)
            _append_csv(df_features_novas, features_wavelet_csv)

            df_segmented = descriptive_statistics_segmented(
                df_raw, df_quality, LEADS, fs=fs
            )
            _append_csv(df_segmented, segmented_stats_csv)
            print(f"[BATCH {batch_idx}/{total_batches}] segmented statistics finished")
            print(f"[BATCH {batch_idx}/{total_batches}] done")

            del df_raw
            del df_raw_filtered
            del df_quality
            del df_filtered_sqi
            del df_features
            del df_features_novas
            del df_segmented
            gc.collect()

        print("[ALL RECORDS] Batch processing finished. Running final models...")

        run_pca_statistic(
            input_csv=segmented_stats_csv,
            output_dir=pca_output_dir,
            save_outputs=True,
        )

        if not features_stats_csv.exists() or not features_wavelet_csv.exists():
            raise ValueError(
                "Feature files were not generated. Check batch logs for failures."
            )

        df_features = pd.read_csv(features_stats_csv)
        df_features_novas = pd.read_csv(features_wavelet_csv)

        # print("[ALL RECORDS] Removing outliers from FFT features...")
        # df_features = remove_outliers(df_features, threshold=0.001)
        # print("[ALL RECORDS] Removing outliers from wavelet features...")
        # df_features_novas = remove_outliers(df_features_novas, threshold=0.001)

        # df_features.to_csv(
        #     batch_output_dir / "features_stats_all_without_outliers.csv", index=False
        # )
        # df_features_novas.to_csv(
        #     batch_output_dir / "features_wavelet_all_without_outliers.csv", index=False
        # )

        df_ica_features_stats = ICA(
            df_features,
            feature_cols=[
                col
                for col in df_features.columns
                if col not in ["ecg_id", "segment_id", "label"]
            ],
        )
        df_pca_features_stats = PCA_SIMPLE(df_features, n_components=5)
        plotar_ica_estatico(
            df_ica_features_stats,
            "ICA FEATURE STATS",
            output_dir=ica_output_dir,
            file_prefix="ica_feature_stats",
            show_plot=False,
        )
        plotar_ica_estatico(
            df_pca_features_stats,
            "PCA FEATURE STATS",
            output_dir=ica_output_dir,
            file_prefix="pca_feature_stats",
            show_plot=False,
        )

        df_ica = ICA(
            df_features_novas,
            feature_cols=[
                col
                for col in df_features_novas.columns
                if col not in ["ecg_id", "segment_id", "label"]
            ],
        )
        df_pca = PCA_SIMPLE(df_features_novas, n_components=5)
        plotar_ica_estatico(
            df_ica,
            "ICA FEATURES WAVELET",
            output_dir=ica_output_dir,
            file_prefix="ica_features_wavelet",
            show_plot=False,
        )
        plotar_ica_estatico(
            df_pca,
            "PCA FEATURES WAVELET",
            output_dir=ica_output_dir,
            file_prefix="pca_features_wavelet",
            show_plot=False,
        )

        ae_result_stats = run_autoencoder_from_dataframe(
            df=df_features,
            output_dir=autoencoder_stats_output_dir,
            latent_dim=autoencoder_latent_dim,
            num_epochs=autoencoder_epochs,
            n_pca_components=3,
        )
        ae_result_wavelet = run_autoencoder_from_dataframe(
            df=df_features_novas,
            output_dir=autoencoder_wavelet_output_dir,
            latent_dim=autoencoder_latent_dim,
            num_epochs=autoencoder_epochs,
            n_pca_components=3,
        )
        print(
            f"Autoencoder FFT concluido. Latente: {ae_result_stats['latent_shape']} | "
            f"Saidas: {autoencoder_stats_output_dir}"
        )
        print(
            f"Autoencoder wavelet concluido. Latente: {ae_result_wavelet['latent_shape']} | "
            f"Saidas: {autoencoder_wavelet_output_dir}"
        )
        print("Pipeline (all records, batched) concluido com sucesso.")
        print(f"Saidas quality em: {quality_all_csv}")
        print(f"Saidas features FFT em: {features_stats_csv}")
        print(f"Saidas features wavelet em: {features_wavelet_csv}")
        print(f"Saidas estatisticas em: {stats_output_dir}")
        print(f"Saidas PCA em: {pca_output_dir}")
        print(f"Saidas ICA em: {ica_output_dir}")
        print(f"Saidas Autoencoder FFT em: {autoencoder_stats_output_dir}")
        print(f"Saidas Autoencoder wavelet em: {autoencoder_wavelet_output_dir}")
        return

    print("[1/5] Rodando evidence_quality + filtro ...")
    df_raw = CreateDataRaw.create_dataframe(
        number_of_pacients=number_of_pacients,
        process_all=process_all,
        data_path=data_path,
        data_label=data_label,
        scp_path=scp_path,
        shuffle=shuffle,
        to_csv=True,
    )
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_csv, index=False)
    df_raw_filtered = ECGSignalCleaner.clean_signals(
        df_raw,
        ["I", "II", "III", "V3", "V6"],
    )
    df_quality = SignalQualityEvaluator.evaluate_quality(
        df_raw_filtered, fs=fs, window_sec=window_sec
    )
    df_quality.to_csv(quality_csv, index=False)
    df_filtered_sqi = SignalQualityEvaluator.remove_bad_data(df_quality)

    print(f"Raw salvo em: {raw_csv}")
    print(f"Quality salvo em: {quality_csv}")

    print("[2/5] Rodando statistical_analysis...")
    run_statistical_analysis(
        input_csv=str(raw_csv),
        quality_csv=str(quality_csv),
        output_dir=str(stats_output_dir),
    )

    print("[3/5] Rodando extração de features de energia para segmento")
    df_features = validation_extraction(df_raw_filtered, df_filtered_sqi)
    df_features.to_csv(features_fft_csv)
    # print("[3/5] Removendo outliers (FFT features)")
    # df_features = remove_outliers(df_features, threshold=0.001)
    # opcional balanceia pelo menor:
    # df_features = SignalQualityEvaluator.balancear_classes_undersampling(
    #     df_features, max_ratio=2
    # )

    #print("[4/5] Rodando extração de features via wavelet")
    #df_features_novas = gerar_features_clinicas(df_raw, df_filtered_sqi)
    # print("[4/5] Removendo outliers (wavelet features)")
    # df_features_novas = remove_outliers(df_features_novas, threshold=0.001)

    # print("[5/5] Rodando pca´s, ica e autoencoder")
    # run_pca_statistic(
    #     input_csv=stats_output_dir / "descriptive_statistics_segmented.csv",
    #     output_dir=pca_output_dir,
    #     save_outputs=True,
    # )
    # df_ica_features_stats = ICA(
    #     df_features,
    #     feature_cols=[
    #         col
    #         for col in df_features.columns
    #         if col not in ["ecg_id", "segment_id", "label"]
    #     ],
    # )
    # df_pca_features_stats = PCA_SIMPLE(df_features, n_components=5)
    # plotar_ica_estatico(
    #     df_ica_features_stats,
    #     "ICA FEATURE STATS",
    #     output_dir=ica_output_dir,
    #     file_prefix="ica_feature_stats",
    #     show_plot=False,
    # )
    # plotar_ica_estatico(
    #     df_pca_features_stats,
    #     "PCA FEATURE STATS",
    #     output_dir=ica_output_dir,
    #     file_prefix="pca_feature_stats",
    #     show_plot=False,
    # )
    # df_ica = ICA(
    #     df_features_novas,
    #     feature_cols=[
    #         col
    #         for col in df_features_novas.columns
    #         if col not in ["ecg_id", "segment_id", "label"]
    #     ],
    # )
    # df_pca = PCA_SIMPLE(df_features_novas, n_components=5)
    # plotar_ica_estatico(
    #     df_ica,
    #     "ICA FEATURES WAVELET",
    #     output_dir=ica_output_dir,
    #     file_prefix="ica_features_wavelet",
    #     show_plot=False,
    # )
    # plotar_ica_estatico(
    #     df_pca,
    #     "PCA FEATURES WAVELET",
    #     output_dir=ica_output_dir,
    #     file_prefix="pca_features_wavelet",
    #     show_plot=False,
    # )

    # ae_result_stats = run_autoencoder_from_dataframe(
    #     df=df_features,
    #     output_dir=autoencoder_stats_output_dir,
    #     latent_dim=autoencoder_latent_dim,
    #     num_epochs=autoencoder_epochs,
    #     n_pca_components=3,
    # )
    # ae_result_wavelet = run_autoencoder_from_dataframe(
    #     df=df_features_novas,
    #     output_dir=autoencoder_wavelet_output_dir,
    #     latent_dim=autoencoder_latent_dim,
    #     num_epochs=autoencoder_epochs,
    #     n_pca_components=3,
    # )
    # print(
    #     f"Autoencoder FFT concluido. Latente: {ae_result_stats['latent_shape']} | "
    #     f"Saidas: {autoencoder_stats_output_dir}"
    # )
    # print(
    #     f"Autoencoder wavelet concluido. Latente: {ae_result_wavelet['latent_shape']} | "
    #     f"Saidas: {autoencoder_wavelet_output_dir}"
    # )

    # print("Pipeline concluido com sucesso.")
    # print(f"Saidas estatisticas em: {stats_output_dir}")
    # print(f"Saidas PCA em: {pca_output_dir}")
    # print(f"Saidas ICA em: {ica_output_dir}")
    # print(f"Saidas Autoencoder FFT em: {autoencoder_stats_output_dir}")
    # print(f"Saidas Autoencoder wavelet em: {autoencoder_wavelet_output_dir}")


if __name__ == "__main__":
    run_pipeline(process_all=False,number_of_pacients=2000)
