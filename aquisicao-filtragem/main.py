from pathlib import Path

from evidence_quality import CreateDataRaw, SignalQualityEvaluator
from pca_from_statistical_analysis import main as run_pca_statistic
from statistical_analysis import run_statistical_analysis
from signal_cleaning_validation import ECGSignalCleaner
from feature_extraction_fft import validation_extraction
from ica import *
from wavelet import gerar_features_clinicas


def run_pipeline(
    number_of_pacients: int = 1000,
    data_path: str = "../ignored_data/00000/",
    data_label: str = "../data500/ptbxl_database.csv",
    scp_path: str = "../data500/scp_statements.csv",
    shuffle: bool = False,
    fs: int = 500,
    window_sec: int = 2,
) -> None:
    data_dir = Path("../data")
    stats_output_dir = data_dir / "statistical_analysis_outputs"
    pca_output_dir = stats_output_dir / "pca"
    raw_csv = data_dir / "raw_data.csv"
    quality_csv = data_dir / "quality_data_raw.csv"

    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Rodando evidence_quality + filtro ...")
    df_raw = CreateDataRaw.create_dataframe(
        number_of_pacients=number_of_pacients,
        data_path=data_path,
        data_label=data_label,
        scp_path=scp_path,
        shuffle=shuffle,
        to_csv=True,
    )
    df_raw_filtered = ECGSignalCleaner.clean_signals(df_raw,["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])
    df_quality = SignalQualityEvaluator.evaluate_quality(df_raw_filtered, fs=fs, window_sec=window_sec)
    #df_quality.to_csv(quality_csv, index=False)
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
    df_features = validation_extraction(df_raw_filtered,df_filtered_sqi)
    #opcional - remove outliers:
    #df_features = remove_outliers(df_features)
    #opcional balanceia pelo menor:
    #df_features = SignalQualityEvaluator.balancear_classes_undersampling(df_features, max_ratio=2)

    print("[4/5] Rodando extração de features via wavelet")
    df_features_novas = gerar_features_clinicas(df_raw, df_filtered_sqi)

    print("[5/5] Rodando pca´s e ica")
    run_pca_statistic(
        input_csv=stats_output_dir / "descriptive_statistics_segmented.csv",
        output_dir=pca_output_dir,
        save_outputs=True,
    )
    df_ica_features_stats = ICA(df_features, feature_cols=[col for col in df_features.columns if col not in ["ecg_id","segment_id","label"]])
    df_pca_features_stats = PCA_SIMPLE(df_features, n_components=5)
    plotar_ica_estatico(df_ica_features_stats,"ICA FEATURE STATS")
    plotar_ica_estatico(df_pca_features_stats, "PCA FEATURE STATS")
    df_ica = ICA(df_features_novas, feature_cols=[col for col in df_features_novas.columns if col not in ["ecg_id","segment_id","label"]])
    df_pca = PCA_SIMPLE(df_features_novas, n_components=5)
    plotar_ica_estatico(df_ica, "ICA FEATURES WAVELET")
    plotar_ica_estatico(df_pca, "PCA FEATURES WAVELET")

    print("Pipeline concluido com sucesso.")
    print(f"Saidas estatisticas em: {stats_output_dir}")
    print(f"Saidas PCA em: {pca_output_dir}")


if __name__ == "__main__":
    run_pipeline()
