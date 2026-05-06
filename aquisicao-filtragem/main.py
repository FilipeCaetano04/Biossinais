from pathlib import Path

from evidence_quality import CreateDataRaw, SignalQualityEvaluator
from pca_from_statistical_analysis import main as run_pca_statistic
from statistical_analysis import run_statistical_analysis


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

    print("[1/3] Rodando evidence_quality...")
    df_raw = CreateDataRaw.create_dataframe(
        number_of_pacients=number_of_pacients,
        data_path=data_path,
        data_label=data_label,
        scp_path=scp_path,
        shuffle=shuffle,
        to_csv=True,
    )
    df_quality = SignalQualityEvaluator.evaluate_quality(df_raw, fs=fs, window_sec=window_sec)
    df_quality.to_csv(quality_csv, index=False)
    print(f"Raw salvo em: {raw_csv}")
    print(f"Quality salvo em: {quality_csv}")

    print("[2/3] Rodando statistical_analysis...")
    run_statistical_analysis(
        input_csv=str(raw_csv),
        quality_csv=str(quality_csv),
        output_dir=str(stats_output_dir),
    )

    print("[3/3] Rodando pca_statistic...")
    run_pca_statistic(
        input_csv=stats_output_dir / "descriptive_statistics_segmented.csv",
        output_dir=pca_output_dir,
        save_outputs=True,
    )

    print("Pipeline concluido com sucesso.")
    print(f"Saidas estatisticas em: {stats_output_dir}")
    print(f"Saidas PCA em: {pca_output_dir}")


if __name__ == "__main__":
    run_pipeline()
