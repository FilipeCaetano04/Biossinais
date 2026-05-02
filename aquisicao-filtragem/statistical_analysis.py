import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


LEADS: List[str] = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_raw_data(csv_path: str = "../data/raw_data.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in LEADS:
        if col not in df.columns:
            raise ValueError(f"Derivacao ausente no dataset: {col}")
    return df


def descriptive_statistics(df: pd.DataFrame, leads: List[str]) -> pd.DataFrame:
    desc = df[leads].describe().T
    desc["variance"] = df[leads].var()
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["outlier_pct"] = 0.0

    for lead in leads:
        q1 = desc.loc[lead, "25%"]
        q3 = desc.loc[lead, "75%"]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df[lead] < lower) | (df[lead] > upper)
        desc.loc[lead, "outlier_pct"] = 100.0 * outlier_mask.mean()

    rename_map = {
        "count": "count",
        "mean": "mean",
        "std": "std",
        "min": "min",
        "25%": "q1",
        "50%": "median",
        "75%": "q3",
        "max": "max",
    }
    desc = desc.rename(columns=rename_map)
    ordered_cols = ["count", "mean", "median", "variance", "std", "min", "q1", "q3", "iqr", "max", "outlier_pct"]
    return desc[ordered_cols]


def plot_histograms(df: pd.DataFrame, leads: List[str], output_path: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()

    for i, lead in enumerate(leads):
        sns.histplot(df[lead], bins=60, kde=True, color="#2f6c8f", alpha=0.85, ax=axes[i])
        axes[i].set_title(f"Histograma - {lead}")
        axes[i].set_xlabel("Amplitude")
        axes[i].set_ylabel("Frequencia")
        axes[i].grid(alpha=0.2)

    fig.suptitle("E3 - Histogramas por Derivacao", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, leads: List[str], output_path: str) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))
    long_df = df[leads].melt(var_name="Derivacao", value_name="Amplitude")
    sns.boxplot(data=long_df, x="Derivacao", y="Amplitude", color="#76b7b2", fliersize=2)
    plt.title("E3 - Boxplot das 12 Derivacoes")
    plt.xlabel("Derivacao")
    plt.ylabel("Amplitude")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def sample_points_per_label(df: pd.DataFrame, max_points_per_label: int = 100000, random_state: int = 42) -> pd.DataFrame:
    sampled = []
    for label, group in df.groupby("label"):
        n_take = min(len(group), max_points_per_label)
        sampled.append(group.sample(n=n_take, random_state=random_state) if n_take < len(group) else group)
    return pd.concat(sampled, ignore_index=True)


def sample_balanced_by_ecg(df: pd.DataFrame, max_ecg_per_label: int = 120, random_state: int = 42) -> pd.DataFrame:
    ecg_label = df[["ecg_id", "label"]].drop_duplicates()
    counts = ecg_label["label"].value_counts()
    min_count = int(counts.min())
    n_per_label = min(min_count, max_ecg_per_label)

    selected_ids = []
    for label, group in ecg_label.groupby("label"):
        sampled = group.sample(n=n_per_label, random_state=random_state)
        selected_ids.extend(sampled["ecg_id"].tolist())

    return df[df["ecg_id"].isin(selected_ids)].copy()


def plot_boxplot_stratified_by_label(df: pd.DataFrame, leads: List[str], output_path: str, max_points_per_label: int = 80000) -> None:
    sns.set_theme(style="whitegrid")
    df_plot = sample_points_per_label(df, max_points_per_label=max_points_per_label)
    long_df = df_plot[["label"] + leads].melt(id_vars="label", var_name="Derivacao", value_name="Amplitude")

    g = sns.catplot(
        data=long_df,
        x="Derivacao",
        y="Amplitude",
        col="label",
        kind="box",
        col_wrap=3,
        sharey=True,
        showfliers=False,
        color="#76b7b2",
        height=3.2,
        aspect=1.35,
    )
    g.fig.suptitle("E3 - Boxplot Estratificado por Classe", y=1.02)
    g.set_axis_labels("Derivacao", "Amplitude")
    g.tight_layout()
    g.savefig(output_path, dpi=220)
    plt.close(g.fig)


def plot_boxplot_balanced_for_visualization(df: pd.DataFrame, leads: List[str], output_path: str, max_ecg_per_label: int = 120) -> None:
    sns.set_theme(style="whitegrid")
    df_balanced = sample_balanced_by_ecg(df, max_ecg_per_label=max_ecg_per_label)
    long_df = df_balanced[["label"] + leads].melt(id_vars="label", var_name="Derivacao", value_name="Amplitude")

    plt.figure(figsize=(15, 7))
    sns.boxplot(data=long_df, x="Derivacao", y="Amplitude", hue="label", showfliers=False, linewidth=0.7)
    plt.title("E3 - Boxplot Balanceado por Classe (Amostragem por ECG)")
    plt.xlabel("Derivacao")
    plt.ylabel("Amplitude")
    plt.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_qq(df: pd.DataFrame, leads: List[str], output_path: str, sample_n: int = 5000) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()

    for i, lead in enumerate(leads):
        series = df[lead].dropna()
        if len(series) > sample_n:
            series = series.sample(sample_n, random_state=42)
        stats.probplot(series, dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot - {lead}")
        axes[i].grid(alpha=0.2)

    fig.suptitle("E3 - Q-Q Plots por Derivacao", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def correlation_analysis(df: pd.DataFrame, leads: List[str]) -> pd.DataFrame:
    return df[leads].corr(method="pearson")


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: str) -> None:
    sns.set_theme(style="white")
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_df, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, linewidths=0.5)
    plt.title("E3 - Heatmap de Correlacao (Pearson)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def validate_einthoven_goldberger(df: pd.DataFrame) -> pd.DataFrame:
    residual_iii = df["III"] - (df["II"] - df["I"])
    residual_avr = df["AVR"] + 0.5 * (df["I"] + df["II"])
    residual_avl = df["AVL"] - 0.5 * (df["I"] - df["III"])
    residual_avf = df["AVF"] - 0.5 * (df["II"] + df["III"])

    validations: Dict[str, Dict[str, float]] = {
        "Einthoven_III_eq_II_minus_I": {
            "corr": df["III"].corr(df["II"] - df["I"]),
            "mae": np.abs(residual_iii).mean(),
            "rmse": np.sqrt(np.mean(residual_iii**2)),
        },
        "Goldberger_AVR_eq_-0.5_I_plus_II": {
            "corr": df["AVR"].corr(-0.5 * (df["I"] + df["II"])),
            "mae": np.abs(residual_avr).mean(),
            "rmse": np.sqrt(np.mean(residual_avr**2)),
        },
        "Goldberger_AVL_eq_0.5_I_minus_III": {
            "corr": df["AVL"].corr(0.5 * (df["I"] - df["III"])),
            "mae": np.abs(residual_avl).mean(),
            "rmse": np.sqrt(np.mean(residual_avl**2)),
        },
        "Goldberger_AVF_eq_0.5_II_plus_III": {
            "corr": df["AVF"].corr(0.5 * (df["II"] + df["III"])),
            "mae": np.abs(residual_avf).mean(),
            "rmse": np.sqrt(np.mean(residual_avf**2)),
        },
    }

    return pd.DataFrame(validations).T.reset_index().rename(columns={"index": "equation"})


def run_e3(
    input_csv: str = "../data/raw_data.csv",
    output_dir: str = "../data/e3_outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df_raw = load_raw_data(input_csv)

    stats_df = descriptive_statistics(df_raw, LEADS)
    corr_df = correlation_analysis(df_raw, LEADS)
    integrity_df = validate_einthoven_goldberger(df_raw)
    ecg_label_counts = df_raw[["ecg_id", "label"]].drop_duplicates()["label"].value_counts().rename("ecg_count")

    stats_df.to_csv(os.path.join(output_dir, "descriptive_statistics.csv"), index=True)
    corr_df.to_csv(os.path.join(output_dir, "correlation_matrix.csv"), index=True)
    integrity_df.to_csv(os.path.join(output_dir, "einthoven_goldberger_validation.csv"), index=False)
    ecg_label_counts.to_csv(os.path.join(output_dir, "ecg_counts_per_label.csv"), index=True)

    plot_histograms(df_raw, LEADS, os.path.join(output_dir, "histograms_leads.png"))
    plot_boxplot(df_raw, LEADS, os.path.join(output_dir, "boxplot_leads.png"))
    plot_boxplot_stratified_by_label(df_raw, LEADS, os.path.join(output_dir, "boxplot_stratified_by_label.png"))
    plot_boxplot_balanced_for_visualization(df_raw, LEADS, os.path.join(output_dir, "boxplot_balanced_by_label.png"))
    plot_qq(df_raw, LEADS, os.path.join(output_dir, "qqplot_leads.png"))
    plot_correlation_heatmap(corr_df, os.path.join(output_dir, "correlation_heatmap.png"))

    print("E3 concluido com sucesso.")
    print(f"Arquivo de entrada: {input_csv}")
    print(f"Saidas geradas em: {output_dir}")


if __name__ == "__main__":
    run_e3()
