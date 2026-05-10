"""
Pipeline de limpeza de ECG + validacao estatistica.

Objetivo deste script:
1) Ler um sinal de ECG do PTB-XL (arquivo WFDB)
2) Limpar o sinal em 3 etapas:
   - Notch 50 Hz (ruido da rede eletrica)
   - Band-pass Butterworth ordem 4 (0.5 a 40 Hz)
   - Correcao de baseline por mediana movel
3) Mostrar sinal antes/depois
4) Rodar teste t pareado para verificar se a media do sinal foi preservada
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy import signal, stats


@dataclass
class ValidationResult:
    """Guarda os resultados de validacao para uma derivacao."""

    lead: str
    n_samples: int
    mean_raw: float
    mean_clean: float
    mean_diff: float
    std_diff: float
    t_statistic: float
    p_value: float
    ci95_low: float
    ci95_high: float
    cohens_d_paired: float
    interpretation: str


def _make_odd(kernel_size: int, signal_length: int) -> int:
    """
    Garante que o tamanho de janela da mediana seja impar e valido.

    Filtro de mediana exige janela impar.
    Tambem nao pode ser maior que o tamanho do sinal.
    """
    k = max(3, int(kernel_size))

    if k >= signal_length:
        k = signal_length - 1

    if k % 2 == 0:
        k += 1

    # Caso extremo de sinal muito curto
    if k < 3:
        k = 3

    if k >= signal_length:
        k = signal_length - (1 - signal_length % 2)

    return int(k)


def apply_notch_50hz(ecg_signal: np.ndarray, fs: int, q: float = 30.0) -> np.ndarray:
    """
    Remove ruido de rede eletrica de 50 Hz usando filtro notch.

    - fs: frequencia de amostragem
    - q: fator de qualidade (maior q = notch mais estreito)
    """
    b, a = signal.iirnotch(w0=50.0, Q=q, fs=fs)
    return signal.filtfilt(b, a, ecg_signal)


def apply_bandpass_butterworth(ecg_signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Passa-banda Butterworth de ordem 4 entre 0.5 e 40 Hz.

    - 0.5 Hz ajuda a remover oscilacao lenta (ex.: respiracao)
    - 40 Hz ajuda a remover ruido de alta frequencia
    """
    b, a = signal.butter(N=4, Wn=[0.5, 40.0], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, ecg_signal)


def estimate_baseline_median(
    ecg_signal: np.ndarray,
    fs: int,
    win_short_sec: float = 0.2,
    win_long_sec: float = 0.6,
) -> np.ndarray:
    """
    Estima a baseline usando duas medianas moveis em sequencia.

    Essa estrategia eh comum em ECG para capturar drift lento sem destruir
    os complexos QRS.
    """
    short_kernel = _make_odd(int(win_short_sec * fs), len(ecg_signal))
    long_kernel = _make_odd(int(win_long_sec * fs), len(ecg_signal))

    baseline_step1 = signal.medfilt(ecg_signal, kernel_size=short_kernel)
    baseline = signal.medfilt(baseline_step1, kernel_size=long_kernel)
    return baseline


def clean_ecg_signal(
    ecg_signal: np.ndarray, fs: int, preserve_mean: bool = True
) -> np.ndarray:
    """
    Executa o pipeline completo de limpeza, na ordem pedida:
    1) Notch 50Hz
    2) Band-pass 0.5-40Hz
    3) Correcao de baseline por mediana
    """
    notch_filtered = apply_notch_50hz(ecg_signal, fs=fs)
    band_filtered = apply_bandpass_butterworth(notch_filtered, fs=fs)
    baseline = estimate_baseline_median(band_filtered, fs=fs)
    baseline_corrected = band_filtered - baseline

    # recoloca a media original para garantir que o nivel DC do sinal
    # (media global) nao seja deslocado pelo processamento.
    if preserve_mean:
        baseline_corrected = (
            baseline_corrected - np.mean(baseline_corrected) + np.mean(ecg_signal)
        )

    return baseline_corrected


def paired_t_validation(
    raw_signal: np.ndarray, clean_signal: np.ndarray, lead: str
) -> ValidationResult:
    """
    Aplica teste t pareado ponto-a-ponto: bruto vs limpo.

    O teste t pareado responde: "a media das diferencas e zero?"
    Se p-value for alto (tipicamente > 0.05), nao ha evidencia forte de
    mudanca na media.
    """
    if len(raw_signal) != len(clean_signal):
        raise ValueError("raw_signal e clean_signal precisam ter o mesmo tamanho")

    diff = clean_signal - raw_signal
    ttest = stats.ttest_rel(clean_signal, raw_signal)

    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    sem_diff = stats.sem(diff)
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean_diff,
        scale=sem_diff,
    )

    if std_diff == 0.0:
        cohens_d = 0.0
    else:
        cohens_d = mean_diff / std_diff

    if ttest.pvalue > 0.05:
        interpretation = "Nao rejeita H0: sem evidencia de mudanca da media."
    else:
        interpretation = (
            "Rejeita H0: diferenca de media detectada. "
            "Interpretar junto com mean_diff e Cohen's d."
        )

    return ValidationResult(
        lead=lead,
        n_samples=n,
        mean_raw=float(np.mean(raw_signal)),
        mean_clean=float(np.mean(clean_signal)),
        mean_diff=mean_diff,
        std_diff=std_diff,
        t_statistic=float(ttest.statistic),
        p_value=float(ttest.pvalue),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        cohens_d_paired=float(cohens_d),
        interpretation=interpretation,
    )


def plot_before_after(
    time_axis: np.ndarray,
    raw_signal: np.ndarray,
    clean_signal: np.ndarray,
    lead_name: str,
    output_path: Path,
) -> None:
    """Gera grafico comparando sinal bruto e limpo."""
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal, color="lightgray", linewidth=1.0, label="Bruto")
    plt.plot(time_axis, clean_signal, color="tab:red", linewidth=1.1, label="Limpo")
    plt.title(f"ECG antes/depois da limpeza - Derivacao {lead_name}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_cleaning_and_validation(record_path: str, output_dir: str) -> pd.DataFrame:
    """
    Executa limpeza e validacao para todas as derivacoes de um registro.

    Salva:
    - csv com sinal limpo
    - csv com estatisticas do teste t pareado
    - grafico antes/depois para cada derivacao
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    record = wfdb.rdrecord(record_path)
    fs = int(record.fs)
    lead_names = record.sig_name
    raw_matrix = record.p_signal

    cleaned_matrix = np.zeros_like(raw_matrix)
    validation_rows: List[Dict[str, float | str | int]] = []

    # Tempo para plot (em segundos)
    time_axis = np.arange(raw_matrix.shape[0]) / fs

    for idx, lead in enumerate(lead_names):
        raw_lead = raw_matrix[:, idx]
        clean_lead = clean_ecg_signal(raw_lead, fs)
        cleaned_matrix[:, idx] = clean_lead

        validation = paired_t_validation(raw_lead, clean_lead, lead)
        validation_rows.append(validation.__dict__)

        plot_path = output / f"before_after_{lead}.png"
        plot_before_after(time_axis, raw_lead, clean_lead, lead, plot_path)

    # Salva sinais limpos em CSV
    df_clean = pd.DataFrame(cleaned_matrix, columns=lead_names)
    df_clean.insert(0, "TEMPO", time_axis)
    df_clean.to_csv(output / "cleaned_signal.csv", index=False)

    # Salva resultados estatisticos
    df_validation = pd.DataFrame(validation_rows)
    df_validation.to_csv(output / "paired_t_validation.csv", index=False)

    print("Pipeline concluido.")
    print(f"Registro processado: {record_path}")
    print(f"Frequencia de amostragem: {fs} Hz")
    print(f"Saidas em: {output.resolve()}")
    print("\nResumo por derivacao (lead, p-value, mean_diff, Cohen d):")

    for _, row in df_validation.iterrows():
        print(
            f"- {row['lead']}: p={row['p_value']:.4g} | "
            f"mean_diff={row['mean_diff']:.6f} | "
            f"d={row['cohens_d_paired']:.4f}"
        )

    return df_validation


import random
from scipy.signal import butter, filtfilt, iirnotch


class ECGSignalCleaner:
    @classmethod
    def _apply_notch_filter(
        cls, data: np.ndarray, fs: int, freq_to_remove=50.0, q=30.0
    ) -> np.ndarray:
        """Remove ruído da rede elétrica (50Hz / 60Hz)."""
        nyquist = fs / 2
        # Verifica se a frequência de amostragem suporta o filtro Notch
        if freq_to_remove >= nyquist:
            return data

        b, a = iirnotch(freq_to_remove, q, fs)
        return filtfilt(b, a, data)

    @classmethod
    def _apply_bandpass_filter(
        cls, data: np.ndarray, fs: int, lowcut=0.5, highcut=40.0, order=4
    ) -> np.ndarray:
        """
        High-pass (> 0.5Hz): Remove a oscilação da respiração.
        Low-pass (< 40Hz): Remove ruído muscular de alta frequência.
        """
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        if high >= 1.0:
            b, a = butter(order, low, btype="high")
        else:
            b, a = butter(order, [low, high], btype="band")

        # padlen dinâmico para evitar erros em segmentos muito curtos
        padlen = min(len(data) - 1, 150)
        return filtfilt(b, a, data, padlen=padlen)

    @classmethod
    def clean_signals(
        cls, df_raw: pd.DataFrame, leads: list, fs: int = 500, saveToCSV=False
    ) -> pd.DataFrame:
        """Aplica os filtros por paciente e segmento."""
        print(f"🧹 Iniciando limpeza do sinal (Frequência: {fs}Hz)...")
        df_clean: pd.DataFrame = df_raw.copy()

        def process_group(group):
            g = group.copy()
            for lead in leads:
                signal = g[lead].values
                # 1. Tira oscilação da rede elétrica (50Hz)
                signal = cls._apply_notch_filter(signal, fs, freq_to_remove=50.0)
                # 2. Tira Respiração (0.5Hz) e Ruído Muscular (40Hz)
                signal = cls._apply_bandpass_filter(
                    signal, fs, lowcut=0.5, highcut=40.0
                )
                g[lead] = signal
            return g

        group_cols = ["ecg_id"]
        if "segment_id" in df_clean.columns:
            group_cols.append("segment_id")

        # O apply garante que o filtro zere e reinicie para cada paciente/segmento
        df_clean = df_clean.groupby(group_cols, group_keys=False).apply(process_group)

        print(f"✅ Limpeza concluída para as {len(leads)} derivações!")
        if saveToCSV:
            df_clean.to_csv("data/filtered_data.csv")
        return df_clean


def plotar_comparacao_filtros(
    df_sujo: pd.DataFrame, df_limpo: pd.DataFrame, freq=500, n_classes=5
):
    """
    Sorteia um paciente para cada diagnóstico e plota a derivação II
    sobrepondo o sinal antes e depois do filtro.
    """
    # 1. Identificar coluna de label
    label_col = "label_clinico" if "label_clinico" in df_sujo.columns else "label"
    if label_col not in df_sujo.columns:
        possiveis = [c for c in df_sujo.columns if c.startswith("label")]
        label_col = possiveis[0] if possiveis else "label"

    # Pega as top N classes com mais dados para garantir que tem pacientes
    classes_disponiveis = df_sujo[label_col].value_counts().index.tolist()
    classes_selecionadas = classes_disponiveis[:n_classes]

    # Prepara a Figura
    fig, axes = plt.subplots(
        nrows=n_classes, ncols=1, figsize=(12, 8), sharex=True, sharey=True
    )
    if n_classes == 1:
        axes = [axes]

    for ax, classe in zip(axes, classes_selecionadas):
        # Isola os pacientes dessa classe específica
        pacientes_da_classe = df_sujo[df_sujo[label_col] == classe]["ecg_id"].unique()

        # Sorteia UM paciente
        paciente_sorteado = random.choice(pacientes_da_classe)

        # Extrai os dados do paciente SUJO e do LIMPO
        sinal_sujo = df_sujo[df_sujo["ecg_id"] == paciente_sorteado]
        sinal_limpo = df_limpo[df_limpo["ecg_id"] == paciente_sorteado]

        # Pega a derivação II
        coluna_lead = "II" if "II" in df_sujo.columns else "lead_II"
        y_sujo = sinal_sujo[coluna_lead].values
        y_limpo = sinal_limpo[coluna_lead].values

        # Eixo do tempo real baseado na frequência (500Hz)
        tempo = np.arange(len(y_sujo)) / freq

        # 🎨 PLOTAGEM
        # Sinal Sujo (Fundo vermelho claro, mais espesso e transparente)
        ax.plot(
            tempo,
            y_sujo,
            color="lightcoral",
            linewidth=2.5,
            alpha=0.6,
            label="Sem Filtro (Original)",
        )
        # Sinal Limpo (Frente azul escuro, mais fino e nítido)
        ax.plot(
            tempo, y_limpo, color="#004c99", linewidth=1.2, label="Com Filtro (Limpo)"
        )

        # Ajustes visuais de cada linha
        ax.set_title(
            f"Diagnóstico: {classe} | Paciente Sorteado: ID {paciente_sorteado}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legenda apenas no primeiro gráfico para não poluir
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.xlabel("Tempo (segundos)", fontsize=12, fontweight="bold")
    plt.suptitle(
        "Efeito da Filtragem IIR (Notch 50Hz + Bandpass 0.5-40Hz) na Derivação II",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # run_cleaning_and_validation(
    #     record_path="./ignored_data/00000/00001_hr",
    #     output_dir="./data/cleaning_validation_outputs",
    # )

    root = Path("ignored_data")
    out_root = Path("data/cleaning_validation_outputs_all")
    out_root.mkdir(parents=True, exist_ok=True)
    # todos os registros de todos os subdiretorios
    record_bases = sorted(str(p).replace(".hea", "") for p in root.rglob("*_hr.hea"))
    all_reports = []
    for rec in record_bases[:1]:
        rec_name = Path(rec).name  # ex: 00001_hr
        rec_out = out_root / rec_name
        df_val = run_cleaning_and_validation(record_path=rec, output_dir=str(rec_out))
        # guarda resumo global
        df_val["record"] = rec_name
        all_reports.append(df_val)
    # consolidado final
    if all_reports:
        df_all = pd.concat(all_reports, ignore_index=True)
        df_all.to_csv(out_root / "paired_t_validation_all_records.csv", index=False)
        print(
            f"Consolidado salvo em: {out_root / 'paired_t_validation_all_records.csv'}"
        )
