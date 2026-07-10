"""
Avalia detectores baseados em distância (Euclidiana Mínima, Mahalanobis)
com PCA como redutor de dimensionalidade.

Escolhe q na primeira rodada de treino-teste preservando VARIANCIA_ALVO
da variância, plota VE(q) vs q, fixa q para as demais 99 rodadas,
e preenche a tabela de estatísticas de desempenho médio.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as PCA

from src.classification.detectores import (
    distanciaminimacentroide,
    Mahalanobis,
    gerar_linha_resultado,
)

DATA_DIR = Path("data_classification")
RANDOM_SEED = 42
EPOCHS = 100
TRAIN_RATIO = 0.8
VARIANCIA_ALVO = 0.95


def load_data() -> tuple[np.ndarray, np.ndarray]:
    df_norm = pd.read_csv(DATA_DIR / "fft_extracted_features_NORM.csv", index_col=0)
    df_mi = pd.read_csv(DATA_DIR / "fft_extracted_features_MI.csv", index_col=0)
    df = pd.concat([df_norm, df_mi], ignore_index=True)
    df = df.drop(columns=["ecg_id", "segment_id"], errors="ignore")
    df["label"] = df["label"].replace({"NORM": -1, "MI": 1})
    Y = df["label"].values.ravel()
    X = df.drop(columns=["label"]).values.T
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=int)


def plot_ve_vs_q(
    variancia_acumulada: np.ndarray,
    q_escolhido: int,
    output_path: str | Path | None = None,
):
    q_vals = np.arange(1, len(variancia_acumulada) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(q_vals, variancia_acumulada * 100, "b-o", markersize=3)
    plt.axhline(
        y=VARIANCIA_ALVO * 100,
        color="r",
        linestyle="--",
        label=f"Threshold {VARIANCIA_ALVO * 100:.0f}%",
    )
    plt.axvline(
        x=q_escolhido,
        color="g",
        linestyle="--",
        label=f"q = {q_escolhido} ({variancia_acumulada[q_escolhido - 1] * 100:.1f}%)",
    )
    plt.xlabel("q (n\\'umero de componentes)")
    plt.ylabel("Vari\\^ancia Explicada Acumulada VE(q) [%]")
    plt.title("Vari\\^ancia Explicada vs N\\'umero de Componentes PCA")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def escolher_q_primeira_rodada(X: np.ndarray, Y: np.ndarray) -> tuple[int, np.ndarray]:
    n_samples = X.shape[1]
    rng = np.random.default_rng(RANDOM_SEED)

    idx_perm = rng.permutation(n_samples)
    X_shuffled = X[:, idx_perm]
    Y_shuffled = Y[idx_perm]

    idx_neg = np.where(Y_shuffled < 0)[0]
    X_neg = X_shuffled[:, idx_neg]

    n_neg = len(idx_neg)
    n_neg_trn = int(np.floor(TRAIN_RATIO * n_neg))
    X_neg_trn = X_neg[:, :n_neg_trn]

    me = np.mean(X_neg_trn, axis=1, keepdims=True)
    se = np.std(X_neg_trn, axis=1, ddof=1, keepdims=True)
    se[se == 0] = 1e-9
    X_neg_trn_norm = (X_neg_trn - me) / se

    pca_full = PCA(n_components=None)
    pca_full.fit(X_neg_trn_norm.T)
    variancia_ve = np.cumsum(pca_full.explained_variance_ratio_)
    q = int(np.searchsorted(variancia_ve, VARIANCIA_ALVO)) + 1

    return q, variancia_ve


def avaliar_detector_com_pca(
    X: np.ndarray,
    Y: np.ndarray,
    detector_class,
    detector_kwargs: dict | None = None,
    nome: str = "Detector",
    q_fixo: int | None = None,
) -> dict:
    if detector_kwargs is None:
        detector_kwargs = {}

    n_samples = X.shape[1]
    rng = np.random.default_rng(RANDOM_SEED)
    perf_list = []

    pca = PCA(n_components=q_fixo)
    for epoch in range(EPOCHS):
        idx_perm = rng.permutation(n_samples)
        X_shuffled = X[:, idx_perm]
        Y_shuffled = Y[idx_perm]

        idx_neg = np.where(Y_shuffled < 0)[0]
        idx_pos = np.where(Y_shuffled > 0)[0]

        X_neg = X_shuffled[:, idx_neg]
        X_pos = X_shuffled[:, idx_pos]
        Y_neg = Y_shuffled[idx_neg]
        Y_pos = Y_shuffled[idx_pos]

        n_neg = len(idx_neg)
        n_neg_trn = int(np.floor(TRAIN_RATIO * n_neg))
        X_neg_trn = X_neg[:, :n_neg_trn]
        X_neg_tst = X_neg[:, n_neg_trn:]
        Y_neg_tst = Y_neg[n_neg_trn:]

        X_tst = np.hstack((X_pos, X_neg_tst))
        Y_tst = np.concatenate((Y_pos, Y_neg_tst))

        me = np.mean(X_neg_trn, axis=1, keepdims=True)
        se = np.std(X_neg_trn, axis=1, ddof=1, keepdims=True)
        se[se == 0] = 1e-9
        X_neg_trn_norm = (X_neg_trn - me) / se
        X_tst_norm = (X_tst - me) / se

        X_trn = pca.fit_transform(X_neg_trn_norm.T)
        X_tst_pca = pca.transform(X_tst_norm.T)

        detector = detector_class(**detector_kwargs)

        inicio_treino = time.perf_counter()
        detector.fit(X_trn)
        fim_treino = time.perf_counter()

        inicio_teste = time.perf_counter()
        Y_pred = detector.predict(X_tst_pca)
        fim_teste = time.perf_counter()

        tempo_treino = fim_treino - inicio_treino
        tempo_teste = fim_teste - inicio_teste

        vn = np.sum((Y_tst < 0) & (Y_pred < 0))
        fp = np.sum((Y_tst < 0) & (Y_pred > 0))
        fn = np.sum((Y_tst > 0) & (Y_pred < 0))
        vp = np.sum((Y_tst > 0) & (Y_pred > 0))

        n_total = len(Y_tst)
        acc = 100 * (vp + vn) / n_total if n_total > 0 else 0
        sens = 100 * vp / (vp + fn) if (vp + fn) > 0 else 0
        espec = 100 * vn / (vn + fp) if (vn + fp) > 0 else 0
        prec = 100 * vp / (vp + fp) if (vp + fp) > 0 else 0
        f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0

        perf_list.append(
            [acc, sens, espec, prec, f1, tempo_treino * 1000, tempo_teste * 1000]
        )

    PERF = np.array(perf_list)
    return gerar_linha_resultado(nome, PERF)


def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    X, Y = load_data()
    n_norm = np.sum(Y < 0)
    n_anom = np.sum(Y > 0)
    print(f"Dados carregados: X {X.shape}, Y {Y.shape}")
    print(f"  Normais (NEG): {n_norm}, Anômalos (POS): {n_anom}")

    # --- Escolhe q na primeira rodada (PCA no treino) ---
    print("\n--- Primeira rodada: seleção de q ---")
    q, variancia_ve = escolher_q_primeira_rodada(X, Y)
    print(f"q escolhido = {q} (variância explicada = {variancia_ve[q - 1] * 100:.1f}%)")

    plot_path = output_dir / "ve_vs_q.png"
    plot_ve_vs_q(variancia_ve, q, output_path=plot_path)
    print(f"Gráfico VE(q) salvo em: {plot_path}")

    # --- Avaliação dos detectores com PCA (q fixo) ---
    resultados = []

    print("\n=== Distância Euclidiana Mínima com PCA ===")
    res_euc = avaliar_detector_com_pca(
        X,
        Y,
        detector_class=distanciaminimacentroide,
        detector_kwargs={"robusto": False},
        nome="Distancia Euclidiana",
        q_fixo=q,
    )
    resultados.append(res_euc)
    print(pd.DataFrame([res_euc]).to_string(index=False))

    print("\n=== Distância de Mahalanobis com PCA ===")
    res_mah = avaliar_detector_com_pca(
        X,
        Y,
        detector_class=Mahalanobis,
        detector_kwargs={"alpha": 0.20, "metodo": "percentil"},
        nome="Distancia Mahalanobis",
        q_fixo=q,
    )
    resultados.append(res_mah)
    print(pd.DataFrame([res_mah]).to_string(index=False))

    # --- Tabela Final ---
    print("\n\n=== TABELA DE RESULTADOS ===")
    df_resultados = pd.DataFrame(resultados)
    cols = [
        "Modelo",
        "Acurácia",
        "Sensibilidade",
        "Especificidade",
        "Precisão",
        "F1-escore",
        "Tempo de execução treino",
        "Tempo de execução teste",
    ]
    df_resultados = df_resultados[cols]
    print(df_resultados.to_string(index=False))

    csv_path = output_dir / "tabela_resultados_classification_with_pca.csv"
    df_resultados.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")


if __name__ == "__main__":
    main()
