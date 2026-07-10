import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import time


# distancias euclideanas
class distanciaminimacentroide:
    def __init__(self, robusto, alpha=0.20):
        self.robusto = robusto
        self.limiar = None
        self.alpha = alpha

    def fit(self, x_train):
        x_train = np.asarray(x_train)
        if self.robusto:
            self.centroide = np.median(x_train, axis=0)
        else:
            self.centroide = np.mean(x_train, axis=0)
        diff = x_train - self.centroide
        d2_train = np.sum(diff**2, axis=1)
        self.limiar = np.percentile(d2_train, 100 * (1 - self.alpha))
        return self

    def predict(self, x_test):
        x_test = np.asarray(x_test)
        diff = x_test - self.centroide

        # Distância euclidiana ao quadrado
        d2 = np.sum(diff**2, axis=1)
        return np.where(d2 > self.limiar, 1, -1)


class knn:
    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        return self

    def predict(self, x_test, batch_size=512):
        x_test = np.asarray(x_test)
        predicoes = []
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i : i + batch_size]
            distancias = np.sqrt(
                np.sum((batch[:, None, :] - self.x_train[None, :, :]) ** 2, axis=2)
            )
            indices_vizinhos = np.argsort(distancias, axis=1)[:, : self.k]

            classes_vizinhos = self.y_train[indices_vizinhos]

            for vizinhos in classes_vizinhos:
                valores, contagens = np.unique(vizinhos, return_counts=True)
                predicao = valores[np.argmax(contagens)]
                predicoes.append(predicao)
        return np.array(predicoes)


class Mahalanobis:
    def __init__(self, alpha=0.20, metodo="chi2"):
        self.alpha = alpha
        self.metodo = metodo
        self.centroids = None
        self.covariance_matrix = None
        self.inv_covariance_matrix = None
        self.classes = None

    def fit(self, x_train):
        N, p = x_train.shape
        self.media = np.mean(x_train, axis=0)
        self.covariance_matrix = np.cov(x_train, rowvar=False)
        self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        distancia = self.mahalanobis_distance(x_train)
        if self.metodo == "chi2":
            # Metodo 1 - valor critico da qui-quadrado
            self.k = chi2.ppf(1 - self.alpha, df=p)
        else:
            # Metodo 2 - percentil empirico das distancias observadas
            self.k = np.percentile(distancia, 100 * (1 - self.alpha))

        return self

    def mahalanobis_distance(self, x):
        x = np.asarray(x)
        diff = x - self.media
        return np.sum((diff @ self.inv_covariance_matrix) * diff, axis=1)

    def predict(self, x_test):
        x_test = np.asarray(x_test)
        distancias = self.mahalanobis_distance(x_test)
        return np.where(distancias > self.k, 1, -1)


class PCADetector:
    def __init__(self, n_components, alpha=0.20, metodo="chi2"):
        self.n_components = n_components
        self.Q = None
        self.alpha = alpha
        self.metodo = metodo
        self.limiar = None
        self.mean = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N, p = X.shape
        covariance_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        variancia_explicada = (eigenvalues / np.sum(eigenvalues)) * 100
        variancia_acumulada = np.cumsum(variancia_explicada) / 100
        V_q = eigenvectors[:, : self.n_components]
        self.Q = V_q.T
        # Projeção e reconstrução do Treino
        Z_trn = X @ self.Q.T
        X_rec_trn = Z_trn @ self.Q

        # Erro de reconstrução quadrático do treino
        E_trn = X - X_rec_trn
        e2_trn = np.sum(E_trn**2, axis=1)
        # Determinar o limiar de decisão com base no método escolhido
        if self.metodo == "chi2":
            # Metodo 1 - valor critico da qui-quadrado
            self.limiar = chi2.ppf(1 - self.alpha, df=p - self.n_components)
        else:
            self.limiar = np.percentile(e2_trn, 100 * (1 - self.alpha))
        return self

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=float)
        Z = X @ self.Q.T
        X_rec = Z @ self.Q
        E = X - X_rec
        return np.sum(E**2, axis=1)

    def predict(self, X):
        e2 = self.reconstruction_error(X)
        return np.where(e2 > self.limiar, 1, -1)


def gerar_linha_resultado(nome_modelo, PERF):
    medias = np.mean(PERF, axis=0)
    desvios = np.std(PERF, axis=0, ddof=1)

    return {
        "Modelo": nome_modelo,
        "Acurácia": f"{medias[0]:.1f}% ± {desvios[0]:.1f}",
        "Sensibilidade": f"{medias[1]:.1f}% ± {desvios[1]:.1f}",
        "Especificidade": f"{medias[2]:.1f}% ± {desvios[2]:.1f}",
        "Precisão": f"{medias[3]:.1f}% ± {desvios[3]:.1f}",
        "F1-escore": f"{medias[4]:.1f}% ± {desvios[4]:.1f}",
        "Tempo de execução treino": f"{medias[5]:.1f}ms ± {desvios[5]:.1f}",
        "Tempo de execução teste": f"{medias[6]:.1f}ms ± {desvios[6]:.1f}",
    }


def pca(X, Y):
    Y = np.asarray(Y).ravel()
    n_features, n_samples = X.shape
    p_trn = 0.8  # Porcentagem de dados para treino
    epochs = 100
    perf_list = []
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        # Embaralha as colunas (amostras) mantendo a correspondência com Y
        idx_perm = rng.permutation(n_samples)
        X_shuffled = X[:, idx_perm]
        Y_shuffled = Y[idx_perm]

        # Encontra os casos positivos (anômalos)
        idx_pos = np.where(Y_shuffled > 0)[0]
        X_pos = X_shuffled[:, idx_pos]
        Y_pos = Y_shuffled[idx_pos]

        # Encontra os casos negativos (normais)
        idx_neg = np.where(Y_shuffled < 0)[0]
        X_neg = X_shuffled[:, idx_neg]
        Y_neg = Y_shuffled[idx_neg]

        # Divisão dos dados negativos em treino e teste
        n_neg = len(idx_neg)
        n_neg_trn = int(np.floor(p_trn * n_neg))

        X_neg_trn = X_neg[:, :n_neg_trn]
        X_neg_tst = X_neg[:, n_neg_trn:]
        Y_neg_tst = Y_neg[n_neg_trn:]

        # Concatena os dados de teste (Positivos + Negativos de teste)
        X_tst = np.hstack((X_pos, X_neg_tst))
        Y_tst = np.concatenate((Y_pos, Y_neg_tst))
        Y_tst = np.asarray(Y_tst).ravel().astype(int)
        n_tst = X_tst.shape[1]

        # Normalização dos conjuntos (baseado no treino negativo)
        # ddof=1 garante o desvio padrão amostral (igual ao std(..., 0, 2) do Octave)
        me = np.mean(X_neg_trn, axis=1, keepdims=True)
        se = np.std(X_neg_trn, axis=1, ddof=1, keepdims=True)

        # Evita divisão por zero caso algum desvio padrão seja nulo
        se[se == 0] = 1e-9

        X_neg_trn_norm = (X_neg_trn - me) / se
        X_tst_norm = (X_tst - me) / se

        pca_model = PCADetector(n_components=5, alpha=0.20, metodo="percentil")
        inicio_treino = time.perf_counter()
        pca_model.fit(X_neg_trn_norm.T)
        fim_treino = time.perf_counter()

        inicio_teste = time.perf_counter()
        Y_pred = pca_model.predict(X_tst_norm.T)
        fim_teste = time.perf_counter()

        tempo_treino = fim_treino - inicio_treino
        tempo_teste = fim_teste - inicio_teste
        vn, fp, fn, vp = confusion_matrix(Y_tst, Y_pred, labels=[-1, 1]).ravel()

        acc1 = 100 * (vp + vn) / len(Y_tst)
        sensibilidade = 100 * vp / (vp + fn) if (vp + fn) > 0 else 0
        especificidade = 100 * vn / (vn + fp) if (vn + fp) > 0 else 0
        precisao = 100 * vp / (vp + fp) if (vp + fp) > 0 else 0

        f1 = (
            2 * precisao * sensibilidade / (precisao + sensibilidade)
            if (precisao + sensibilidade) > 0
            else 0
        )
        mg = np.sqrt(sensibilidade * especificidade)

        perf_list.append(
            [
                acc1,
                sensibilidade,
                especificidade,
                precisao,
                f1,
                mg,
                tempo_treino * 1000,
                tempo_teste * 1000,
            ]
        )
    # Transforma a lista de resultados em um array Numpy
    PERF = np.array(perf_list)

    return gerar_linha_resultado("PCA (Autoencoder Linear)", PERF)


def euclidiana(X, Y, usar_pca=False):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).ravel()
    n_features, n_samples = X.shape
    p_trn = 0.8  # Porcentagem de dados para treino
    epochs = 100
    perf_list = []
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        # Embaralha as amostras
        idx_perm = rng.permutation(n_samples)

        X_shuffled = X[:, idx_perm]
        Y_shuffled = Y[idx_perm]

        # Casos negativos/normais
        idx_neg = np.where(Y_shuffled < 0)[0]
        Xneg = X_shuffled[:, idx_neg]
        Yneg = Y_shuffled[idx_neg]

        # Casos positivos/anômalos
        idx_pos = np.where(Y_shuffled > 0)[0]
        Xpos = X_shuffled[:, idx_pos]
        Ypos = Y_shuffled[idx_pos]

        # Separação dos negativos em treino e teste
        Nneg = len(idx_neg)
        Nneg_trn = int(np.floor(p_trn * Nneg))

        Xneg_trn = Xneg[:, :Nneg_trn]
        Xneg_tst = Xneg[:, Nneg_trn:]
        Yneg_tst = Yneg[Nneg_trn:]

        # Normalização baseada apenas nos negativos de treino
        me = np.mean(Xneg_trn, axis=1, keepdims=True)
        se = np.std(Xneg_trn, axis=1, ddof=1, keepdims=True)

        se[se == 0] = 1e-9

        Xneg_trn_norm = (Xneg_trn - me) / se

        # Teste = positivos + negativos separados para teste
        Xtst = np.hstack((Xpos, Xneg_tst))
        Ytst = np.concatenate((Ypos, Yneg_tst))

        Xtst_norm = (
            Xtst - me
        ) / se  # Normaliza dados de teste com estatísticas dos dados de treino

        # PCA opcional: fit no treino, transforma ambos
        if usar_pca:
            if epoch == 0:
                pca_temp = PCA(n_components=None)
                pca_temp.fit(Xneg_trn_norm.T)
                var_acum = np.cumsum(pca_temp.explained_variance_ratio_)
                q = np.searchsorted(var_acum, 0.98) + 1
                print(f"  q fixado em {q} (98% variancia, 1a epoca)")
            pca = PCA(n_components=q)
            X_trn = pca.fit_transform(Xneg_trn_norm.T)
            X_tst = pca.transform(Xtst_norm.T)
        else:
            X_trn = Xneg_trn_norm.T
            X_tst = Xtst_norm.T

        euclidiana_model = distanciaminimacentroide(robusto=False)

        inicio_treino = time.perf_counter()
        euclidiana_model.fit(X_trn)
        fim_treino = time.perf_counter()

        inicio_teste = time.perf_counter()
        Ypred_all = euclidiana_model.predict(X_tst)
        fim_teste = time.perf_counter()

        tempo_treino = fim_treino - inicio_treino
        tempo_teste = fim_teste - inicio_teste
        prod_rotulos = Ytst * Ypred_all  # Produto Schur dos rótulos reais e preditos
        Num_acertos_total = np.sum(
            prod_rotulos > 0
        )  # Acerto ocorre qdo produto_rotulos > 0
        ACC1 = 100 * Num_acertos_total / len(Ytst)

        Num_VP = np.sum(
            (Ytst > 0) & (Ypred_all > 0)
        )  # No. exemplos positivos preditos corretamente no teste
        VP_rate = 100 * Num_VP / np.sum(Ytst > 0)

        Num_VN = np.sum(
            (Ytst < 0) & (Ypred_all < 0)
        )  # No. exemplos negativos preditos corretamente no teste
        VN_rate = 100 * Num_VN / np.sum(Ytst < 0)

        Num_FP = np.sum(
            (Ytst < 0) & (Ypred_all > 0)
        )  # No. exemplos negativos preditos como positivos
        FP_rate = 100 * Num_FP / np.sum(Ytst < 0)

        Num_FN = np.sum(
            (Ytst > 0) & (Ypred_all < 0)
        )  # No. exemplos negativos preditos corretamente no teste
        FN_rate = 100 * Num_FN / np.sum(Ytst > 0)

        ACC2 = 100 * (Num_VP + Num_VN) / (Num_VP + Num_VN + Num_FP + Num_FN)

        Sensibilidade = 100 * Num_VP / (Num_VP + Num_FN)

        Especificidade = 100 * Num_VN / (Num_VN + Num_FP)

        MG = np.sqrt(Sensibilidade * Especificidade)

        Precisao = 100 * Num_VP / (Num_VP + Num_FP)

        F1 = 2 * Precisao * Sensibilidade / (Precisao + Sensibilidade)

        perf_list.append(
            [
                ACC1,
                Sensibilidade,
                Especificidade,
                Precisao,
                F1,
                MG,
                tempo_treino * 1000,
                tempo_teste * 1000,
            ]
        )
    # Transforma a lista de resultados em um array Numpy
    PERF = np.array(perf_list)
    return gerar_linha_resultado(
        "Distancia Euclidiana",
        PERF,
    )


def mahala(X, Y, usar_pca=False):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).ravel()
    n_features, n_samples = X.shape
    Ptrn = 0.8  # Porcentagem de dados para treino
    epochs = 100
    perf_list = []
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        I = rng.permutation(n_samples)
        X = X[:, I]
        Y = Y[I]

        # Encontra os casos negativos (não-convulsivos)
        Ineg = np.where(Y < 0)[0]
        Nneg = len(Ineg)

        Xneg = X[:, Ineg]
        Yneg = Y[Ineg]

        Nneg_trn = int(np.floor(Ptrn * Nneg))
        Xneg_trn = Xneg[:, :Nneg_trn]
        Xneg_tst = Xneg[:, Nneg_trn:]
        Yneg_tst = Yneg[Nneg_trn:]
        Nneg_tst = Nneg - Nneg_trn

        me = np.mean(
            Xneg_trn, axis=1, keepdims=True
        )  # vetor de atributos medio (exemplos de treino)
        se = np.std(
            Xneg_trn, axis=1, ddof=1, keepdims=True
        )  # Desvio padrao dos atributos (exemplos de treino)
        se[se == 0] = 1e-9
        Xneg_trn = (
            Xneg_trn - me
        ) / se  # Normaliza por Z-score os dados de treino (exemplos negativos)

        # Encontra os casos positivos (anômalos ou convulsivos)
        Ipos = np.where(Y > 0)[0]
        Npos = len(Ipos)

        Xpos = X[:, Ipos]
        Ypos = Y[Ipos]

        Xtst = np.concatenate((Xpos, Xneg_tst), axis=1)
        Ytst = np.concatenate((Ypos, Yneg_tst))
        Ntst = Xtst.shape[1]

        Xtst = (
            Xtst - me
        ) / se  # Normaliza dados de teste com estatísticas dos dados de treino

        # PCA opcional: fit no treino, transforma ambos
        if usar_pca:
            if epoch == 0:
                pca_temp = PCA(n_components=None)
                pca_temp.fit(Xneg_trn.T)
                var_acum = np.cumsum(pca_temp.explained_variance_ratio_)
                q = np.searchsorted(var_acum, 0.98) + 1
                print(f"  q fixado em {q} (98% variancia, 1a epoca)")
            pca = PCA(n_components=q)
            X_trn = pca.fit_transform(Xneg_trn.T)
            X_tst = pca.transform(Xtst.T)
        else:
            X_trn = Xneg_trn.T
            X_tst = Xtst.T

        model = Mahalanobis(alpha=0.20, metodo="percentil")

        inicio_treino = time.perf_counter()
        model.fit(X_trn)
        fim_treino = time.perf_counter()

        inicio_teste = time.perf_counter()
        Ypred_all = model.predict(X_tst)
        fim_teste = time.perf_counter()

        tempo_treino = fim_treino - inicio_treino
        tempo_teste = fim_teste - inicio_teste
        prod_rotulos = Ytst * Ypred_all  # Produto Schur dos rótulos reais e preditos
        Num_acertos_total = np.sum(
            prod_rotulos > 0
        )  # Acerto ocorre qdo produto_rotulos > 0
        ACC1 = 100 * Num_acertos_total / len(Ytst)

        Num_VP = np.sum(
            (Ytst > 0) & (Ypred_all > 0)
        )  # No. exemplos positivos preditos corretamente no teste
        VP_rate = 100 * Num_VP / np.sum(Ytst > 0)

        Num_VN = np.sum(
            (Ytst < 0) & (Ypred_all < 0)
        )  # No. exemplos negativos preditos corretamente no teste
        VN_rate = 100 * Num_VN / np.sum(Ytst < 0)

        Num_FP = np.sum(
            (Ytst < 0) & (Ypred_all > 0)
        )  # No. exemplos negativos preditos como positivos
        FP_rate = 100 * Num_FP / np.sum(Ytst < 0)

        Num_FN = np.sum(
            (Ytst > 0) & (Ypred_all < 0)
        )  # No. exemplos negativos preditos corretamente no teste
        FN_rate = 100 * Num_FN / np.sum(Ytst > 0)

        ACC2 = 100 * (Num_VP + Num_VN) / (Num_VP + Num_VN + Num_FP + Num_FN)

        Sensibilidade = 100 * Num_VP / (Num_VP + Num_FN)

        Especificidade = 100 * Num_VN / (Num_VN + Num_FP)

        MG = np.sqrt(Sensibilidade * Especificidade)

        Precisao = 100 * Num_VP / (Num_VP + Num_FP)

        F1 = 2 * Precisao * Sensibilidade / (Precisao + Sensibilidade)

        perf_list.append(
            [
                ACC1,
                Sensibilidade,
                Especificidade,
                Precisao,
                F1,
                MG,
                tempo_treino * 1000,
                tempo_teste * 1000,
            ]
        )
    PERF = np.array(perf_list)

    return gerar_linha_resultado("Distancia Mahalanobis", PERF)


def atividade2(df1, df2):
    # ATIVIDADE 2: calculo do posto das matrizes
    X_norm = df1.drop(columns=["Unnamed: 0", "ecg_id", "segment_id", "label"]).values
    X_anorm = df2.drop(columns=["Unnamed: 0", "ecg_id", "segment_id", "label"]).values
    rank_norm = np.linalg.matrix_rank(X_norm)
    rank_anorm = np.linalg.matrix_rank(X_anorm)
    OUTPUT_DIR = "./"
    with open(f"{OUTPUT_DIR}/atividade2_ranks.txt", "w") as f:
        f.write(f"Posto da Matriz Normal (X_norm): {rank_norm}\n")
        f.write(f"Posto da Matriz Anormal (X_anorm): {rank_anorm}\n")

    print(f"[Atividade 2] Posto Normal: {rank_norm} | Posto Anormal: {rank_anorm}")


if __name__ == "__main__":
    df1 = pd.read_csv("fft_extracted_features_NORM.csv")
    df2 = pd.read_csv("fft_extracted_features_MI.csv")
    atividade2(df1, df2)
    resultado = pd.concat([df1, df2], ignore_index=True)
    resultado = resultado.drop(columns=["Unnamed: 0", "ecg_id", "segment_id"])
    resultado["label"] = resultado["label"].replace({"NORM": -1, "MI": 1})
    Y = resultado["label"]
    X = resultado.drop(columns=["label"]).values.T
    Y = np.asarray(Y)
    X = np.asarray(X, dtype=float)
    linhas_resultado = []
    print("Executando PCA...")
    linhas_resultado.append(pca(X, Y))

    print("Executando Distância Euclidiana...")
    linhas_resultado.append(euclidiana(X, Y))

    print("Executando Distância de Mahalanobis...")
    linhas_resultado.append(mahala(X, Y))

    df_resultados = pd.DataFrame(linhas_resultado)

    print(df_resultados)

    df_resultados.to_csv("tabela_resultados_detectores_TC3.csv", index=False)
