from sklearn.decomposition import PCA
from classification import mahala, euclidiana
import numpy as np


def apply_pca(X):
    pca = PCA(n_components=None)
    pca.fit(X.T)

    # % [V L VEi]=pcacov(cov(X'));
    V = pca.components_.T  # Autovetores (componentes principais)
    VEi = pca.explained_variance_ratio_  # Variância explicada por cada componente
    VEq = np.cumsum(VEi)
    variance = 0.98
    q = np.searchsorted(VEq, variance) + 1
    print(f"O valor de q encontrado foi: {q}")
    Vq = V[:, :q]
    Qq = Vq.T
    X_pca = Qq @ X  # Projeta os dados originais nos novos componentes
    return X_pca, pca


def atividade_6(X, y):
    X_pca, pca = apply_pca(X)
    print(f"Shape of X after PCA: {X_pca.shape}")

    # Classificação usando distância de Mahalanobis
    print("Classificação usando distância de Mahalanobis:")
    y_pred_mahala = mahala(X_pca, y)
    print(f"Predicted labels (Mahalanobis): {y_pred_mahala}")

    # Classificação usando distância Euclidiana
    print("Classificação usando distância Euclidiana:")
    y_pred_euclidiana = euclidiana(X_pca, y)
    print(f"Predicted labels (Euclidiana): {y_pred_euclidiana}")
