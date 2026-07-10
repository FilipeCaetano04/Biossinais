import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

from src.classification.detectores import Mahalanobis

df_norm = pd.read_csv(
    "data_classification/fft_extracted_features_NORM.csv", index_col=0
)
df_mi = pd.read_csv("data_classification/fft_extracted_features_MI.csv", index_col=0)
df = pd.concat([df_norm, df_mi], ignore_index=True)
df = df.drop(columns=["ecg_id", "segment_id"], errors="ignore")
df["label"] = df["label"].replace({"NORM": -1, "MI": 1})
Y = df["label"].values
X = df.drop(columns=["label"]).values

rng = np.random.default_rng(42)
idxs = rng.permutation(len(Y))
neg = idxs[Y[idxs] == -1]
pos = idxs[Y[idxs] == 1]
n_trn = int(0.8 * len(neg))

trn_idx, tst_idx = neg[:n_trn], np.concatenate([neg[n_trn:], pos])
rng.shuffle(tst_idx)

X_trn, Y_trn = X[trn_idx], Y[trn_idx]
X_tst, Y_tst = X[tst_idx], Y[tst_idx]

mu, sd = X_trn.mean(axis=0), X_trn.std(axis=0, ddof=1)
sd[sd == 0] = 1e-9
X_trn = (X_trn - mu) / sd
X_tst = (X_tst - mu) / sd

model = Mahalanobis()
model.fit(X_trn)
distancias = model.mahalanobis_distance(X_tst)

fpr, tpr, _ = roc_curve(Y_tst, distancias, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b-", linewidth=2, label=f"Mahalanobis (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Mahalanobis")
plt.legend()
plt.grid(True, alpha=0.3)
Path("results").mkdir(exist_ok=True)
plt.savefig("results/roc_mahalanobis.png", dpi=150)
plt.show()
