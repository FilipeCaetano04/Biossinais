from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import re


LEADS = ["I", "II", "III", "V3", "V6"]
FEATURE_COLS = [f"mean_{lead}" for lead in LEADS if lead not in ["II", "III", "AVR"]]
FEATURE_MEAN = [f'median_{lead}' for lead in LEADS if lead not in ["II", "III", "AVR"]]
FEATURE_STD = [f'std_{lead}' for lead in LEADS if lead not in ["II", "III", "AVR"]]

def ICA(df: pd.DataFrame, feature_cols, n_components=3):
    """
    Recebe um DF já filtrado, sem NaNs e sem Outliers.
    Apenas escala e transforma.
    """
    # 1. Isola as features e a label
    X = df[feature_cols]
    labels = df['label'] 

    # 2. Normalização 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Execução do ICA
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    X_ica = ica.fit_transform(X_scaled)

    col_names = [f'IC{i+1}' for i in range(n_components)]
    df_res = pd.DataFrame(X_ica, columns=col_names)
    
    df_res['Label'] = labels.values 
    
    return df_res

def remove_outliers(df: pd.DataFrame, label_col='label', threshold=0.001):
    df_final = []
    cols_numericas = df.select_dtypes(include=[np.number]).columns
    cols_numericas = [c for c in cols_numericas if c not in ['segment_id', 'ecg_id']]

    for classe in df[label_col].unique():
        df_classe = df[df[label_col] == classe].copy()
        
        # 1. Calcula limites de todas as colunas de uma vez
        limite_inf = df_classe[cols_numericas].quantile(threshold)
        limite_sup = df_classe[cols_numericas].quantile(1 - threshold)
        
        # 2. Cria uma máscara booleana global (True se estiver dentro do limite)
        mask = (df_classe[cols_numericas] >= limite_inf) & (df_classe[cols_numericas] <= limite_sup)
        
        # 3. Mantem apenas as linhas onde todas as colunas numéricas deram True
        mask_todas_colunas_ok = mask.all(axis=1)
        df_classe = df_classe[mask_todas_colunas_ok]
        
        print(f"  - Classe {classe}: outlier filtering concluido.")
        df_final.append(df_classe)

    return pd.concat(df_final).reset_index(drop=True)

def _to_filename(text: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return sanitized or "ica_plot"


def plotar_ica_estatico(
    df_ica,
    title: str = "",
    output_dir: str | Path | None = None,
    file_prefix: str | None = None,
    show_plot: bool = True,
):
    df_ica = df_ica.copy()
    df_ica['Label'] = df_ica['Label'].astype(str)
    df_ica = df_ica[~df_ica['Label'].isin(['nan', 'NaN', 'None', 'n/a', ''])]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    df_ica_labels = df_ica.columns
    print(df_ica_labels)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for label in df_ica['Label'].unique():
        sub_df = df_ica[df_ica['Label'] == label]
        ax.scatter(sub_df[df_ica_labels[0]], sub_df[df_ica_labels[1]], sub_df[df_ica_labels[2]], 
                   label=label, s=30, alpha=0.6)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('IC1')
    ax.set_ylabel('IC2')
    ax.set_zlabel('IC3')
    ax.legend(title="Diagnóstico", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or _to_filename(title)
        fig.savefig(output_path / f"{prefix}_3d.png", dpi=220, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)

    g = sns.pairplot(df_ica, hue='Label', palette='viridis', 
                     diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
    g.figure.suptitle(f"{title} - pairplot", fontsize=12)
    g.figure.tight_layout()

    if output_path is not None:
        prefix = file_prefix or _to_filename(title)
        g.figure.savefig(
            output_path / f"{prefix}_pairplot.png", dpi=220, bbox_inches="tight"
        )

    if show_plot:
        plt.show()
    plt.close(g.figure)

def PCA_SIMPLE(df_elite, n_components=3, return_metadata=False):
    """
    Executa o PCA nas features selecionadas
    """
    # Ignora IDs e a coluna de texto 'label'
    cols_ignorar = ['ecg_id', 'segment_id', 'label']
    features = [c for c in df_elite.columns if c not in cols_ignorar]
    
    X = df_elite[features]
    labels = df_elite['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    max_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_exp = pca.explained_variance_ratio_ * 100
    print(f"PCA concluído!")
    for i, v in enumerate(var_exp):
        print(f"   - PC{i+1}: {v:.2f}% da variância")
    print(f"   - Total acumulado: {np.sum(var_exp):.2f}%")

    col_names = [f'PC{i+1}' for i in range(max_components)]
    df_pca = pd.DataFrame(X_pca, columns=col_names)
    df_pca['Label'] = labels.values # Mantendo o alinhamento

    if return_metadata:
        return df_pca, pca, scaler, features

    return df_pca
