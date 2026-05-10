import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dimensionality_reduction.autoencoder import AutoEncoder1D


def project_root():
    return Path(__file__).resolve().parents[2]


def resolve_paths(input_path=None, output_dir=None):
    root = project_root()
    if input_path is None:
        input_path = (
            root
            / "data"
            / "statistical_analysis_outputs"
            / "descriptive_statistics_segmented.csv"
        )
    if output_dir is None:
        output_dir = root / "data" / "statistical_analysis_outputs" / "autoencoder"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(input_path), output_dir


def load_and_validate_data(input_path):
    df = pd.read_csv(input_path)

    if "label" not in df.columns:
        raise ValueError("O CSV precisa ter uma coluna chamada 'label'.")

    df = df.dropna(subset=["label"]).reset_index(drop=True)

    metadata_cols = ["ecg_id", "segment_id", "label"]
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    metadata = df[metadata_cols].copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ["ecg_id", "segment_id"]]

    if len(feature_cols) == 0:
        raise ValueError(
            "Nenhuma coluna numerica encontrada para treinar o autoencoder."
        )

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print("Features usadas no AutoEncoder:")
    print(feature_cols)

    return X, metadata, feature_cols


def preprocess_features(X):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    return X_scaled, X_tensor, imputer, scaler


def create_dataloaders(X_tensor, batch_size=32, train_ratio=0.8, seed=42):
    dataset = TensorDataset(X_tensor, X_tensor)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(input_dim, latent_dim, device, lr=1e-3):
    model = AutoEncoder1D(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def train_autoencoder(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=200,
):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            x_hat = model(x_batch)
            loss = criterion(x_hat, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                x_hat = model(x_batch)
                loss = criterion(x_hat, y_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f}"
            )


def build_latent_dataframe(model, X_tensor, metadata, latent_dim, device):
    model.eval()
    with torch.no_grad():
        latent = model.encode(X_tensor.to(device)).cpu().numpy()

    latent_cols = [f"Z{i + 1}" for i in range(latent_dim)]
    ae_df = pd.DataFrame(latent, columns=latent_cols)
    ae_df = pd.concat([ae_df, metadata.reset_index(drop=True)], axis=1)
    return ae_df, latent


def save_latent_scores(ae_df, output_dir):
    output_path = Path(output_dir) / "autoencoder_latent_scores.csv"
    ae_df.to_csv(output_path, index=False)
    return output_path


def run_pca_on_latent(latent, metadata, n_components=5):
    max_components = min(n_components, latent.shape[0], latent.shape[1])
    pca_model = PCA(n_components=max_components)
    pca_scores = pca_model.fit_transform(latent)

    pca_cols = [f"PC{i + 1}" for i in range(max_components)]
    pca_df = pd.DataFrame(pca_scores, columns=pca_cols)
    pca_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)

    return pca_df, pca_model


def save_latent_pca_scores(pca_df, output_dir):
    output_path = Path(output_dir) / "autoencoder_latent_pca_scores.csv"
    pca_df.to_csv(output_path, index=False)
    return output_path


def plot_latent_space_2d(ae_df, output_dir):
    required_cols = {"Z1", "Z2", "label"}
    if not required_cols.issubset(ae_df.columns):
        raise ValueError("O DataFrame precisa ter Z1, Z2 e label para plot 2D.")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=ae_df,
        x="Z1",
        y="Z2",
        hue="label",
        alpha=0.75,
        s=45,
        edgecolor="none",
    )

    plt.title("AutoEncoder - espaco latente 2D")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    output_path = Path(output_dir) / "autoencoder_z1_z2.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()

    return output_path


def plot_latent_space_3d(ae_df, output_dir):
    required_cols = {"Z1", "Z2", "Z3", "label"}
    if not required_cols.issubset(ae_df.columns):
        raise ValueError("O DataFrame precisa ter Z1, Z2, Z3 e label para plot 3D.")

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    labels = ae_df["label"].unique()
    palette = sns.color_palette("tab10", n_colors=len(labels))
    color_map = dict(zip(labels, palette))

    for label in labels:
        subset = ae_df[ae_df["label"] == label]
        ax.scatter(
            subset["Z1"],
            subset["Z2"],
            subset["Z3"],
            label=label,
            s=45,
            alpha=0.75,
            color=color_map[label],
        )

    ax.set_title("AutoEncoder - espaco latente 3D")
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    ax.set_zlabel("Z3")
    ax.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    output_path = Path(output_dir) / "autoencoder_z1_z2_z3.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()

    return output_path


def plot_latent_pca_2d(pca_df, output_dir):
    required_cols = {"PC1", "PC2", "label"}
    if not required_cols.issubset(pca_df.columns):
        raise ValueError("O DataFrame precisa ter PC1, PC2 e label para plot 2D.")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="label",
        alpha=0.75,
        s=45,
        edgecolor="none",
    )

    plt.title("AutoEncoder + PCA - espaco latente 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    output_path = Path(output_dir) / "autoencoder_latent_pca_2d.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()

    return output_path


def plot_latent_pca_3d(pca_df, output_dir):
    required_cols = {"PC1", "PC2", "PC3", "label"}
    if not required_cols.issubset(pca_df.columns):
        raise ValueError("O DataFrame precisa ter PC1, PC2, PC3 e label para plot 3D.")

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    labels = pca_df["label"].unique()
    palette = sns.color_palette("tab10", n_colors=len(labels))
    color_map = dict(zip(labels, palette))

    for label in labels:
        subset = pca_df[pca_df["label"] == label]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            subset["PC3"],
            label=label,
            s=45,
            alpha=0.75,
            color=color_map[label],
        )

    ax.set_title("AutoEncoder + PCA - espaco latente 3D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    output_path = Path(output_dir) / "autoencoder_latent_pca_3d.png"
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()

    return output_path


if __name__ == "__main__":
    input_path, output_dir = resolve_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, metadata, _ = load_and_validate_data(input_path)
    X_scaled, X_tensor, _, _ = preprocess_features(X)
    train_loader, val_loader = create_dataloaders(X_tensor, batch_size=32)

    latent_dim = 15
    model, criterion, optimizer = build_model(
        input_dim=X_scaled.shape[1], latent_dim=latent_dim, device=device
    )

    train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=200,
    )

    ae_df, latent = build_latent_dataframe(
        model=model,
        X_tensor=X_tensor,
        metadata=metadata,
        latent_dim=latent_dim,
        device=device,
    )

    save_latent_scores(ae_df, output_dir)

    pca_df, pca_model = run_pca_on_latent(
        latent=latent, metadata=metadata, n_components=3
    )
    save_latent_pca_scores(pca_df, output_dir)

    print("\nShape original:", X_scaled.shape)
    print("Shape latente:", latent.shape)
    print("Variancia explicada PCA (latente):", pca_model.explained_variance_ratio_)
    print(ae_df.head())

    plot_latent_space_2d(ae_df, output_dir)
    plot_latent_space_3d(ae_df, output_dir)
    plot_latent_pca_2d(pca_df, output_dir)
    plot_latent_pca_3d(pca_df, output_dir)
