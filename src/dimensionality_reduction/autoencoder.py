import torch.nn as nn


class AutoEncoder1D(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class AutoEncoderConv1D(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 5000 -> 2500
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 2500 -> 1250
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 1250 -> 625
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(5),  # 625 -> 125
        )

        self.decoder_conv = nn.Sequential(
            nn.Upsample(
                scale_factor=5, mode="linear", align_corners=False
            ),  # 125 -> 625
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="linear", align_corners=False
            ),  # 625 -> 1250
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="linear", align_corners=False
            ),  # 1250 -> 2500
            nn.Conv1d(64, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(
                scale_factor=2, mode="linear", align_corners=False
            ),  # 2500 -> 5000
            nn.Conv1d(32, 12, kernel_size=15, padding=7),
        )

        self.encoder_fc = nn.Sequential(nn.Linear(256 * 125, latent_dim))
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 256 * 125), nn.ReLU())

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.flatten(start_dim=1)
        z = self.encoder_fc(x)
        return z

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 256, 125)
        x = self.decoder_conv(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
