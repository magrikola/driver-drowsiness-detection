import torch
import torch.nn as nn


class FeatureAutoencoder(nn.Module):
    """
    Autoencoder block for compressing CNN feature sequences.

    Expected input shape:
        [batch_size, sequence_length, feature_dim]
        Example: [32, 10, 128]

    Encoder output shape:
        [batch_size, sequence_length, latent_dim]
        Example: [32, 10, 64]

    Reconstructed output shape:
        [batch_size, sequence_length, feature_dim]
        Example: [32, 10, 128]
    """

    def __init__(self, input_dim=128, latent_dim=64, dropout_rate=0.3):
        super(FeatureAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(96, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(96, input_dim)
        )

    def encode(self, x):
        """
        Compress CNN feature sequence.

        Input:
            x: [B, T, 128]

        Output:
            latent: [B, T, 64]
        """

        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, T, F], but got {x.shape}")

        batch_size, sequence_length, feature_dim = x.shape

        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected feature_dim={self.input_dim}, but got {feature_dim}"
            )

        # [B, T, F] -> [B*T, F]
        x_flat = x.reshape(batch_size * sequence_length, feature_dim)

        # [B*T, F] -> [B*T, latent_dim]
        latent_flat = self.encoder(x_flat)

        # [B*T, latent_dim] -> [B, T, latent_dim]
        latent = latent_flat.reshape(batch_size, sequence_length, self.latent_dim)

        return latent

    def decode(self, latent):
        """
        Reconstruct original CNN feature sequence.

        Input:
            latent: [B, T, 64]

        Output:
            reconstructed: [B, T, 128]
        """

        if latent.dim() != 3:
            raise ValueError(f"Expected latent shape [B, T, latent_dim], but got {latent.shape}")

        batch_size, sequence_length, latent_dim = latent.shape

        if latent_dim != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, but got {latent_dim}"
            )

        # [B, T, latent_dim] -> [B*T, latent_dim]
        latent_flat = latent.reshape(batch_size * sequence_length, latent_dim)

        # [B*T, latent_dim] -> [B*T, input_dim]
        reconstructed_flat = self.decoder(latent_flat)

        # [B*T, input_dim] -> [B, T, input_dim]
        reconstructed = reconstructed_flat.reshape(
            batch_size,
            sequence_length,
            self.input_dim
        )

        return reconstructed

    def forward(self, x):
        """
        Full autoencoder forward pass.

        Input:
            x: [B, T, 128]

        Output:
            latent: [B, T, 64]
            reconstructed: [B, T, 128]
        """

        latent = self.encode(x)
        reconstructed = self.decode(latent)

        return latent, reconstructed


class EarlyStopping:
    """
    Early stopping utility for stopping training when validation loss stops improving.
    """

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True


if __name__ == "__main__":
    autoencoder = FeatureAutoencoder(
        input_dim=128,
        latent_dim=64,
        dropout_rate=0.3
    )

    dummy_input = torch.randn(32, 10, 128)

    latent, reconstructed = autoencoder(dummy_input)

    print("Autoencoder test successful.")
    print(f"Input shape:         {dummy_input.shape}")
    print(f"Latent shape:        {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Total parameters:    {sum(p.numel() for p in autoencoder.parameters()):,}")