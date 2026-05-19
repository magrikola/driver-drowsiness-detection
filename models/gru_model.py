import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU-based temporal model for drowsiness detection.

    Takes CNN feature sequences as input and learns temporal patterns.
    Input shape:  (batch_size, sequence_length, cnn_feature_dim) -> (32, 10, 128)
    Output shape: (batch_size, num_classes) -> (32, 2)
    """

    def __init__(self, input_dim=128, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        gru_out, hidden = self.gru(x)
        # gru_out: (batch, seq_len, hidden_dim)

        # Use the last time step output
        last_output = gru_out[:, -1, :]

        # Regularization
        out = self.dropout(last_output)
        out = self.batch_norm(out)

        # Classification
        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = GRUModel()
    dummy_input = torch.randn(32, 10, 128)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")