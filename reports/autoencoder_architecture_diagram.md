# Autoencoder Architecture Diagram

```text
CNN Feature Sequence
Shape: [batch_size, 10, 128]
            |
            v
Encoder
Linear Layer: 128 -> 96
BatchNorm1d
ReLU
Dropout
            |
            v
Linear Layer: 96 -> 64
BatchNorm1d
ReLU
            |
            v
Latent Representation
Shape: [batch_size, 10, 64]
            |
            v
Decoder
Linear Layer: 64 -> 96
BatchNorm1d
ReLU
Dropout
            |
            v
Linear Layer: 96 -> 128
            |
            v
Reconstructed CNN Feature Sequence
Shape: [batch_size, 10, 128]