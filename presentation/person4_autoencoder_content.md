
# Person 4 — Autoencoder and Regularization

## My Role

My part of the project is the Autoencoder and regularization block. The Autoencoder is added after the CNN feature extractor and before the GRU temporal model.

## Why We Use an Autoencoder

The CNN extracts visual features from each video frame. Each frame is represented as a 128-dimensional feature vector. Since each video sample has 10 frames, the Autoencoder input shape is [batch_size, 10, 128].

The purpose of the Autoencoder is to compress these CNN features into a smaller latent representation. In our design, the encoder reduces the feature size from 128 to 64.

## Autoencoder Flow

CNN feature sequence:

[batch_size, 10, 128]

Encoder output:

[batch_size, 10, 64]

Decoder output:

[batch_size, 10, 128]

The decoder tries to reconstruct the original CNN feature vector from the compressed latent representation.

## Loss Function

The Autoencoder uses Mean Squared Error loss. This loss compares the original CNN features with the reconstructed CNN features. If the reconstruction loss is low, it means that the compressed latent representation still keeps important information.

## Integration with GRU

The latent output of the encoder can be passed to the GRU model. If the Autoencoder is used before the GRU, the GRU input dimension should be changed from 128 to 64.

Original CNN + GRU:

CNN -> GRU(input_dim=128) -> Classifier

With Autoencoder:

CNN -> Autoencoder Encoder -> GRU(input_dim=64) -> Classifier

## Regularization

I added several regularization techniques:

- Dropout
- Batch Normalization
- Early Stopping
- L2 Regularization
- Latent feature compression

These methods help reduce overfitting and improve model stability.