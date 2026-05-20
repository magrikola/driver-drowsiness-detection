# Hyperparameter Tuning Table

| Trial | Input Dimension | Latent Dimension | Dropout Rate | Learning Rate | L2 Weight Decay | BatchNorm | Notes |
|------:|----------------|-----------------|--------------|---------------|-----------------|-----------|-------|
| 1 | 128 | 32 | 0.20 | 0.001 | 1e-4 | Yes | Stronger compression, but possible information loss |
| 2 | 128 | 64 | 0.30 | 0.001 | 1e-4 | Yes | Selected default configuration |
| 3 | 128 | 64 | 0.50 | 0.001 | 1e-4 | Yes | Stronger dropout, may reduce overfitting |
| 4 | 128 | 128 | 0.30 | 0.001 | 1e-4 | Yes | No real compression, used as comparison |
| 5 | 128 | 64 | 0.30 | 0.0005 | 1e-5 | Yes | Lower learning rate, more stable but slower training |

## Selected Configuration

The selected Autoencoder configuration is:

- Input dimension: 128
- Latent dimension: 64
- Dropout rate: 0.30
- Learning rate: 0.001
- L2 weight decay: 1e-4
- Batch Normalization: Yes

This configuration was selected because it reduces the CNN feature dimension from 128 to 64 while still keeping enough information for the GRU model.