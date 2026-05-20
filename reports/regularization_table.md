# Regularization Table

| Regularization Method | Where It Is Used | Purpose |
|----------------------|------------------|---------|
| Dropout | Encoder and decoder hidden layers | Reduces overfitting by randomly deactivating neurons during training |
| Batch Normalization | After linear layers | Stabilizes training and improves learning speed |
| Early Stopping | During training | Stops training when validation loss does not improve |
| L2 Regularization | Adam optimizer with weight_decay | Penalizes large weights and improves generalization |
| Latent Feature Compression | Encoder output | Forces the model to learn compact and meaningful representations |

## Explanation

The Autoencoder uses multiple regularization techniques to improve model stability. Dropout helps prevent the model from depending too much on specific neurons. Batch Normalization makes training smoother by normalizing intermediate feature values. Early Stopping prevents unnecessary training after validation loss stops improving. L2 regularization is applied through the optimizer to reduce overly large weights.

The latent dimension is also a form of regularization. By reducing the feature dimension from 128 to 64, the model is forced to keep only the most useful information from the CNN feature vectors.