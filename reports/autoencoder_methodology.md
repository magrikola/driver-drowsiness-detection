# Autoencoder Methodology

In this project, the Autoencoder is used as a feature compression block after the CNN feature extractor. The CNN model extracts a 128-dimensional feature vector from each video frame. Since each video sample contains 10 frames, the input of the Autoencoder has the shape [batch_size, 10, 128].

The main purpose of the Autoencoder is to learn a compressed latent representation of the CNN features. In our architecture, the encoder reduces each 128-dimensional feature vector into a 64-dimensional latent vector. This means that the Autoencoder tries to keep the most important information while reducing the feature size.

The decoder reconstructs the original 128-dimensional CNN feature vector from the compressed latent representation. During training, the Autoencoder compares the reconstructed feature vector with the original CNN feature vector. Mean Squared Error loss is used for this reconstruction task.

The latent output of the encoder has the shape [batch_size, 10, 64]. This compressed sequence can be passed to the GRU model for temporal learning. In that case, the GRU input dimension should be changed from 128 to 64.

The Autoencoder also includes regularization techniques such as Dropout, Batch Normalization, Early Stopping, and L2 regularization. These methods help reduce overfitting and make the model more stable, especially because the project uses a limited subset of the full dataset.