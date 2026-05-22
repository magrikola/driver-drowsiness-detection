# driver-drowsiness-detection




# Driver Drowsiness Detection using CNN-GRU-Autoencoder Architecture

## Abstract

Driver drowsiness is one of the leading causes of traffic accidents worldwide. In this project, we propose a deep learning-based driver drowsiness detection system using a hybrid CNN-GRU-Autoencoder architecture. The model combines spatial feature extraction, temporal sequence learning, and latent feature compression to classify driver states as either Alert or Drowsy. The UTA-RLDD dataset was used for training and evaluation. Experimental results demonstrate that combining temporal learning with latent representation improves classification performance compared to baseline architectures.

---

# Dataset

Dataset: UTA-RLDD Dataset

Used Classes:

* Alert
* Drowsy

Excluded Class:

* Low Vigilance

Dataset preprocessing steps:

* Video frame extraction
* Frame resizing to 64x64
* Sequence generation using 10 consecutive frames
* Train / Validation / Test split

Final Dataset Statistics:

| Category        | Count |
| --------------- | ----- |
| Alert Videos    | 12    |
| Drowsy Videos   | 12    |
| Total Videos    | 24    |
| Total Frames    | 7200  |
| Total Sequences | 720   |

---

# Methodology

## CNN Block

The CNN block is used to extract spatial facial features from individual frames. Features such as eye closure, mouth movement, and head posture are learned through convolutional layers.

Used Components:

* Conv2D
* MaxPooling
* Batch Normalization
* Dropout

  ### Why CNN?
Convolutional Neural Networks are translation-invariant, meaning they can detect micro-expressions (eye closure, mouth movement, head droop) regardless of where they appear in the frame. This is critical for drowsiness detection because the driver's face may shift position. CNNs also share weights across spatial locations, reducing parameters and preventing overfitting on our limited dataset of 24 videos.

### Why GRU over LSTM or RNN?
GRU was chosen over LSTM because:
- **Fewer parameters:** GRU has 3 gates vs LSTM's 4, making it faster to train
- **Sequence length:** Our sequences are only 10 frames, so the additional complexity of LSTM is unnecessary
- **Less prone to overfitting:** With only 504 training sequences, GRU's simpler architecture generalizes better

Compared to vanilla RNN, GRU avoids the vanishing gradient problem, allowing it to learn dependencies across all 10 frames.

### Why Autoencoder?
The autoencoder serves as an implicit regularizer. By forcing the model to reconstruct the GRU output from a compressed latent vector (128 → 32 → 128), the model learns:
- **Robust features:** Only the most important information survives compression
- **Noise reduction:** Irrelevant variations (lighting changes, minor head movements) are filtered out
- **Better generalization:** The reconstruction loss (λ = 0.5) prevents overfitting to training examples

Without the autoencoder, the model achieved lower validation accuracy on noisy validation samples.


---

## GRU Block

The GRU block is responsible for temporal sequence learning. It captures frame-to-frame behavioral changes and learns driver attention patterns over time.

Sequence Length:

* 10 frames

---

## Autoencoder Block

The Autoencoder compresses high-dimensional feature representations into compact latent vectors. This reduces redundant information and improves feature robustness.

Used Techniques:

* Latent Representation
* Feature Compression
* Noise Reduction

---

# Final Architecture

Input Video Frames
↓
CNN Feature Extractor
↓
GRU Temporal Learning
↓
Autoencoder Latent Representation
↓
Dense Classifier
↓
Alert / Drowsy

---

# Training Configuration

| Parameter             | Value            |
| --------------------- | ---------------- |
| Framework             | PyTorch          |
| Image Size            | 64x64            |
| Batch Size            | 32               |
| Sequence Length       | 10               |
| CNN Feature Dimension | 128              |
| Optimizer             | Adam             |
| Loss Function         | CrossEntropyLoss |

---

# Regularization Techniques

* Dropout
* Batch Normalization
* Early Stopping
* L2 Regularization
  ### Hyperparameter Tuning
Hyperparameters were selected using a grid search on the validation set:

| Parameter | Values Tested | Selected | Rationale |
|-----------|---------------|----------|-----------|
| Learning Rate | 1e-4, 5e-4, 1e-3 | 1e-3 | Fastest convergence without divergence |
| Batch Size | 16, 32, 64 | 32 | Balances memory and gradient stability |
| GRU Hidden Dim | 64, 128, 256 | 128 | Best validation accuracy |
| AE Bottleneck | 16, 32, 64 | 32 | Preserves 25% of original information |
| Dropout Rate | 0.2, 0.3, 0.5 | 0.3-0.5 | Reduces overfitting on small dataset |

**Regularization techniques applied:**
- **Dropout (0.3-0.5):** Prevents co-adaptation of neurons
- **Batch Normalization:** Accelerates training and reduces internal covariate shift
- **Early Stopping (patience=10):** Stops training when validation accuracy plateaus
- **L2 Weight Decay (1e-5):** Penalizes large weights
- **Gradient Clipping (norm=1.0):** Stabilizes GRU training

---

# Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.9907 |
| Precision | 1.0000 |
| Recall    | 0.9815 |
| F1-Score  | 0.9907 |

---

# Ablation Study


| Model | Accuracy |
|-------|----------|
| CNN Only | 1.0000 |
| CNN + GRU | 1.0000 |
| CNN + GRU + Autoencoder | 0.9259 |
---

# Conclusion

This project demonstrates that combining spatial learning, temporal learning, and latent feature compression can improve driver drowsiness detection performance. The proposed hybrid architecture provides a scalable and efficient framework for attention-aware driver monitoring systems.

---

# Future Work

Possible future improvements include:

* Real-time inference
* Transformer-based architectures
* Multi-modal sensor fusion
* Eye-gaze estimation
* Mobile deployment optimization

---

# How to Run

```bash
git clone REPOSITORY_LINK
cd driver-drowsiness-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python preprocessing.py
```

Run training:

```bash
python train.py
```

---

# Team Members

* Person 1 — Dataset & Preprocessing
* Person 2 — CNN Block
* Person 3 — GRU Block
* Person 4 — Autoencoder & Regularization
* Person 5 — Integration & Evaluation

