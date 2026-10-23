# Driver Drowsiness Detection with a CNN-GRU-Autoencoder

Binary classification (Alert vs. Drowsy) of short facial-video sequences from the UTA-RLDD dataset. A CNN extracts features from each frame, a GRU models the 10-frame sequence, and an autoencoder is trained alongside as an auxiliary reconstruction objective. Team project (five members), implemented in PyTorch.

> **Read this first.** The results below come from a very small subset of the dataset (24 videos) and from a random split of sequences rather than of videos. They are likely optimistic. See [Limitations](#limitations) before interpreting them.

## Contents

- [Repository structure](#repository-structure)
- [Dataset and preprocessing](#dataset-and-preprocessing)
- [Model](#model)
- [Training configuration](#training-configuration)
- [Results](#results)
- [Limitations](#limitations)
- [Getting started](#getting-started)
- [Future work](#future-work)
- [Team](#team)
- [Dataset citation](#dataset-citation)

## Repository structure

```
models/         Block implementations written by team members (CNN, GRU, autoencoder)
notebooks/      Preprocessing and training notebooks (written for Google Colab)
reports/        Final report and supporting tables (dataset, tuning, regularization)
results/        Saved figures (confusion matrix, CNN metrics)
presentation/   Presentation notes
dataset/        Placeholder only; the dataset is not included
```

`notebooks/final_model_train.ipynb` contains the final experiment as a plain Python script (a Colab export saved with an `.ipynb` extension), so run it as a script or paste it into Colab rather than opening it as a notebook.

## Dataset and preprocessing

**Dataset:** UTA Real-Life Drowsiness Dataset (UTA-RLDD), 180 videos of 60 participants, about 10 minutes each. This project uses 24 videos from Fold 1:

| Class | Original label | Videos |
|---|---|---|
| Alert | 0 (KSS 1-3) | 12 |
| Drowsy | 10 (KSS 8-9) | 12 |
| Excluded | 5 (Low Vigilance) | 0 |

**Preprocessing** (`notebooks/data_preprocessing.ipynb`):

1. Keep every 10th frame of a video, up to 300 frames per video.
2. Convert BGR to RGB, resize to 64x64 and scale pixel values to [0, 1].
3. Group the kept frames into non-overlapping sequences of 10 frames (30 sequences per video).
4. Split the 720 sequences 70/15/15 with a stratified random split (`train_test_split`, `random_state=42`).

| Split | Sequences |
|---|---|
| Train | 504 |
| Validation | 108 |
| Test | 108 |

## Model

The final experiment is defined inline in `notebooks/final_model_train.ipynb`.

```
Input: 10 frames, 64x64x3
  -> CNN encoder (applied to each frame)         -> 128-d feature per frame
  -> GRU (2 layers, bidirectional, 128 hidden)   -> 128-d sequence vector
  -> Classifier: Linear(128,64) - ReLU - Dropout(0.5) - Linear(64,2)
  -> Alert / Drowsy

Auxiliary branch (training loss only):
  GRU output (128-d) -> Autoencoder 128-64-32-64-128 -> reconstruction of the GRU output
```

- **CNN encoder:** three Conv3x3 + BatchNorm + ReLU blocks (32, 64, 128 channels), max pooling after the first two, global average pooling, Linear(128,128) and Dropout(0.3).
- **GRU:** 2 layers, bidirectional, 128 hidden units, dropout 0.3, followed by Linear(256,128) on the last time step.
- **Autoencoder:** encoder 128 -> 64 -> 32, decoder 32 -> 64 -> 128. It reconstructs the GRU output and its MSE loss is added to the classification loss. **The classifier reads the GRU output directly; the autoencoder's latent code is not used for classification.**
- **Loss:** `CrossEntropy + 0.5 * MSE(reconstruction, GRU output)`.

Design intent: a CNN captures spatial cues such as eye closure and head posture, a GRU is a compact choice for short sequences, and the reconstruction task was meant to act as an implicit regularizer. The ablation below does not show a benefit from the GRU or the autoencoder on this data.

`models/` holds the individual blocks written by each team member. Their default settings (for example a unidirectional GRU with 64 hidden units and a 64-d autoencoder latent) differ from the final experiment above.

## Training configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Input | 10 frames of 64x64x3 |
| Batch size | 32 |
| Optimizer | Adam, learning rate 1e-3 |
| Weight decay (L2) | 1e-5 |
| Epochs | 50 |
| Model selection | Checkpoint with the best validation accuracy |
| Dropout | 0.3 (CNN head, GRU, GRU head), 0.5 (classifier) |
| Loss | CrossEntropy + 0.5 x reconstruction MSE |

## Results

Reported on the 108-sequence test set:

| Metric | Value |
|---|---|
| Accuracy | 0.9907 |
| Precision | 1.0000 |
| Recall | 0.9815 |
| F1-Score | 0.9907 |

Ablation as documented by the team:

| Model | Accuracy |
|---|---|
| CNN only | 1.0000 |
| CNN + GRU | 1.0000 |
| CNN + GRU + Autoencoder | 0.9259 |

The two tables disagree about the full model (0.9907 vs. 0.9259) and the ablation ranks the CNN-only model highest. Saved figures are in `results/`. Please read the limitations below.

## Limitations

- **Small dataset.** Only 24 of the 180 videos (Fold 1) are used, and only the first 300 kept frames of each video, giving 720 sequences.
- **Sequence-level split.** The split is applied to sequences, not to videos or subjects. Sequences from the same video can appear in both the training and the test set (data leakage), so the scores are likely optimistic and do not measure generalization to unseen drivers.
- **Small test set.** The test set has 108 sequences, so a few sequences change the accuracy noticeably (107/108 = 0.9907, 100/108 = 0.9259).
- **Inconsistent numbers.** The origin of the difference between the Results and Ablation tables is not documented and has not been re-verified. With the current evidence, the claim that the GRU and the autoencoder improve performance is not supported.
- **Autoencoder role.** In the final experiment the autoencoder only adds a reconstruction loss; it is not part of the classification path.
- **Documentation does not fully match the code.** The reports in `reports/` describe a hyperparameter grid search, early stopping with patience 10 and gradient clipping. The final training script contains none of these, and no grid-search or gradient-clipping code exists in the repository. An `EarlyStopping` helper (patience 5) in `models/autoencoder.py` is not used by the final script.
- **Reproducibility.** The notebooks use hard-coded Google Drive paths, the dataset is not included and there is no `requirements.txt`.
- **Suggested next steps.** Split by video or subject, use more videos and participants, repeat training with several random seeds and report confidence intervals.

## Getting started

Requirements: Python 3, PyTorch, NumPy, OpenCV, scikit-learn, matplotlib and seaborn. The notebooks were written for Google Colab with Google Drive.

1. Download the UTA-RLDD dataset (Fold 1 archives `Fold1_part1.zip` and `Fold1_part2.zip`) from its authors and place them in your Drive.
2. Open `notebooks/data_preprocessing.ipynb` in Colab and adjust the paths in the first cells. It extracts frames, builds the sequences and saves the `X_*.npy` and `y_*.npy` files.
3. Run `notebooks/final_model_train.ipynb` as a script in Colab (set `data_path` to the folder with the saved arrays). It trains the model, selects the best checkpoint on the validation set and prints the test metrics.
4. Optional: `notebooks/cnn_training.ipynb`, `notebooks/cnn_gru_training.ipynb` and `notebooks/autoencoder_test.ipynb` contain the earlier per-block experiments.

## Future work

- Split by video or subject and evaluate on the full dataset
- Real-time inference
- Transformer-based architectures
- Multi-modal sensor fusion and eye-gaze estimation
- Mobile deployment optimization

## Team

Team of five, with these responsibilities:

- Dataset and preprocessing
- CNN block
- GRU block
- Autoencoder and regularization
- Integration and evaluation

## Dataset citation

Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection.* IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
