# Dataset Information

Dataset: UTA-RLDD Dataset

Task: Binary Driver Drowsiness Detection

Used classes:

* Alert
* Drowsy

Excluded class:

* Low Vigilance

Subset:

* Fold1_part1
* Fold1_part2

Number of videos:

* Alert: 12
* Drowsy: 12
* Total: 24

Frame extraction settings:

* Image size: 64x64
* Sequence length: 10 frames
* Frame step: 10
* Maximum frames per video: 300

Final dataset:

* Total frames: 7200
* Alert frames: 3600
* Drowsy frames: 3600
* Total sequences: 720

Final data shape:

* sequences.npy: (720, 10, 64, 64, 3)
* labels.npy: (720,)

Label mapping:

* Alert: 0
* Drowsy: 1

Train/Validation/Test split:

* Train: 70%
* Validation: 15%
* Test: 15%

Final split shapes:

* Train: (504, 10, 64, 64, 3)
* Validation: (108, 10, 64, 64, 3)
* Test: (108, 10, 64, 64, 3)
