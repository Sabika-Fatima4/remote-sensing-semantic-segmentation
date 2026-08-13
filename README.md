# Remote Sensing Semantic Segmentation

Semantic segmentation of remote-sensing imagery using U-Net models, transfer learning, class-balanced training, and Partial Cross Entropy with sparse point supervision.

## Project Overview

This project follows a four-stage model-development process on the DeepGlobe land-cover dataset:

**Initial baseline → class-balanced model → pretrained ResNet34 U-Net → Partial Cross Entropy with sparse point labels**

The goal was first to develop a strong fully supervised segmentation baseline, then investigate whether similar performance could be retained when only a very small fraction of pixels were labeled.

---

## Dataset

**Dataset:** `ratnaonline1/deepglobe-land-cover-classification-dataset`

The masks contain 7 land-cover classes:

| ID | Class |
|---:|---|
| 0 | Urban |
| 1 | Agriculture |
| 2 | Rangeland |
| 3 | Forest |
| 4 | Water |
| 5 | Barren |
| 6 | Unknown |

### Preprocessing

- Images resized to **256 × 256**
- Images resized with bilinear interpolation
- Masks resized with nearest-neighbor interpolation
- RGB masks converted to integer class IDs
- Train/validation split: **80/20**
- Random seed: **42**
- Batch size: **8**

---

## Model Development

The models were developed progressively rather than as isolated experiments.

### 1. Initial U-Net

A small U-Net was implemented from scratch with:

- Double convolution blocks
- 3 encoder stages
- Max pooling
- 256-channel bottleneck
- Transposed-convolution decoder
- 7-class output layer
- Cross Entropy Loss
- Adam optimizer

**Result**

| Metric | Score |
|---|---:|
| Mean IoU | **15.45%** |
| Mean Dice | **22.01%** |

The initial model showed weak overall segmentation and failed to learn several classes effectively.

---

### 2. Class-Balanced Model

Class-balanced training was introduced to address the poor performance of underrepresented/difficult classes.

**Result**

| Metric | Score |
|---|---:|
| Mean IoU | **24.03%** |
| Mean Dice | **32.96%** |

The change improved the overall result, with a particularly strong improvement for Forest:

- Forest IoU: **9.34% → 49.17%**

However, Rangeland, Barren, and Unknown remained difficult.

---

### 3. Best Fully Supervised Baseline

The next stage used a **ResNet34 U-Net** with ImageNet-pretrained encoder weights.

This became the strongest fully supervised reference model.

**Result**

| Metric | Score |
|---|---:|
| Mean IoU | **55.74%** |
| Mean Dice | **66.27%** |

This model was then used as the baseline for the sparse-supervision experiment.

---

## Partial Cross Entropy Experiment

### Why Partial CE?

The next question was:

> **Can useful segmentation performance be retained when full pixel-level supervision is replaced by extremely sparse point labels?**

Instead of using the complete segmentation mask during training, **100 pixels were randomly selected per image** and only those pixels retained their class labels.

All other pixels were marked as unlabeled.

For a 256 × 256 image:

- Total pixels = **65,536**
- Labeled points = **100**
- Labeled fraction ≈ **0.153%**

For a batch of 8 images:

- Labeled pixels = **800**
- Total pixels = **524,288**
- Labeled fraction = **0.1526%**

### Partial Cross Entropy

Cross entropy was computed only at labeled pixel locations. Unlabeled pixels were assigned an ignore value of `-1` and did not contribute to the loss.

Training setup:

- Model: **ResNet34 U-Net**
- Encoder initialization: **ImageNet**
- Loss: **Partial Cross Entropy**
- Optimizer: **AdamW**
- Learning rate: **1e-4**
- Weight decay: **1e-4**
- Epochs: **15**
- Points per image: **100**
- Validation-based best-checkpoint selection

---

## Final Results

### Full-mask baseline vs. Partial CE

| Metric | Full-mask ResNet34 U-Net | Partial CE |
|---|---:|---:|
| Mean IoU | **55.74%** | **53.31%** |
| Mean Dice | **66.27%** | **62.93%** |
| Training supervision | Full mask | 100 points/image |
| Labeled pixels | 100% | **≈0.15%** |

The Partial CE model lost only:

- **2.43 percentage points** Mean IoU
- **3.34 percentage points** Mean Dice

despite using only approximately **0.15% of the pixels as labeled supervision**.

### Partial CE per-class IoU

| Class | IoU |
|---|---:|
| Urban | 64.19% |
| Agriculture | **82.44%** |
| Rangeland | 13.49% |
| Forest | **78.48%** |
| Water | 75.85% |
| Barren | 58.70% |
| Unknown | 0.00% |
| **Mean** | **53.31%** |

Rangeland was the main class affected by sparse supervision, while Agriculture, Forest, and Water remained comparatively strong.

---

## Predicted Class Distribution

| Class | Full-mask baseline | Partial CE |
|---|---:|---:|
| Urban | 9.817% | 9.335% |
| Agriculture | 52.894% | **63.531%** |
| Rangeland | 10.641% | **2.414%** |
| Forest | 12.206% | 10.307% |
| Water | 5.006% | 4.449% |
| Barren | 9.436% | 9.964% |
| Unknown | 0.000% | 0.000% |

The sparse-supervision model predicts substantially more Agriculture and substantially less Rangeland, matching the per-class IoU results.

---

## Key Takeaway

The project shows a clear progression:

```text
Initial U-Net
    ↓
15.45% Mean IoU
    ↓
Class-balanced training
    ↓
24.03% Mean IoU
    ↓
ResNet34 U-Net + transfer learning
    ↓
55.74% Mean IoU
    ↓
Partial CE + 100 points/image
    ↓
53.31% Mean IoU
```

The main finding is that **Partial Cross Entropy retained most of the performance of the fully supervised baseline while reducing the labeled supervision to approximately 0.15% of image pixels**.

---

## Repository Contents

- `remotesensing-4.ipynb` — complete notebook containing all four model stages and experiments
- `Remote_Sensing_Partial_CE_Report.docx` — technical report
  

Large datasets and model checkpoints are intentionally excluded from the repository.

---

## Technologies

- Python
- PyTorch
- Segmentation Models PyTorch
- U-Net
- ResNet34
- Cross Entropy Loss
- Partial Cross Entropy
- NumPy
- Matplotlib
- Hugging Face Datasets
