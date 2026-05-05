# Marine Mammal Classification: Dolphin vs Whale

### CNN Image Classification with PyTorch — IF3270 Machine Learning Praktikum 3

---

## Project Overview

A deep learning project that classifies marine mammal images into **two classes**:
- 🐬 **Class 0** — Dolphin (*Lumba-lumba*)
- 🐋 **Class 1** — Whale (*Paus*)

Built using **Convolutional Neural Networks (CNN)** from scratch and **Transfer Learning** with EfficientNet-B0.

---

## Dataset Summary

| Metric | Value |
|---|---|
| Total Samples | 138 |
| Training Set | 96 |
| Validation Set | 21 |
| Test Set | 21 |
| Classes | 2 (Dolphin, Whale) |
| Image Size | Resized to 224×224 |
| Imbalance Ratio | 1.09:1 (near balanced) |

---

## Models Compared

### AlexNet (From Scratch)
- Custom-built CNN architecture replicating AlexNet design
- **58.3M** trainable parameters
- Trained from zero on marine mammal data
- Overfitting observed due to limited training data (96 samples)

### EfficientNet-B0 (Transfer Learning)
- Pretrained on **ImageNet** (1.2M images, 1000 classes)
- Fine-tuned in **two phases**: frozen backbone → full fine-tuning
- **4.0M** total parameters (~14× fewer than AlexNet!)
- Achieved **85.6% Macro F1** on test set vs 34% for AlexNet Scratch

---

## Performance Comparison

| Model | Macro F1 | Accuracy | Total Parameters |
|---|---|---|---|
| AlexNet Scratch | 0.3438 | 52.38% | 58,289,538 |
| **EfficientNet-B0 (FT)** | **0.8558** | **85.71%** | **4,010,110** |

> Transfer learning wins decisively — **14× fewer parameters, 2.5× better F1 score.**

---

## Workflow

### Stage 1 — Exploratory Data Analysis (EDA)
- Class distribution analysis → confirmed near-balanced dataset
- Image dimension & aspect ratio variation → justified 224×224 resizing
- RGB pixel intensity distribution → color not discriminative; shape/texture needed
- PCA 2D projection → classes overlap in pixel space; CNN required

### Stage 2 — Data Preprocessing
- Resize all images to 224×224 (AlexNet/EfficientNet standard)
- ImageNet normalization (mean/std)
- Data augmentation: `RandomHorizontalFlip`, `RandomVerticalFlip`, `Rotation (±20°)`, `ColorJitter`, `RandomGrayscale`
- Stratified train/val/test split (70/15/15)

### Stage 3 — Modeling & Training
- **Phase 1**: Train classifier head only (backbone frozen, lr=3e-3)
- **Phase 2**: Fine-tune full model (lr=3e-5, CosineAnnealingLR)
- Early stopping based on validation Macro F1

### Stage 4 — Analysis & Evaluation
- Confusion matrix, per-class F1, learning curves
- Bias-variance trade-off analysis
- Transfer learning vs from-scratch trade-off discussion

---

## Tech Stack

| Category | Library |
|---|---|
| Deep Learning | `PyTorch` |
| Computer Vision | `torchvision`, `PIL`, `OpenCV` |
| Data Analysis | `NumPy`, `Pandas`, `Matplotlib`, `Seaborn` |
| ML Metrics | `scikit-learn` |
| Experiment Tracking | Local (Google Drive) |

---

## Collaborators

| Name | Student ID |
|---|---|
| **Danendra Shafi Athallah** | 13523136 |
| **Jonathan Kenan Budianto** | 13523139 |

> Group Number: **40** | Course: **IF3270 Machine Learning**

---

## Key Insights

1. **Dataset size is the bottleneck** — With only 138 images, training from scratch easily overfits; transfer learning is essential.
2. **Color is not discriminative** — Both dolphins and whales appear in similar ocean backgrounds; the model must rely on **shape and texture features**.
3. **Pretrained features transfer well** — ImageNet includes marine animal classes, so EfficientNet-B0's features are highly relevant out-of-the-box.
4. **Parameter count ≠ performance** — EfficientNet-B0 (4M params) dramatically outperforms AlexNet (58M params) on this task.
5. **Two-phase fine-tuning works** — Freezing backbone first, then unfreezing with a small learning rate, prevents catastrophic forgetting.
