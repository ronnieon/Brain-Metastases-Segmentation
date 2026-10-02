# 3D BUC-Net for Brain Metastases Segmentation

This repository contains a TensorFlow/Keras implementation of the **3D BUC-Net (Bottleneck U-Net Cascade)** for the automatic segmentation of brain metastases in MRI scans.

**Reference Paper**: *Automatic segmentation of brain metastases in MRI using 3D BUC-Net* (Scientific Reports, 2025)

---

## 🧠 Architecture Overview

The model uses a cascaded architecture designed to accurately segment brain tumors (including different sub-regions like edema, necrosis, and enhancing core).

*   **Stage 1 (Coarse Segmentation)**: A 3D Bottle U-Net takes the input MRI and generates an initial, coarse segmentation map.
    *   **Loss**: $0.4 \times \text{Dice Loss} + 0.6 \times \text{Cross-Entropy Loss}$
*   **Stage 2 (Refined Segmentation)**: A second 3D Bottle U-Net takes the original input MRI concatenated with the coarse segmentation map from Stage 1 to produce the final, refined segmentation.
    *   **Loss**: $\text{Dice Loss} + \text{Focal Loss} (\gamma=2)$

### Key Components
*   **Bottleneck Blocks**: Used to reduce computational complexity while maintaining representational power. The bottleneck sequence is: `1×1×1 Convolution (Reduce)` $\rightarrow$ `3×3×3 Convolution (Spatial)` $\rightarrow$ `1×1×1 Convolution (Restore)`.
*   **Normalization**: Instance Normalization (`InstanceNorm`).
*   **Activation**: Leaky ReLU (`alpha=0.01`).
*   **Input**: 4-channel 3D MRI (FLAIR, T1, T1ce, T2).
*   **Output Classes**: 4 classes (Background, Necrosis, Edema, Enhancing Tumor).

---

## 📊 Dataset: BraTS 2020

The code is configured to run on the **BraTS 2020** dataset.
*   **Label Mapping**:
    *   `0`: Background (Class 0)
    *   `1`: Necrotic Tumor Core (Class 1)
    *   `2`: Peritumoral Edema (Class 2)
    *   `4`: GD-Enhancing Tumor (Class 3)
*   **Modalities Used**: FLAIR, T1, T1ce, T2.

---

## ⚙️ Configuration & Hyperparameters

The notebook (`buc_net_tf.ipynb`) includes a dynamic configuration system to easily switch between a low-resource testing environment and a full GPU training environment.

### Toggle: `LOW_MEMORY_MODE`
Set this flag at the top of the configuration cell to adjust hyperparameters automatically.

| Parameter | `LOW_MEMORY_MODE = True` (e.g., 8GB Mac) | `LOW_MEMORY_MODE = False` (GPU Workstation) |
| :--- | :--- | :--- |
| **Patch Size** | `64 × 64 × 64` | `128 × 128 × 128` |
| **Stride (Overlap)** | `32 × 32 × 32` (50%) | `64 × 64 × 64` (50%) |
| **Batch Size** | `1` | `2` |
| **Epochs** | `6` (Smoke test) | `120` |
| **Encoder Channels** | `[8, 16, 32, 64, 128]` | `[16, 32, 64, 128, 256]` |

### General Training Settings
*   **Optimizer**: RMSprop
*   **Learning Rate**: `0.001`
*   **Dropout Rate**: `0.5`
*   **Train/Val Split**: 80% / 20%
*   **Class Weights (Focal Loss)**: `[0.1, 3.5, 2.0, 3.5]` (Inversely proportional to BraTS class frequencies).

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed:
*   TensorFlow (>= 2.x)
*   Nibabel (for NIfTI image loading)
*   NumPy, Matplotlib

### 2. Dataset Setup
Download the BraTS 2020 dataset and place the training data in the following directory structure:
```
Dataset/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    ├── BraTS20_Training_002/
    └── ...
```
*(You can modify `DATA_DIR` in Cell 1 to point to your custom dataset path).*

### 3. Running the Notebook
1.  Open `buc_net_tf.ipynb`.
2.  Set `LOW_MEMORY_MODE` to `True` if you are testing on a laptop, or `False` if you are ready to train on a GPU.
3.  Run the cells sequentially. The notebook handles dataset loading, patch extraction, model building, and training automatically.
4.  Trained models and checkpoints will be saved to the `saved_models/buc_net/` directory.

---

## 📝 Custom Loss Functions

The model utilizes custom hybrid loss functions to handle the severe class imbalance present in medical image segmentation:

*   **Stage 1 (`DiceCELoss`)**: Combines standard Dice loss with Binary Cross Entropy. Good for rough localization.
*   **Stage 2 (`DiceFocalLoss`)**: Combines standard Dice loss with Categorical Focal Loss. Excellent for refining hard-to-segment boundaries (like thin layers of edema or small enhancing cores).
