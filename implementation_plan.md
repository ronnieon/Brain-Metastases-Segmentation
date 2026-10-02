# BUC-Net: 3D Bottleneck U-Net Cascade Implementation Plan

## Background

The paper proposes **3D BUC-Net** (Bottleneck U-Net Cascade), a two-stage cascaded 3D segmentation network for automatic brain metastases (BMs) detection and segmentation in MRI. The key innovations over a standard 3D U-Net are:

1. **Bottleneck Modules** in every encoder/decoder block — reduce parameters while maintaining accuracy.
2. **Two-Stage Cascade** — Stage 1 produces a coarse segmentation; Stage 2 takes the raw MRI **+** the Stage 1 coarse map to produce a refined segmentation.
3. **Stage-specific Loss Functions** — Stage 1 uses a weighted Dice+CE loss; Stage 2 uses Dice+Focal loss.
4. **Instance Normalization + LeakyReLU** instead of Batch Norm + ReLU.
5. **Gaussian fusion** for patch stitching during inference (instead of simple averaging).
6. **Largest connected component post-processing** after inference.

The existing notebook (`main_tf.ipynb`) has a basic 2-encoder 3D U-Net with BN+ReLU, a simple averaging stitcher, and BCE loss — which is essentially an incomplete baseline. We will replace it entirely with the BUC-Net.

---

## Open Questions

> **Q1: Input channels for your dataset?**
> The paper uses single-channel MRI (T1W-mDIXON or T1W-TFE with skull stripping). Your existing notebook uses **4-channel** BraTS2020 data (FLAIR, T1, T1ce, T2). The BUC-Net paper does not mention multi-channel input — should I:
> Keep 4-channel input (FLAIR, T1, T1ce, T2) and adapt BUC-Net to accept 4 channels (recommended, uses more information)?


> **Q2: Patch size adaptation?**
> The paper uses `32 × 320 × 320` patches on a **40GB A100 GPU**. BraTS2020 volumes are `240 × 240 × 155`. This means the 320×320 spatial size doesn't fit, and the depth dimension is different. I plan to use **`64 × 128 × 128`** patches (a GPU-friendly size for standard hardware), preserving the asymmetric depth-first design. Is this acceptable, or do you have a specific GPU/memory target?

> **Q3: End-to-end vs. sequential training?**
> The paper says the two stages are trained "end-to-end." However, an alternative is to train Stage 1 first (with its own loss), then freeze it and train Stage 2. End-to-end is tricky in TF/Keras with different losses per stage. I plan to implement:
> - **End-to-end** (one combined model): Both stages share gradients, total loss = L1 + L2.
> This is the most faithful to the paper. Shall I proceed this way?

> **Q4: Binary vs. multi-class segmentation?**
> The paper reports both binary (tumor vs. background) and multi-class (tumor sub-regions) results. Since you're using BraTS2020 (which has 4 classes: 0=background, 1=necrosis, 2=edema, 4=enhancing), should I implement:
> Multi-class (4 classes) — more informative for BraTS, but increases complexity?

---

## Proposed Changes

### Component 1: Architecture — 3D Bottle U-Net

The `3D Bottle U-Net` is the backbone for both stages.

**Bottleneck Module** (replaces standard double conv):
```
Input → Conv3D(1×1×1, C//4) → IN → LeakyReLU →
        Conv3D(3×3×3, C//4) → IN → LeakyReLU →
        Conv3D(1×1×1, C)    → IN → LeakyReLU
```

**Encoder path**: 5 levels with channels `[16, 32, 64, 128, 256]`
- Each level: BottleneckModule → MaxPool3D(2×2×2) for downsampling

**Decoder path**: 5 levels with channels `[256, 128, 64, 32, 16]`
- Each level: ConvTranspose3D(2×2×2) → Concatenate with skip → BottleneckModule

**Output**: Conv3D(1×1×1, sigmoid) for binary or softmax for multi-class

---

### Component 2: Architecture — BUC-Net (Cascade)

```
Input MRI (B, D, H, W, C)
    │
    ▼
┌─────────────────┐
│  Stage 1:       │  Loss L1 = 0.4·Dice + 0.6·CE
│  3D Bottle      │
│  U-Net          │
└────────┬────────┘
         │ Coarse seg map (B, D, H, W, 1)
         ▼
   Concatenate with raw MRI input
         │ (B, D, H, W, C+1)
         ▼
┌─────────────────┐
│  Stage 2:       │  Loss L2 = Dice + Focal
│  3D Bottle      │
│  U-Net          │
└────────┬────────┘
         │ Refined seg map (B, D, H, W, 1)
         ▼
      Output
```

Both stages share the **same Bottle U-Net architecture** but have **independent weights** (different instances).

Total loss: `L_total = L1 + L2`

---

### Component 3: Custom Loss Functions

**Stage 1** — `DiceCELoss(alpha=0.4, beta=0.6)`:
```python
L1 = 0.4 * dice_loss + 0.6 * binary_cross_entropy
```

**Stage 2** — `DiceFocalLoss(gamma=2)`:
```python
L2 = dice_loss + focal_loss(gamma=2, class_weights_inversely_proportional_to_size)
```

Both are implemented as custom `tf.keras.losses.Loss` subclasses.

---

### Component 4: Data Pipeline

**Preprocessing**:
- Z-score normalization (per-volume, on non-zero voxels) — same as existing notebook
- Crop to non-zero bounding box
- Resample to median voxel spacing (for BraTS2020, spacing is already uniform, so this is a no-op)

**Augmentation** (per paper):
- Random flip (axes: left-right, anterior-posterior)
- Random zoom
- Random elastic deformation
- Gamma adjustment
- Mirroring

**Patch sampling**:
- Patch size: `64 × 128 × 128` (or configurable)
- Tumor-center sampling bias (50% patches centered near tumor, 50% random)

---

### Component 5: Training Configuration

| Parameter | Paper | This implementation |
|-----------|-------|---------------------|
| Optimizer | RMSprop | RMSprop |
| Initial LR | 0.001 | 0.001 |
| LR decay | Not specified | Cosine decay or ReduceLROnPlateau |
| Batch size | 2 | 2 |
| Epochs | 150 | 150 |
| Patch size | 32×320×320 | 64×128×128 |
| Dropout | 0.5 (in encoders) | 0.5 |

---

### Component 6: Inference Pipeline

**Sliding window** with:
- Patch size matching training (e.g., `64 × 128 × 128`)
- Stride = 50% of patch size (overlapping)
- **Gaussian weighting** for patch stitching (center voxels trusted more than edges)
- **Largest connected component** post-processing

---

### Component 7: Metrics

Implement all evaluation metrics from the paper:
- DSC (Dice Similarity Coefficient)
- HD95 (95th percentile Hausdorff Distance)
- ASD (Average Surface Distance)
- Precision, Recall (per-lesion)
- ROC/PR curves with AUC

---

### New File

#### [NEW] [buc_net_tf.ipynb](file:///Users/rajnishdadarwal/Documents/Files/Machine%20Learning/Brain%20Metastases/Codebase/buc_net_tf.ipynb)

New TensorFlow Jupyter notebook implementing BUC-Net with the following cell structure:

| Cell | Contents |
|------|----------|
| 0 | Imports & GPU config |
| 1 | Config constants (paths, patch size, LR, etc.) |
| 2 | Dataset loading & visualization |
| 3 | Preprocessing (z-score normalization, crop, augmentation) |
| 4 | Patch extraction & BrainTumor data generator (updated) |
| 5 | **Bottleneck module** implementation |
| 6 | **3D Bottle U-Net** implementation |
| 7 | **BUC-Net cascade** model assembly (Stage 1 + Stage 2) |
| 8 | **Custom loss functions** (DiceCE for S1, DiceFocal for S2) |
| 9 | Training loop |
| 10 | **Gaussian inference** with sliding window |
| 11 | Post-processing (largest connected component) |
| 12 | Evaluation metrics (DSC, HD95, ASD, ROC/PR curves) |
| 13 | Visualization |

---

## Verification Plan

### Automated Tests
- Confirm model builds without error: `model.summary()`
- Confirm forward pass with dummy data: `model(tf.zeros([1, 64, 128, 128, 4]))`
- Confirm loss computes correctly on dummy batch
- Confirm output shapes match expected (binary: `[B, 64, 128, 128, 1]`)

### Manual Verification
- Confirm bottleneck module reduces parameters vs standard double conv
- Train for 2 epochs to confirm no NaN loss
- Visualize Stage 1 vs Stage 2 prediction improvement on a sample patient



>>>>>>>> When ready for full training: just set LOW_MEMORY_MODE = False at the top of the config cell. All paper-original values are preserved in the else branches.



Why the model predicts zero Necrosis
Three compounding reasons:

1. Necrosis is extremely rare — In BraTS2020, the class distribution is roughly:

Background: ~97%
Edema: ~1.5%
Necrosis: ~0.8% ← rarest foreground class
Enhancing: ~0.7%

A model with too few training steps defaults to predicting Background everywhere — it's the "safe" low-loss option.

2. Training is only 2 epochs with 4 steps/epoch — That's just 8 gradient updates total. The model is essentially random. It would take hundreds of epochs to learn Necrosis reliably (hence the paper uses 120 epochs).

3. Small patches may miss Necrosis entirely — With PATCH_SIZE=(64,64,64) and tumor-biased sampling, many patches won't contain any Necrosis voxels, so the model never gets loss signal for that class.

This will improve automatically with more training
Once you switch to a GPU and set LOW_MEMORY_MODE = False (→ EPOCHS=120, BATCH_SIZE=2, PATCH_SIZE=(128,128,128)), the model will see enough Necrosis examples over time. The paper reports DSC ≈ 0.78 for Necrosis after full training. The CLASS_WEIGHTS = [0.1, 3.5, 2.0, 3.5] in the Focal loss are already set to up-weight Necrosis (3.5×) to help with exactly this imbalance.





This reveals something important about how the model is failing, not just that it's undertrained.

Why Enhancing works but Necrosis doesn't (despite similar rarity)
The key is where these two classes live spatially inside the brain:

Whole tumour:
┌─────────────────────────────┐
│  Peritumoral Edema(label 2) │
│  ┌──────────────────────┐   │
│  │      Tumour Core     │   │
│  │  ┌────────────────┐  │   │  ← Necrosis (label 1) sits INSIDE Enhancing (label 3)
│  │  │Necrosis(label1)│  │   │   
│  │  └────────────────┘  │   │
│  │  Enhancing(label 3)  │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
Necrosis sits inside the tumour core, surrounded by Enhancing tumour. The model is almost certainly doing this:

It detects the tumour core region correctly
But labels the entire core as Enhancing (class 3) instead of splitting it into Necrosis (class 1) + Enhancing (class 3)
Result: Enhancing gets true positives from the border regions → non-zero DSC
Necrosis gets zero predicted voxels → DSC=0, Precision=NaN, Recall=0








Is Stage 2 adding onto the findings of Stage 1, or are they separate?
Stage 2 is adding onto the findings of Stage 1. They are part of a connected cascaded pipeline, not separate models acting independently.

Here is exactly how they interact:

Stage 1 (The "Scout"): The raw MRI goes into Stage 1. Stage 1 outputs a "coarse" (rough) guess of where the tumor sub-regions are.
The Hand-off: This coarse prediction from Stage 1 is then concatenated (stacked as a 5th channel) onto the original 4 MRI channels.
Stage 2 (The "Refiner"): Stage 2 looks at both the original MRI plus the hints provided by Stage 1. Because Stage 2 already has a rough map of the tumor to start with, it can focus its learning capacity on refining the tricky boundaries (like the thin layers of edema or small necrotic cores).
They are trained together end-to-end. Stage 1's goal isn't to be perfect, but to give Stage 2 the best possible starting hint.

2. Why is Stage 1 only predicting Enhancing while Stage 2 only predicts Edema?
This behavior is a classic symptom of the model being severely undertrained in your current test run (due to being run for only a few epochs with very few steps to save memory).

Because the model has only seen a handful of gradient updates, it hasn't learned the complex relationships of the tumor yet. Instead, it is taking "lazy" shortcuts to minimize the loss function as fast as possible:

Stage 1 latching onto Enhancing: The enhancing tumor core is highly visible and very bright on the T1ce MRI modality. In early training, Stage 1 might just act like a simple brightness threshold on the T1ce channel, completely missing the more subtle edema and necrosis.
Stage 2 latching onto Edema: Edema is the largest foreground class (it takes up the most volume). It is also extremely bright on the FLAIR MRI modality. Because Stage 1 handed Stage 2 a map that only contained the enhancing core, Stage 2 might have learned to look at the FLAIR channel and predict the large, bright edema areas to quickly boost its Dice score, ignoring the core regions entirely.
How this resolves: When you run the model for the full 120 epochs (with the LOW_MEMORY_MODE = False configuration), the optimizer will force both stages out of these "lazy" local minima. Stage 1 will learn to identify the whole tumor region coarsely, and Stage 2 will learn to segment all 4 classes (Background, Necrosis, Edema, Enhancing) accurately.