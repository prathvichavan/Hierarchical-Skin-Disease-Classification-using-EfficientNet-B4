# 🩺 Hierarchical Skin Disease Classification using EfficientNet-B4

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-3.10.0-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/GPU-Tesla%20T4-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
</p>

<p align="center">
  <b>A two-level hierarchical deep learning framework for automated skin disease classification with clinical-grade recall and Grad-CAM explainability.</b>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Hierarchical Architecture](#-hierarchical-architecture)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [Explainability (Grad-CAM)](#-explainability-grad-cam)
- [External Evaluation](#-external-evaluation)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Inference Pipeline](#-inference-pipeline)
- [Future Work](#-future-work)
- [Citation](#-citation)

---

## 🧬 Overview

This project presents a **two-level hierarchical deep learning framework** for automated skin disease classification using **EfficientNet-B4** as the backbone.

Unlike flat classifiers, this pipeline:
1. First distinguishes **Benign vs. Malignant** lesions (clinical gatekeeper)
2. Then performs **fine-grained subtype classification** within each category

This hierarchical approach mirrors clinical decision-making and **maximizes recall for malignant cases** — minimizing false negatives, which is critical for patient safety.

---

## 🏗️ Hierarchical Architecture

```
Input Image
     │
     ▼
┌─────────────────────────┐
│   Level-1 Classifier    │  ← EfficientNet-B4 + Sigmoid
│   Benign vs Malignant   │  ← Threshold = 0.3 (recall-optimized)
└─────────────────────────┘
         │              │
    [Benign]       [Malignant]
         │              │
         ▼              ▼
┌──────────────┐  ┌──────────────┐
│  Level-2     │  │  Level-2     │
│  Benign      │  │  Malignant   │
│  Classifier  │  │  Classifier  │
│  (Softmax)   │  │  (Sigmoid)   │
└──────────────┘  └──────────────┘
   │   │    │         │       │
Inflam. Infec. Benign  BCC  Melanoma
       Tumors
```

### Class Hierarchy

| Level-1 | Level-2 | Diseases |
|---------|---------|----------|
| **Benign** | Inflammatory | Atopic Dermatitis, Eczema, Psoriasis Lichen Planus |
| **Benign** | Infectious | Tinea Ringworm, Warts Viral |
| **Benign** | Benign Tumors | Melanocytic Nevi, Benign Keratosis, Seborrheic Keratosis |
| **Malignant** | Melanoma | Melanoma |
| **Malignant** | BCC | Basal Cell Carcinoma |

---

## 📊 Dataset

**Source:** [Kaggle - Skin Disease Dataset (Balanced, 2.5K per class)](https://www.kaggle.com/datasets/prathvichavan/skin-diesease-dataset-per-class-2-5k-images)

| Split | Level-1 | Count |
|-------|---------|-------|
| Total | — | 25,000 |
| Train | Benign | 16,000 |
| Train | Malignant | 4,000 |
| Val | Benign | 4,000 |
| Val | Malignant | 1,000 |

### Level-2 Class Distribution

| Category | Class | Count |
|----------|-------|-------|
| Benign | Inflammatory | 7,500 |
| Benign | Benign Tumors | 7,500 |
| Benign | Infectious | 5,000 |
| Malignant | BCC | 2,500 |
| Malignant | Melanoma | 2,500 |

### Key Parameters

```python
IMG_SIZE     = 380       # EfficientNet-B4 native resolution
BATCH_SIZE   = 16
RANDOM_STATE = 42
```

---

## 🧠 Model Architecture

**Backbone:** EfficientNet-B4 pretrained on ImageNet

```
EfficientNet-B4 (frozen during warm-up)
        │
GlobalAveragePooling2D
        │
   Dropout(0.4)
        │
Dense(1, sigmoid)       ← Level-1 & Level-2 Malignant
Dense(3, softmax)       ← Level-2 Benign
```

### Why EfficientNet-B4?
- Compound scaling of depth, width, and resolution
- Native input resolution of 380×380 — ideal for dermoscopic images
- State-of-the-art accuracy-efficiency tradeoff
- Pretrained ImageNet weights for strong feature initialization

---

## 🏋️ Training Strategy

All models follow a **two-phase training** strategy:

### Phase 1 — Warm-up (Head Only)
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=3e-5) |
| Frozen | All EfficientNet layers |
| Epochs | 5–10 |
| Purpose | Train new classification head |

### Phase 2 — Fine-tuning (Partial Unfreeze)
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-5) |
| Unfrozen | Last 30 layers of EfficientNet-B4 |
| Epochs | 15–20 |
| Purpose | Domain adaptation |

### Callbacks
```python
EarlyStopping(patience=5, restore_best_weights=True)
ReduceLROnPlateau(patience=3, factor=0.3)
ModelCheckpoint(monitor='val_auc_pr', save_best_only=True)
```

### Class Imbalance Handling
```python
# Level-1 class weights (4:1 benign-to-malignant ratio)
class_weights = {0: 0.625, 1: 2.5}
```

### Decision Threshold Optimization (Level-1)
Since malignant cases are clinically critical, recall is prioritized over precision:

| Threshold | Recall (Malignant) |
|-----------|-------------------|
| 0.2 | 97.8% |
| **0.3** ✅ | **97.2%** |
| 0.4 | 96.3% |
| 0.5 | 95.3% |

**Final Threshold: 0.3** → False Negative Rate: **2.6%**

---

## 📈 Results

### Level-1: Benign vs. Malignant

| Metric | Benign | Malignant |
|--------|--------|-----------|
| Precision | 0.99 | 0.83 |
| **Recall** | 0.95 | **0.97** |
| F1-Score | 0.97 | 0.90 |
| AUC-PR | — | **~0.98** |

**Overall Accuracy: 96%** | **Val AUC: 0.98**

### Level-2 Benign: Inflammatory / Infectious / Benign Tumors

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Inflammatory | 0.79 | 0.84 | 0.81 |
| Infectious | 0.71 | 0.67 | 0.69 |
| Benign Tumors | 0.93 | 0.92 | 0.93 |
| **Macro Avg** | **0.81** | **0.81** | **0.81** |

**Validation Accuracy: 83%**

### Level-2 Malignant: BCC / Melanoma

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| BCC | 1.00 | 0.99 | 0.99 |
| Melanoma | 0.99 | 1.00 | 0.99 |
| **Overall** | **0.99** | **0.99** | **0.99** |

**Validation Accuracy: 99%**

---

## 🔍 Explainability (Grad-CAM)

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is applied to the Level-1 malignancy classifier to highlight discriminative regions.

```python
# Last convolutional layer used for Grad-CAM
LAST_CONV_LAYER = "top_conv"
```

### Key Observations
- Malignant predictions → Grad-CAM consistently highlights **central lesion regions**
- Benign predictions → More **diffuse activation** across the image
- Background skin areas are suppressed, reducing spurious correlations
- This confirms robust spatial attention aligned with clinical features

### Clinical Significance
> Grad-CAM overlays enable dermatologists to verify model decisions, increasing **trust and transparency** for real-world deployment.

---

## 🌍 External Evaluation

The hierarchical pipeline was evaluated on **unseen external images (HAM10000 subset)** — different acquisition conditions, illumination, and background.

### External HAM10000 Results (Level-1)

| Metric | Benign | Malignant |
|--------|--------|-----------|
| Precision | 0.99 | 0.92 |
| Recall | 0.91 | **0.99** |
| F1-Score | 0.95 | 0.96 |

**Overall Accuracy: 95%** | **Malignant Recall: 99%**

> High confidence scores on external data confirm **strong generalization** across datasets.

---

## 📁 Project Structure

```
skin-disease-hierarchical-classification/
│
├── skin_disease_hirearchical_classification.ipynb   # Main notebook
│
├── /kaggle/working/
│   ├── labels.csv                      # Dataset with hierarchical labels
│   ├── level1_final_locked.keras       # Level-1 gatekeeper model
│   ├── level2_benign_final.keras       # Level-2 Benign classifier
│   ├── level2_malignant_final.keras    # Level-2 Malignant classifier
│   ├── level1_config.json              # Level-1 threshold config
│   ├── level2_benign_labels.json       # Benign class label map
│   ├── level2_malignant_labels.json    # Malignant class label map
│   └── level2_benign_report.csv        # Classification report
│
└── /content/
    ├── gradcam_external_melanoma.png   # Grad-CAM visualization
    ├── gradcam_external_benign.png
    └── external_level1_confusion_matrix.png
```

---

## ⚙️ Setup & Installation

### Requirements

```bash
pip install tensorflow==2.19.0
pip install scikit-learn numpy pandas matplotlib seaborn
pip install opencv-python h5py
```

### Or run on Kaggle (recommended)
This notebook is designed for **Kaggle with Tesla T4 GPU**.

1. Fork the notebook on Kaggle
2. Enable GPU accelerator (T4 × 2)
3. Add the dataset: `prathvichavan/skin-diesease-dataset-per-class-2-5k-images`
4. Run all cells

---

## 🚀 Usage

### Single Image Inference

```python
result = hierarchical_predict("path/to/your/image.jpg")

# Output:
# {
#   "Level-1": "Malignant",
#   "Level-1 Confidence": 0.9937,
#   "Final Diagnosis": "Melanoma",
#   "Final Confidence": 0.9707
# }
```

### Batch Inference

```python
results = batch_predict("/path/to/image/folder")
# Returns list of predictions for all .jpg/.png/.jpeg images
```

### Grad-CAM Visualization

```python
img_tensor = preprocess_image("path/to/image.jpg")
heatmap = make_gradcam_heatmap(img_tensor, level1_model)
overlay = overlay_gradcam("path/to/image.jpg", heatmap, alpha=0.4)
```

---

## 🔄 Inference Pipeline

```python
def hierarchical_predict(image_path):
    # Step 1: Preprocess
    img = preprocess_single(image_path)          # → (1, 380, 380, 3)

    # Step 2: Level-1 Gate
    p1 = level1_model.predict(img)[0][0]
    
    if p1 < LEVEL1_THRESHOLD:                    # Threshold = 0.3
        # Step 3a: Benign path
        probs = benign_model.predict(img)[0]
        cls = BENIGN_MAP[np.argmax(probs)]
        return "benign", cls, float(np.max(probs))
    else:
        # Step 3b: Malignant path
        p2 = mal_model.predict(img)[0][0]
        cls = "melanoma" if p2 >= 0.5 else "bcc"
        return "malignant", cls, float(p2)
```

---

## 🔮 Future Work

- [ ] **Data Augmentation** — Add CutMix, MixUp, and RandAugment for improved generalization
- [ ] **Ensemble Models** — Combine EfficientNet-B4, B7, and ViT predictions
- [ ] **Attention Mechanisms** — Replace GlobalAveragePooling with CBAM or SE blocks
- [ ] **Multi-label Classification** — Handle overlapping disease presentations
- [ ] **Web App Deployment** — Flask/FastAPI REST API with Grad-CAM overlay
- [ ] **ONNX Export** — For mobile/edge deployment (TFLite / Core ML)
- [ ] **Clinical Trial** — Validate on certified dermatology datasets (ISIC 2020)
- [ ] **Uncertainty Quantification** — Monte Carlo Dropout for prediction confidence bounds

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{skin_hierarchical_2025,
  title   = {Hierarchical Skin Disease Classification using EfficientNet-B4},
  author  = {[Your Name]},
  year    = {2025},
  url     = {https://www.kaggle.com/code/[your-username]/skin-disease-hirearchical-classification},
  note    = {Two-level hierarchical deep learning framework for dermoscopic image analysis}
}
```

### References

1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
2. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
3. Tschandl, P., et al. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific Data.
4. ISIC Archive. *International Skin Imaging Collaboration*. https://www.isic-archive.com

---

## 🧑‍💻 Author

Made with ❤️ for advancing AI-assisted dermatology.

> ⭐ If this project helped you, please give it a star on Kaggle/GitHub!

---

<p align="center">
  <i>This tool is intended for research purposes only and is not a substitute for professional medical advice.</i>
</p>
