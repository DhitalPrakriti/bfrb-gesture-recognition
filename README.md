# BFRB Gesture Recognition Using Multi-Sensor Deep Learning
AI/ML Capstone Project — Wearable Sensor Classification with Transformers and BiLSTM

### Overview

This project implements and evaluates multiple deep learning and machine learning models for gesture classification using wearable multi-sensor data. The implementation follows the baseline methodology described in the reference paper and extends it with multiple model architectures, fusion strategies, and novel NLP-based approaches.

The goal is to detect Body-Focused Repetitive Behaviors (BFRBs) using data collected from:

- **Inertial Measurement Unit (IMU)**
- **Thermopile sensors**
- **Time-of-Flight (TOF) proximity sensors**

The implementation reproduces evaluation metrics from zhang_ryoo_mukherjee_2025.pdf and introduces novel NLP-based models for comparison:

- Binary F1-score (BFRB detection)
- Macro-averaged F1-score (multi-class gesture classification)

---

## Objectives

- Perform 18-class gesture classification.
- Detect BFRB vs. non-BFRB behaviors (binary detection).
- Compare multiple architectures including sensor fusion strategies.
- Reproduce baseline paper evaluation metrics.
- Analyze the effect of multi-sensor fusion.
- Design and evaluate novel NLP-based BFRB detection models (Milestone 2).
- Compare NLP models against paper baselines using binary F1 and macro-averaged F1.

---

## Project Structure

```
project_root/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda_plots.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_models_paper.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_nlp_datapreprocessing.ipynb
│   ├── 07_model_nlp_transformer.ipynb
│   ├── 08_model_nlp_bilstm.ipynb
│   ├── 09_evaluation_comparison.ipynb
│   └── 10_plot_generation.ipynb
│
├── results/
│   ├── transformer_results.pkl      # auto-generated after running notebooks 07–09
│   ├── bilstm_results.pkl
│   └── all_model_comparison.csv
│
├── plots/
│   ├── example_acc_signal.png
│   ├── gesture_distribution.png
│   ├── sequence_length_distribution.png
│   ├── tof_invalid_rate.png
│   ├── comparison_binary_f1.png
│   ├── comparison_macro_f1.png
│   └── nlp_best_confusion_matrix.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## System Requirements

To run this project, the following system specifications are recommended:

1. Operating System: Windows 10+, macOS, or Linux  
2. Python Version: 3.10 or 3.11  
3. RAM: Minimum 8 GB (16 GB recommended for model training)  
4. Disk Space: At least 2 GB available  
5. CPU: Multi-core processor recommended  
6. GPU: Optional (CUDA-supported GPU for faster training)  
7. Virtual environment support (venv or Conda)  

**Key Python Dependencies:**

```
numpy==1.23.5
pandas==2.0.0
matplotlib==3.7.1
torch==2.1.0
scikit-learn==1.3.0
pickle (standard library)
pathlib (standard library)
```

> See `requirements.txt` for the full list of pinned dependencies.

---

## Setup Instructions

**Step 1: Clone the repository** 
```bash
git clone <repository-url>
cd project_root
```

**Step 2: Create a virtual environment**
```bash
python -m venv venv
```

**Step 3: Activate the environment** 
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Step 4: Install dependencies** 
```bash
pip install -r requirements.txt
```

**Step 5: Verify Python version**
```bash
python --version
Recommended: Python 3.10.6 or 3.11
```

**Step 6: Place raw data files** 

Ensure the following CSV files are placed in `data/raw/` before running any notebooks:
- `train.csv`
- `train_demographics.csv`
- `test.csv`
- `test_demographics.csv`

---

## Reproducing Results — Notebook Execution Order

All notebooks must be executed **sequentially** in the order listed below. Each notebook reads from files saved by the previous one; do not skip steps.

---

### Milestone 1 Notebooks

#### `01_data_ingestion.ipynb`
Loads the four raw CSV files from `data/raw). Serializes as pickle files to `data/processed/` for downstream use. No transformations are applied here.

**Saves:** `train.pkl`, `train_demographics.pkl`, `test.pkl`, `test_demographics.pkl`

---

#### `02_eda_plots.ipynb`
   - Exploratory Data Analysis
   - Sensor distribution visualizations
   - Behavior distribution analysis

**Plots saved to** `plots/`:
- `example_acc_signal.png`
- `gesture_distribution.png`
- `sequence_length_distribution.png`
- `tof_invalid_rate.png`

---

#### `03_data_preprocessing.ipynb`
   - Feature selection
   - Sequence filtering
   - Sequence-level train/test split

**Saves:** `train_clean.pkl`, `test_clean.pkl`, `feature_map.pkl`

---

#### `04_models_paper.ipynb`
 - Implements six models:
     - Model 1: FFT-MLP (All sensors)
     - Model 2: FFT-MLP (IMU + Thermopile)
     - Model 3: CNN-BiLSTM (TOF)
     - Model 4: Late Fusion Ensemble
     - Model 5: Intermediate Fusion
     - Model 6: FFT + Random Forest

**Saves:** `model1_fft_mlp_all_outputs.pt`, `model2_fft_mlp_imu_thm_outputs.pt`, `model3_cnn_bilstm_tof_outputs.pt`, `model4_late_fusion_outputs.pt`, `model5_intermediate_fusion_outputs.pt`, `model6_fft_random_forest_outputs.pt`

---

#### `05_evaluation.ipynb`
   - Computes:
     - Accuracy
     - Macro-F1 (18-class)
     - Binary F1 (BFRB detection)
     - K+1 Macro-F1
   - Generates comparison table

**Saves:** `data/processed/model_comparison_results.csv`

---

### Milestone 2 Notebooks (NLP)

#### `06_nlp_datapreprocessing.ipynb`
 - Converts continuous sensor readings into discrete text tokens for NLP model consumption
 - Each sensor value is binned into `low`, `medium`, or `high`.
 -  Builds a vocabulary with `<PAD>` and `<UNK>` special tokens and converts all sequences to integer ID lists.

**Saves:** `train_text_seq.pkl`, `test_text_seq.pkl`, `nlp_vocab.pkl`, `nlp_feature_thresholds.pkl`, `nlp_max_seq_len.pkl`

---

#### `07_model_nlp_transformer.ipynb`
- Implements a Transformer-based binary sequence classifier for BFRB detection. 
- Architecture: token embedding (dim=64) → sinusoidal positional encoding (Vaswani et al., 2017) → 2-layer Transformer encoder (4 attention heads, FF dim=128) → mean pooling over non-padding tokens → linear classifier. 
- Sequences are truncated to 512 tokens.
- Trained for 10 epochs using Adam (lr=0.001) with cross-entropy loss.

**Results:** Binary F1 = 90.93% | Macro F1 = 85.98%

**Saves:** `results/transformer_results.pkl`

---

#### `08_model_nlp_bilstm.ipynb`
- Implements a Bidirectional LSTM binary classifier operating on the same tokenized sequences. - Architecture: token embedding (dim=64) → 2-layer BiLSTM (hidden dim=128 per direction, 256 after concatenation) → mean pooling over non-padding timesteps → linear classifier. 
- Uses `pack_padded_sequence` to efficiently handle variable-length inputs.
- Trained for 10 epochs using Adam (lr=0.001) with cross-entropy loss and dropout (0.3).

**Results:** Binary F1 = 89.98% | Macro F1 = 84.19%

**Saves:** `results/bilstm_results.pkl`

---

#### `09_evaluation_comparison.ipynb`
- Loads all 8 model outputs (6 paper models + 2 NLP models)
- Computes binary F1 and macro-averaged F1 for each using a unified evaluation function. - Assembles and prints a full comparison table.

**Saves:** `results/all_model_comparison.csv`

---

#### `10_plot_generation.ipynb`
Reads the comparison CSV and NLP result files to produce three plots saved to `plots/`:

| Plot | Description |
|------|-------------|
| `comparison_binary_f1.png` | Bar chart comparing Binary F1-score across all 8 models |
| `comparison_macro_f1.png` | Bar chart comparing Macro-averaged F1-score across all 8 models |
| `nlp_best_confusion_matrix.png` | Confusion matrix for the best NLP model (Transformer) on binary BFRB classification |

---

## Models Implemented

### Milestone 1 — Paper Baseline Models

| Model | Description | Binary F1 | Macro F1 |
|-------|-------------|-----------|----------|
| M1 FFT-MLP (All Sensors) | FFT features from all 332 sensors fed to MLP | 90.93% | 46.12% |
| M2 FFT-MLP (IMU+THM) | FFT features from IMU + Thermopile only | 90.86% | 41.07% |
| M3 CNN-BiLSTM (TOF) | CNN-BiLSTM operating on raw TOF sequences | 91.60% | 29.50% |
| M4 Late Fusion | Weighted ensemble of M2 + M3 predictions | **94.28%** | 50.14% |
| M5 Intermediate Fusion | Shared feature fusion architecture | 91.11% | 39.61% |
| M6 FFT + Random Forest | FFT features with classical Random Forest | 89.81% | 38.80% |

### Milestone 2 — Novel NLP Models

| Model | Description | Binary F1 | Macro F1 |
|-------|-------------|-----------|----------|
| Transformer (NLP) | Token embedding + positional encoding + Transformer encoder | 90.93% | **85.98%** |
| BiLSTM (NLP) | Token embedding + 2-layer Bidirectional LSTM | 89.98% | 84.19% |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Binary F1** | F1-score for BFRB vs. non-BFRB binary detection (primary metric) |
| **Macro-F1 (18-class)** | Macro-averaged F1 over all 18 gesture classes |
| Accuracy | Raw classification accuracy over 18 classes |
| Macro-F1 (K+1) | Macro F1 with BFRB gestures as individual classes + merged non-BFRB |

---

## Final Results Summary

**Best Binary F1:** Late Fusion (M4) — **94.28%**  
**Best Macro F1:** Transformer (NLP) — **85.98%**

The NLP models achieve dramatically higher macro-averaged F1 scores (~85–86%) compared to the paper baselines (~30–50%), while remaining competitive on binary BFRB detection (~90%).

---

## Reproducibility

- Random seed fixed to **42** throughout all notebooks
- Sequence-level train/test split prevents data leakage across all models
- NLP bin thresholds computed from training data only
- All model outputs saved as files for consistent downstream evaluation
- Feature metadata and vocabulary saved separately for transparency.

---

## Limitations

### Milestone 1 — Paper Baseline Limitations

- Only a single train/test split was used (no cross-validation).
- Hyperparameters were not extensively optimized.
- Training was performed on CPU; GPU optimization not explored.
- Class imbalance handling techniques were not applied.
- Sequence truncation may remove temporal information.
- Random Forest baseline does not model temporal dependencies.
- Real-time deployment performance was not evaluated.

### Milestone 2 — NLP Model Limitations

- Only binary classification (BFRB vs. non-BFRB) is performed; 18-class gesture identification is not supported.
- Sequences are truncated to 512 tokens, removing the majority of temporal signal for longer gestures.
- TOF sensor data (320 channels) is excluded from NLP models, losing spatial proximity information.
- Continuous sensor values are binned into only three discrete levels (low, medium, high), discarding fine-grained signal variation.
- Only a single train/test split was used (no cross-validation).
- Hyperparameters were not extensively optimized; fixed settings were used for both models.
- Macro-averaged F1 is computed over 2 classes only, making it not directly comparable to the paper's 18-class macro F1.

---

## Version Control Workflow

- All development performed in properly named feature branches: `<FirstInitial><LastName>_<FeatureName>`
- Pull Requests used for merging feature branches into the `dev` branch
- No direct commits to the `main` branch
- `dev` branch merged into `main` at the end of each milestone
- Each team member contributed meaningful commits with clear messages

---

## Authors

Capstone Project – Winter 2026
BSIT Program
FDU Vancouver
- Prakriti Dhital
- Madan Kumar Devkota
- Parsa Mokari 
- Son Nguyen 
- Xingrui Du 
