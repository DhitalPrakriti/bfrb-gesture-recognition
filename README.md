# BFRB Gesture Recognition Using Multi-Sensor Deep Learning

## Overview

This project implements and evaluates multiple deep learning and machine learning models for gesture classification using wearable multi-sensor data. 

The implementation follows the baseline methodology described in the reference paper and extends it with multiple model architectures and fusion strategies.

The goal is to detect **Body-Focused Repetitive Behaviors (BFRBs)** using data collected from:

- Inertial Measurement Unit (IMU)
- Thermopile sensors
- Time-of-Flight (TOF) proximity sensors

The implementation follows the methodology described in **zhang_ryoo_mukherjee_2025.pdf** and reproduces the reported evaluation metrics:

- **Binary F1-score (BFRB detection)**
- **Macro-averaged F1-score (multi-class gesture classification)**

---

## Objectives

1. Perform 18-class gesture classification.
2. Detect BFRB vs non-BFRB behaviors (binary detection).
3. Compare multiple architectures including sensor fusion strategies.
4. Reproduce baseline paper evaluation metrics.
5. Analyze the effect of multi-sensor fusion.

## Project Structure

project_root/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_data_ingestion.ipynb
│ ├── 02_eda_plots.ipynb
│ ├── 03_data_preprocessing.ipynb
│ ├── 04_models_paper.ipynb
│ └── 05_evaluation.ipynb
│
├── plots/
│ ├── example_acc_signal.png
│ ├── gesture_distribution.png
│ ├── sequence_length_distribution.png
│ └── tof_invalid_rate.png
│
├── README.md
├── requirements.txt
└── .gitignore

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

---

## Setup & Installation

### Step 1: Clone the repository
```bash
git clone <repository-url>
cd project_root

### Step 2: Create a virtual environment
python -m venv venv

### Step 3: Activate the environment
Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate

### Step 4: Install dependencies
pip install -r requirements.txt

### Step 5: Verify installation
python --version
Recommended Python version: 3.10 or 3.11


## Notebook Execution Order

The notebooks must be executed in the following order:

1. **01_data_ingestion.ipynb**
   - Loads raw dataset
   - Saves processed pickle files

2. **02_eda_plots.ipynb**
   - Exploratory Data Analysis
   - Sensor distribution visualizations
   - Behavior distribution analysis

3. **03_data_preprocessing.ipynb**
   - Feature selection
   - Sequence filtering
   - Sequence-level train/test split
   - Saves:
     - `train_clean.pkl`
     - `test_clean.pkl`
     - `feature_map.pkl`

4. **04_models_paper.ipynb**
   - Implements six models:
     - Model 1: FFT-MLP (All sensors)
     - Model 2: FFT-MLP (IMU + Thermopile)
     - Model 3: CNN-BiLSTM (TOF)
     - Model 4: Late Fusion Ensemble
     - Model 5: Intermediate Fusion
     - Model 6: FFT + Random Forest
   - Saves prediction outputs for evaluation

5. **05_evaluation.ipynb**
   - Computes:
     - Accuracy
     - Macro-F1 (18-class)
     - Binary F1 (BFRB detection)
     - K+1 Macro-F1
   - Generates comparison table

6. Check the plots/ directory for generated figures

7. Check data/processed/ for saved outputs and evaluation 

All notebooks must be executed sequentially.

---

## Models Implemented

| Model | Description |
|-------|-------------|
| M1 | FFT-MLP using all 332 sensor features |
| M2 | FFT-MLP using IMU + Thermopile |
| M3 | CNN-BiLSTM using TOF sensor data |
| M4 | Late Fusion (weighted ensemble of M2 + M3) |
| M5 | Intermediate Fusion architecture |
| M6 | FFT features + Random Forest |

Normalization is computed using training statistics only to prevent data leakage.
---

## Evaluation Metrics
The project reports the following metrics:
- Accuracy (18-class classification)
- Macro-F1 (18-class)
- Binary F1 (BFRB vs non-BFRB detection)
- Macro-F1 (K+1 grouped behavior)

Binary F1 is the primary behavioral detection metric.

## Final Results Summary
Late Fusion (Model 4) achieved the best overall performance:
- Binary F1 (BFRB detection): ~0.94
- Macro-F1 (18 gestures): ~0.51
- Accuracy: ~0.49

Fusion architectures consistently outperformed single-sensor models.

---

Validation & Generated Outputs

The following plots are automatically generated and saved in /plots:
1. Sensor signal visualization
2. Gesture class distribution
3. Sequence length distribution
4. TOF invalid rate analysis

Evaluation results are saved as:

data/processed/model_comparison_results.csv

Reproducibility
- Random seed fixed to 42
- Sequence-level split prevents data leakage
- Feature metadata saved separately
- Model outputs saved for consistent evaluation

Version Control Workflow
- All development performed in properly named feature branches.
- Pull Requests used for merging into the development branch.
- No direct commits to the main branch.
- Each team member contributed meaningful commits.

Limitations

- Only a single train/test split was used (no cross-validation).
- Hyperparameters were not extensively optimized.
- Training was performed on CPU; GPU optimization not explored.
- Class imbalance handling techniques were not applied.
- Sequence truncation may remove temporal information.
- Random Forest baseline does not model temporal dependencies.
- Real-time deployment performance was not evaluated.

Authors
Capstone Project – Winter 2026
BSIT Program
FDU Vancouver
- Prakriti Dhital 
- Madan Kumar Devkota 
- Xingri Du 
- Son Nguyen 
- Parsa Mokari