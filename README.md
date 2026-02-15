# BFRB Gesture Recognition Using Multi-Sensor Deep Learning

## Overview

This project implements and evaluates multiple deep learning and machine learning models for gesture classification using wearable multi-sensor data. The goal is to detect Body-Focused Repetitive Behaviors (BFRBs) using IMU, thermopile, and Time-of-Flight (TOF) sensors.

The implementation follows the baseline methodology described in the reference paper and extends it with multiple model architectures and fusion strategies.

---

## Project Structure

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
│   └── 05_evaluation.ipynb
│
├── plots/
│ ├── (generated figures)
├
├── README.md
├── requirements.txt
└── .gitignore

---

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

---

## Evaluation Metrics

- Accuracy (18-class classification)
- Macro-F1 (18-class)
- Binary F1 (BFRB vs non-BFRB)
- Macro-F1 (K+1 grouped behavior)

Late Fusion achieved the best overall performance.

---

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
Recommended Python version: 3.10 or 3.11

Reproducibility
- Random seed fixed to 42
- Sequence-level split prevents data leakage
- Feature metadata saved separately
- Model outputs saved for consistent evaluation
