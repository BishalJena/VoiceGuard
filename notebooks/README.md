# Notebooks Directory

This directory contains Jupyter notebooks for model training and experimentation.

## Main Notebook

**Model_Training.ipynb** - The primary notebook containing:
- Data loading from Figshare and mPower datasets
- Audio preprocessing pipeline (resampling, normalization, silence trimming)
- Hybrid feature extraction using:
  - **Parselmouth (Praat)**: Clinical acoustic features (F0, Jitter, Shimmer, HNR)
  - **Librosa**: Spectral features (MFCCs)
- XGBoost model training and hyperparameter tuning
- Model evaluation with cross-validation
- Export of trained artifacts to `/models/` directory

## How to Use

1. Open the notebook in Google Colab or Jupyter
2. Follow the cells sequentially to:
   - Install dependencies
   - Download datasets
   - Train the model
   - Generate evaluation metrics
3. The notebook will automatically save the three required .pkl files to the `models/` directory

## Requirements

The notebook requires the same dependencies listed in `/requirements.txt` at the root level.
