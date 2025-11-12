# Models Directory

This directory contains the trained machine learning artifacts required by the Gradio application.

## Required Files

Place the following files in this directory after training your model:

1. **parkinsons_model.pkl** - The trained XGBoost classifier model
2. **scaler.pkl** - The fitted StandardScaler for feature normalization
3. **feature_names.pkl** - List of feature names used during training

## How to Generate These Files

Run the `Model_Training.ipynb` notebook in the `/notebooks/` directory. The notebook will:
- Load and preprocess the audio datasets
- Extract hybrid features (Parselmouth + Librosa)
- Train the XGBoost model
- Save all three .pkl files to this directory

## Note

The Gradio app (`app.py`) will not run without these files. Ensure you complete the training process before launching the application.
