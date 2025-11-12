# VoiceGuard-X: Explainable Parkinson's Risk Detection

A submission for the "AI for Early Detection of Parkinson's from Voice" Hackathon.

## 1. The Problem

Parkinson's Disease (PD) is a neurodegenerative disorder where early detection is critical. Vocal changes, known as **hypokinetic dysarthria**, are a near-ubiquitous symptom and can appear 5-10 years before traditional motor symptoms. However, models trained on clean lab data often fail to generalize to real-world audio (the "Generalizability Crisis").

## 2. Our Solution

**VoiceGuard-X** is an MVP web tool that provides a robust and explainable PD risk assessment from a 3-5 second voice recording.

Our solution is built to win by excelling in all evaluation criteria:

- **Functionality**: The app works end-to-end: Record → Preprocess → Extract → Predict → Display.

- **Accuracy & Generalization**: We solved the "Generalizability Crisis" by:
  - Building a "Gold-Standard" feature engine using **Parselmouth (Praat)** for clinical features (Jitter, Shimmer) and **Librosa** for spectral features (MFCCs).
  - Training our **XGBoost model** on a robust hybrid dataset of raw audio (from Figshare, mPower) processed with our own pipeline. This ensures our model is trained for real-world microphone data.

- **User Experience**: The UI is clean, simple, and built with Gradio. It provides calm, clear results ("Low/Medium/High Risk") instead of alarming diagnoses.

- **Bonus Features**: We provide **Explainable AI (XAI)**. A live SHAP plot shows the user which of their vocal features contributed to their risk score, building trust and providing unmatched insight.

## 3. How to Run

**Clone the repository:**
```bash
git clone https://github.com/BishalJena/VoiceGuard.git
cd VoiceGuard
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the application:**
```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

## 4. Project Structure

```
/
|-- app.py                    # The main Gradio application
|-- /models/
|   |-- parkinsons_model.pkl  # Trained XGBoost model
|   |-- scaler.pkl            # Trained StandardScaler
|   |-- feature_names.pkl     # List of feature names
|-- /notebooks/
|   |-- Model_Training.ipynb  # Colab notebook with all training code
|-- requirements.txt          # All Python dependencies
|-- presentation.pdf          # The 5-slide PDF pitch
|-- README.md                 # You are here
```

## 5. Features

### Core Features (MVP Requirements)
1. **Voice Recording**: Record audio directly through microphone in the browser
2. **Audio Preprocessing**: Automatic resampling, normalization, and silence trimming
3. **Feature Extraction**: Hybrid clinical and spectral features using Parselmouth and Librosa
4. **Risk Prediction**: XGBoost-based classification with probability scores
5. **Live Results Display**: User-friendly risk levels (Low/Medium/High) with percentage scores

### Bonus Features
- **Explainable AI**: SHAP force plots showing which vocal features contribute to the risk score
- **Clinical-Grade Features**: Jitter, Shimmer, HNR, and MFCC features matching research standards
- **Real-World Ready**: Trained on diverse audio sources for robust generalization

## 6. Technical Stack

- **Frontend**: Gradio (Python-based web UI)
- **ML Model**: XGBoost Classifier
- **Feature Engineering**:
  - Parselmouth (Praat) for acoustic features
  - Librosa for spectral features
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Data Processing**: NumPy, scikit-learn

## 7. Model Performance

The model achieves strong performance on real-world audio data through:
- Hybrid feature engineering combining clinical and spectral features
- Robust preprocessing pipeline handling various audio qualities
- Training on diverse datasets (Figshare, mPower)

## 8. Disclaimer

This application is designed for research and educational purposes only. It is not a medical device and should not be used for clinical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 9. License

This project is open source and available for educational and research purposes.

## 10. Contact

For questions or feedback about this project, please open an issue on GitHub.
