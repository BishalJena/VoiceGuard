import gradio as gr
import joblib
import numpy as np
import librosa
import parselmouth
import shap
import matplotlib.pyplot as plt
import os

# --- 1. Load Artifacts ONCE at startup ---
# Allow launching UI without models via env flag
UI_ONLY = os.getenv("UI_ONLY", "0") == "1"
# Ensure the 'models' directory is in the same folder as app.py
MODEL_PATH = "models/parkinsons_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
except FileNotFoundError:
    print("FATAL ERROR: Model/Scaler/Features.pkl files not found.")
    print("Please run the ML Training notebook first and place artifacts in 'models/'")
    # You can add a gr.Error() here if in a running app
    model, scaler, feature_names = None, None, None

# Initialize SHAP Explainer (if model loaded)
# We use TreeExplainer for XGBoost
if model:
    explainer = shap.TreeExplainer(model)
else:
    explainer = None

# --- 2. Helper Functions (Paste from ML Notebook) ---
# These functions MUST be identical to the ones used in training.

def preprocess_audio(y, sr, target_sr=22050, top_db=30):
    """
    1. Resamples, 2. Normalizes, 3. Trims silence from audio.
    """
    try:
        y_resampled = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    except Exception as e:
        y_resampled = y

    y_normalized = librosa.util.normalize(y_resampled)
    y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=top_db)
    return y_trimmed, sr

def extract_all_features(y, sr):
    """
    Extracts a hybrid feature set using Parselmouth and Librosa.
    """
    try:
        sound = parselmouth.Sound(y, sr)
        pitch = sound.to_pitch()
        f0_mean = pitch.get_mean(unit="Hertz")
        f0_min = pitch.get_minimum(unit="Hertz")
        f0_max = pitch.get_maximum(unit="Hertz")

        harmonicity = sound.to_harmonicity_cc()
        hnr = harmonicity.get_mean()

        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)",
                                               f0_min, f0_max)

        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)",
                                              0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_abs = parselmouth.praat.call(point_process, "Get jitter (local, absolute)",
                                           0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)",
                                           0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_ppq = parselmouth.praat.call(point_process, "Get jitter (ppq5)",
                                           0.0, 0.0, 0.0001, 0.02, 1.3)

        shimmer_local = parselmouth.praat.call([sound, point_process], "Get shimmer (local)",
                                               0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = parselmouth.praat.call([sound, point_process], "Get shimmer (local_dB)",
                                            0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)",
                                             0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq5)",
                                             0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

        parselmouth_features = [
            f0_mean, f0_min, f0_max, hnr,
            jitter_local, jitter_abs, jitter_rap, jitter_ppq,
            shimmer_local, shimmer_db, shimmer_apq3, shimmer_apq5
        ]
    except Exception as e:
        parselmouth_features = [np.nan] * 12

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    all_features = np.concatenate((parselmouth_features, mfccs_mean))
    all_features = np.nan_to_num(all_features, nan=0.0)

    # Ensure feature count matches what model was trained on
    if len(all_features) != len(feature_names):
        print(f"Warning: Feature mismatch. Expected {len(feature_names)}, got {len(all_features)}")
        # Handle error gracefully - this should not happen if notebooks are identical
        return None

    return all_features

# --- 3. The Main Prediction Function ---
def predict_parkinsons(audio):
    """
    This function is the core of the app.
    It fulfills all 5 MVP requirements in one flow.
    """
    if audio is None:
        return "Please record audio.", None, None

    if model is None:
        return "ERROR: Model is not loaded.", None, None

    # REQ 1: Get audio from microphone
    # Gradio provides (sample_rate, numpy_array)
    sr, y = audio

    # REQ 2: Preprocess Audio
    y_processed, sr_processed = preprocess_audio(y, sr)

    if len(y_processed) == 0:
        return "Audio is silent. Please record a clear 'ahhh' sound.", None, None

    # REQ 3: Extract Features
    features_vector = extract_all_features(y_processed, sr_processed)

    if features_vector is None:
         return "Error in feature extraction. Please try again.", None, None

    features_reshaped = features_vector.reshape(1, -1)

    # Scale features using the *loaded* scaler
    features_scaled = scaler.transform(features_reshaped)

    # REQ 4: Predict Risk
    # Get the probability of class '1' (Parkinson's)
    probability = model.predict_proba(features_scaled)[0][1]

    # REQ 5: Display Live Result (User-Friendly)
    risk_score_text = f"{probability * 100:.2f}% Risk Score"

    if probability < 0.3:
        risk_level = "Risk Level: Low"
    elif probability < 0.7:
        risk_level = "Risk Level: Medium"
    else:
        risk_level = "Risk Level: High"

    # --- BONUS: Generate SHAP Plot ---

    # Get SHAP values for the positive class
    shap_values = explainer.shap_values(features_scaled)

    # Create a Matplotlib figure for the force plot
    # The 'matplotlib=True' argument is the key
    plt.ioff()  # Turn off interactive plotting
    fig = shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        features_scaled[0, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        text_rotation=15
    )
    plt.tight_layout()  # Ensure it fits

    # Return all 3 outputs for the Gradio UI
    return risk_level, risk_score_text, fig

# --- 4. Define the Gradio UI ---
# Use gr.Blocks for a clean, custom layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Hide built-in Trim and Reset buttons inside Audio editor
    gr.HTML(
        """
        <style>
        button[aria-label="Trim audio"] { display: none !important; }
        button[aria-label="Reset audio"] { display: none !important; }
        </style>
        """
    )
    gr.Markdown("# VoiceGuard-X: Explainable Parkinson's Risk")
    gr.Markdown(
        "**Instructions:** Find a quiet place and record yourself saying 'ahhhhh' "
        "in a steady tone for 3-5 seconds. Then, press 'Analyze Voice'."
    )

    with gr.Row():
        # --- REQ 1: Record Voice ---
        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy",  # Passes (sr, y) tuple, which is fast
            label="Record Your Voice",
            max_length=10  # Cap recordings at 10 seconds
            ,elem_id="mic_recorder"
        )

    with gr.Row():
        analyze_button = gr.Button("Analyze Voice")
        reset_prev_button = gr.Button("Reset to Previous")

    gr.Markdown("---")
    gr.Markdown("### Your Results")

    # --- REQ 5: Display Results ---
    with gr.Row():
        label_output = gr.Label(label="Risk Level")
        score_output = gr.Textbox(label="Risk Score")

    gr.Markdown("### Risk Factor Analysis (Bonus Feature)")
    gr.Markdown(
        "This plot shows *why* the model made its decision. "
        "Features in **red** pushed your risk score *higher*. "
        "Features in **blue** pushed it *lower*."
    )
    # The gr.Plot component can directly display matplotlib figures
    plot_output = gr.Plot(label="Feature Contribution")

    # 0) Client-only session memory of current and previous takes (no server storage)
    # Also: Auto-stop recording at 10s using a scoped DOM timer
    audio_input.start_recording(
        None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            // Clear any existing timer before starting a new one
            if (window._vg_rec_stop_timer) {
                clearTimeout(window._vg_rec_stop_timer);
                window._vg_rec_stop_timer = null;
            }
            // After 10s, click Stop within the audio component
            window._vg_rec_stop_timer = setTimeout(() => {
                const root = document.querySelector('#mic_recorder');
                if (!root) return;
                let stopBtn = root.querySelector('button[aria-label*="Stop" i], button[aria-label*="stop" i]');
                if (!stopBtn) {
                    stopBtn = Array.from(root.querySelectorAll('button'))
                        .find(b => /\\bstop\\b/i.test(b.textContent || '') || /\\bstop\\b/i.test((b.getAttribute('aria-label')||'')));
                }
                if (stopBtn) stopBtn.click();
            }, 10000);
        }
        """
    )
    audio_input.stop_recording(
        None,
        inputs=audio_input,
        outputs=None,
        js="""
        (current) => {
            // Clear any running auto-stop timer
            if (window._vg_rec_stop_timer) {
                clearTimeout(window._vg_rec_stop_timer);
                window._vg_rec_stop_timer = null;
            }
            // Shift the last 'current' into 'prev', then set new 'current'
            if (window._vg_current) {
                window._vg_prev = window._vg_current;
            }
            window._vg_current = current;
        }
        """
    )
    # 1) When recording stops, run prediction automatically
    audio_input.stop_recording(
        fn=predict_parkinsons,
        inputs=audio_input,
        outputs=[label_output, score_output, plot_output]
    )
    # 2) Make "Analyze Voice" stop the recording first (then the above handler runs)
    analyze_button.click(
        None,
        None,
        None,
        js="""
        () => {
            // Only interact within our audio component to avoid clicking any
            // unrelated "Stop" buttons from extensions or overlays.
            const root = document.querySelector('#mic_recorder');
            if (!root) return;
            // Prefer aria-label containing "Stop"
            let stopBtn = root.querySelector('button[aria-label*="Stop" i], button[aria-label*="stop" i]');
            // Fallback: find by visible text within the audio component
            if (!stopBtn) {
                stopBtn = Array.from(root.querySelectorAll('button'))
                    .find(b => /\\bstop\\b/i.test(b.textContent || '') || /\\bstop\\b/i.test((b.getAttribute('aria-label')||'')));
            }
            if (stopBtn) stopBtn.click();
        }
        """
    )
    # 3) Custom "Reset to Previous" button restores only the immediately prior take
    reset_prev_button.click(
        None,
        None,
        audio_input,
        js="""
        () => {
            // Return previous recording if present; otherwise do nothing
            return window._vg_prev || null;
        }
        """
    )

# --- 5. Launch the App ---
if __name__ == "__main__":
    if model is None and not UI_ONLY:
        print("Cannot launch app: Model artifacts not found.")
        print("Tip: Set UI_ONLY=1 to launch the UI without models for layout testing.")
    else:
        if model is None and UI_ONLY:
            print("Launching UI-Only mode (no models loaded). Predictions will be disabled.")
        else:
            print("Launching VoiceGuard-X App...")
        demo.launch()
