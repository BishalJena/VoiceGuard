# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning as far as is practical for a hackathon project.

## [0.2.0] - 2025-11-11
### Added
- UI-only launch mode via `UI_ONLY=1` for layout testing without model artifacts.
- Client-side session memory for audio takes (`window._vg_current`, `window._vg_prev`) to support a "Reset to Previous" flow without any server storage.
- Auto-stop for microphone recording at 10 seconds via a scoped client-side timer.

### Changed
- Analyze button now stops the microphone recording automatically and then triggers prediction.
- Audio capture is capped at `10s` using `gr.Audio(max_length=10)`.
- Built-in Trim and Reset controls of the Gradio audio editor are hidden; replaced with a dedicated "Reset to Previous" button that restores only the most recent prior recording for the current session.
- Scoped the Analyze button’s DOM interaction to the audio component (`elem_id="mic_recorder"`) to avoid clicking unrelated buttons on the page.

### Fixed
- Prevented browser/extension screen-share prompts caused by overly broad "Stop" button clicks.
- Eliminated "Audio is too long, and must be at most 10 seconds" errors by auto-stopping the recording at the limit.

## [0.1.0] - 2025-11-11
### Added
- Initial Gradio app with microphone recording, preprocessing, feature extraction, model prediction, and SHAP visualization.


