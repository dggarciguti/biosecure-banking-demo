# BioSecure Banking Demo

Desktop prototype of a secure banking workflow that combines facial authentication with active biometric checks based on gaze direction and eye openness.

The training code and model-development notebooks are intentionally kept out of this repository and can be published separately. This repo focuses on the final runnable application: a Tkinter banking simulation, trained model artifacts, reusable biometric modules, and manual checks for validating each computer-vision component.

## What This Project Demonstrates

The application simulates a banking flow where sensitive operations require more than a simple login:

1. A user registers with a face capture.
2. The system stores a facial embedding for that user.
3. The user logs in through facial verification.
4. The user starts a transfer to another registered user.
5. Before approving the transfer, the app asks for an active biometric challenge:
   - look in specific directions.
   - open/close specific eyes.
6. The transfer is completed only if the challenge is passed.

The goal is to demonstrate a practical biometric security workflow that combines identity verification with liveness-style interaction.

## Key Features

- **Facial registration** with aligned face crops and DeepFace/ArcFace embeddings.
- **Facial login** through 1:1 user verification.
- **Face quality assessment** based on sharpness and lighting.
- **Gaze estimation** using L2CS-Net plus a trained direction classifier.
- **Eye openness detection** using a trained TensorFlow model.
- **Active transaction verification** before money transfers.
- **Local audio prompts** for gaze and eye-openness challenges.
- **Manual diagnostic checks** for camera, face detection, gaze, eye openness, and verification.
- **Runtime user data isolation** through `data/users/`, ignored by Git.

## Tech Stack

- Python 3.11
- Tkinter
- OpenCV
- MediaPipe
- DeepFace / ArcFace
- TensorFlow
- PyTorch
- L2CS-Net
- NumPy / SciPy
- Pillow
- Pygame

## Project Structure

```text
app/
  main.py                 # Application entry point
  gui.py                  # Tkinter user interface
  logic.py                # Registration, login, balances, and transfers
  camera_utils.py         # Camera feed helper for the GUI
  eye_test.py             # Eye-openness transaction challenge
  gaze_test.py            # Gaze-direction transaction challenge
  audios/                 # Local audio prompts
  biometrics/
    alignment.py          # Face alignment
    detection.py          # Face detection wrapper
    eye_openness_detector.py
    face_verification.py
    gaze_estimator.py
    landmarks.py
    quality_assessment.py

data/
  users/                  # Runtime user data; ignored by Git

manual_tests/             # Interactive component checks

models/                   # Trained inference artifacts

requirements.txt
```

## Model Artifacts

The application expects these files under `models/`:

```text
models/
  L2CSNet_gaze360.pkl
  my_classifier_model_final.pth
  scaler.npy
  label_encoder_classes.npy
  best_eye_detection_model_tf/
    saved_model.pb
    fingerprint.pb
    variables/
```

Model roles:

- `L2CSNet_gaze360.pkl`: base gaze-estimation model.
- `my_classifier_model_final.pth`: trained classifier for gaze direction labels.
- `scaler.npy`: feature normalization values for the gaze classifier.
- `label_encoder_classes.npy`: class labels for the gaze classifier.
- `best_eye_detection_model_tf/`: TensorFlow SavedModel for eye open/closed inference.


## Installation

Python 3.11 is recommended.

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The `l2cs` dependency is installed from GitHub. If installation fails, make sure Git is installed and available in your terminal:

```powershell
git --version
```

## Running the Application

From the project root:

```powershell
.\.venv\Scripts\activate
python -m app.main
```

The app requires webcam access.

## Typical Demo Flow

1. Open the application.
2. Register a first user and assign initial funds.
3. Register a second user as the transfer recipient.
4. Log in as the first user with facial verification.
5. Start a transfer to the second user.
6. Select either:
   - gaze challenge; or
   - eye-openness challenge.
7. Complete the requested actions.
8. Confirm that the transfer is approved and balances are updated.

## Manual Checks

Manual checks are interactive scripts for validating individual components without running the full app.

```powershell
python manual_tests/camera_check.py
python manual_tests/detection_quality_check.py
python manual_tests/eye_openness_live_check.py
python manual_tests/gaze_direction_check.py
python manual_tests/gaze_quality_check.py
python manual_tests/face_verification_camera_check.py
python manual_tests/face_verification_files_check.py
```

See [manual_tests/README.md](manual_tests/README.md) for more details.

## Runtime Data and Privacy

Generated users are stored in:

```text
data/users/
```

This folder may contain:

- face crops;
- face embeddings;
- user balance JSON files.

Do not commit generated biometric data. The folder is ignored by Git except for `.gitkeep`.

## Licensing and Third-Party Models

This repository contains application code plus trained inference artifacts. Before reusing the project commercially, review the licenses and terms of the third-party models, libraries, and datasets involved.

Relevant third-party components include:

- L2CS-Net for gaze estimation.
- DeepFace and its wrapped face-recognition models, including ArcFace.
- MediaPipe face detection and face mesh.
- TensorFlow, PyTorch, OpenCV, and other Python dependencies.

The included custom models and preprocessing artifacts are intended for demonstration and portfolio use. If you redistribute pretrained weights from third-party projects, keep the original attribution and verify that their license permits redistribution.

## Citation / Attribution

If you use this project or its model components in a public context, consider citing or linking to the underlying projects and papers:

- L2CS-Net: gaze estimation model used as the base gaze pipeline.
- Gaze360: dataset/model family associated with unconstrained gaze estimation.
- DeepFace: face-recognition framework used for facial embeddings and verification.
