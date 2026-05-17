# Manual Checks

This folder contains interactive checks for individual biometric components. They are useful for debugging and demonstrations, but they are not automated pytest tests.

Run all commands from the project root with the virtual environment activated:

```powershell
.\.venv\Scripts\activate
```

## Available Checks

```powershell
python manual_tests/camera_check.py
```

Verifies that OpenCV can access the webcam.

```powershell
python manual_tests/detection_quality_check.py
```

Runs live face detection and face-quality scoring.

```powershell
python manual_tests/eye_openness_live_check.py
```

Runs live eye open/closed detection with the trained TensorFlow model.

```powershell
python manual_tests/gaze_direction_check.py
```

Runs live gaze-direction estimation using L2CS-Net and the trained classifier.

```powershell
python manual_tests/gaze_quality_check.py
```

Runs gaze estimation and face-quality scoring in the same OpenCV window.

```powershell
python manual_tests/face_verification_camera_check.py
```

Opens a small GUI to register and verify users using the webcam.

```powershell
python manual_tests/face_verification_files_check.py
```

Compares two manually selected image files with the face-verification pipeline.

## Closing Windows

Most checks can be closed by pressing `q` or by closing the window. Camera resources are released on exit.

## Runtime Data

Generated users are stored in `data/users/`, which is ignored by Git. Do not commit generated face crops, embeddings, or balance files.
