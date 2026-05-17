from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from l2cs import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_L2CS = ROOT_DIR / "models" / "L2CSNet_gaze360.pkl"
MODEL_CLASSIFIER = ROOT_DIR / "models" / "my_classifier_model_final.pth"
SCALER_PATH = ROOT_DIR / "models" / "scaler.npy"
LABELS_PATH = ROOT_DIR / "models" / "label_encoder_classes.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=None, output_dim=5, dropout=0.25):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        layers = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = hidden_dim

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GazeEstimator:
    def __init__(self, model_path: Path = MODEL_L2CS, device: torch.device = DEVICE, smooth_window: int = 5):
        self.device = device
        self.pipeline = Pipeline(weights=model_path, arch="ResNet50", device=device)

        self.model = DeepClassifier(output_dim=5).to(device)
        self.model.load_state_dict(torch.load(MODEL_CLASSIFIER, map_location=device))
        self.model.eval()

        scaler_data = np.load(SCALER_PATH, allow_pickle=True).item()
        self.mean = scaler_data["mean"]
        self.scale = scaler_data["scale"]
        self.labels = np.load(LABELS_PATH, allow_pickle=True)

        self.history = deque(maxlen=smooth_window)
        self.last_label = None

    def estimate(self, frame):
        try:
            gaze_results = self.pipeline.step(frame)
            if gaze_results.pitch.size == 0 or gaze_results.yaw.size == 0:
                return {"label": self.last_label or "middle"}

            pitch_deg = np.degrees(float(gaze_results.pitch[0]))
            yaw_deg = np.degrees(float(gaze_results.yaw[0]))

            features = np.array([[pitch_deg, yaw_deg]])
            features_scaled = (features - self.mean) / self.scale
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                logits = self.model(features_tensor)
                pred_idx = torch.argmax(logits, dim=1).item()

            label = self.labels[pred_idx]
            self.history.append(label)

            most_common = max(set(self.history), key=self.history.count)
            if self.history.count(most_common) > len(self.history) * 0.7:
                self.last_label = most_common

            return {"label": self.last_label or label}

        except Exception as exc:
            print(f"Error in gaze estimation: {exc}")
            return {"label": self.last_label or "middle"}
