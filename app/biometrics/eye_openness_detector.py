import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class EyeOpennessDetector:
    def __init__(self, model_path: str, img_size=(90, 90), scale=1.8, threshold=0.5):
        self.img_size = img_size
        self.scale = scale
        self.threshold = threshold
        self._call_via_predict = False

        try:
            self.model = load_model(model_path, compile=False)
            self._call_via_predict = hasattr(self.model, "predict")
            print(f"[INFO] Loaded Keras model from {model_path}")
        except Exception as exc:
            print(f"[WARN] Keras loading failed ({exc}). Trying TensorFlow SavedModel.")
            self.model = tf.saved_model.load(model_path)
            print(f"[INFO] Loaded TensorFlow SavedModel from {model_path}")

        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

    def _crop_eye(self, image, landmarks, eye="left"):
        indexes = self.left_eye_idx if eye == "left" else self.right_eye_idx
        points = np.array([(landmarks[index][0], landmarks[index][1]) for index in indexes])
        x, y, width, height = cv2.boundingRect(points)
        center_x = x + width // 2
        center_y = y + height // 2

        scaled_width = int(width * self.scale)
        scaled_height = int(height * self.scale)

        x1 = max(center_x - scaled_width // 2, 0)
        y1 = max(center_y - scaled_height // 2, 0)
        x2 = min(center_x + scaled_width // 2, image.shape[1])
        y2 = min(center_y + scaled_height // 2, image.shape[0])

        return image[y1:y2, x1:x2]

    def _preprocess(self, eye_crop):
        eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        eye_resized = cv2.resize(eye_rgb, self.img_size)
        return np.expand_dims(eye_resized.astype(np.float32), axis=0)

    def _infer_prob(self, input_tensor):
        if self._call_via_predict:
            output = self.model.predict(input_tensor, verbose=0)
        elif hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
            infer = self.model.signatures["serving_default"]
            outputs = infer(tf.constant(input_tensor))
            first_key = next(iter(outputs))
            output = outputs[first_key].numpy()
        else:
            raise RuntimeError("SavedModel does not expose a 'serving_default' signature.")

        return float(np.array(output).squeeze())

    def predict_eyes(self, image_bgr, landmarks):
        results = {}
        for eye in ["left", "right"]:
            crop = self._crop_eye(image_bgr, landmarks, eye=eye)
            if crop.size == 0:
                results[eye] = {"prob_open": None, "state": "undetected"}
                continue

            prob_open = self._infer_prob(self._preprocess(crop))
            state = "open" if prob_open >= self.threshold else "closed"
            results[eye] = {"prob_open": prob_open, "state": state, "crop": crop}

        return results
