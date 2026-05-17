from typing import Optional

import cv2
import numpy as np


class FaceQualityAssessor:
    def __init__(self, weights: Optional[dict[str, float]] = None, accept_threshold: float = 0.75):
        self.weights = weights or {
            "sharpness": 0.5,
            "brightness": 0.5,
        }
        self.accept_threshold = float(np.clip(accept_threshold, 0.0, 1.0))

        self._lap_norm = 120.0
        self._ten_norm = 20.0
        self._dark_thr = 75
        self._bright_thr = 200
        self._p_low_min = 10
        self._p_high_max = 245

    def _sharpness_score(self, bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()
        lap_score = np.clip(lap_var / self._lap_norm, 0, 1)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = np.sqrt(gx * gx + gy * gy)
        ten_score = np.clip(float(np.mean(tenengrad)) / self._ten_norm, 0, 1)

        return float(np.clip(0.7 * lap_score + 0.3 * ten_score, 0, 1))

    def _brightness_score(self, bgr: np.ndarray) -> float:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        luminance = ycrcb[:, :, 0].astype(np.float32)

        mean_y = float(np.mean(luminance))
        p1 = float(np.percentile(luminance, 1))
        p99 = float(np.percentile(luminance, 99))
        dynamic_range = p99 - p1

        if mean_y < self._dark_thr:
            mean_score = np.clip(mean_y / self._dark_thr, 0, 1)
        elif mean_y > self._bright_thr:
            mean_score = np.clip(1 - (mean_y - self._bright_thr) / (255 - self._bright_thr), 0, 1)
        else:
            mean_score = 1.0

        clip_low = 0.0 if p1 >= self._p_low_min else (self._p_low_min - p1) / self._p_low_min
        clip_high = 0.0 if p99 <= self._p_high_max else (p99 - self._p_high_max) / (255 - self._p_high_max)
        clip_score = np.clip(1.0 - 0.5 * (clip_low + clip_high), 0, 1)
        range_score = np.clip((dynamic_range - 30) / 60.0, 0, 1)

        return float(np.clip(0.6 * mean_score + 0.25 * clip_score + 0.15 * range_score, 0, 1))

    def evaluate(self, image_bgr: np.ndarray) -> dict:
        sharpness = self._sharpness_score(image_bgr)
        brightness = self._brightness_score(image_bgr)

        total = (
            self.weights["sharpness"] * sharpness +
            self.weights["brightness"] * brightness
        )
        total = float(np.clip(total, 0, 1))

        return {
            "scores": {
                "sharpness": round(sharpness, 3),
                "brightness": round(brightness, 3),
            },
            "total": round(total, 3),
            "threshold": self.accept_threshold,
            "decision": "accept" if total >= self.accept_threshold else "reject",
        }
