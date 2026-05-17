import os
from typing import Optional, Union

import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance


def compute_embedding_from_aligned_face(face_bgr: np.ndarray, model_name: str = "ArcFace") -> Optional[np.ndarray]:
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    try:
        representations = DeepFace.represent(
            img_path=face_rgb,
            model_name=model_name,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
    except Exception as exc:
        print(f"[ERROR] Could not compute face embedding: {exc}")
        return None

    if not representations:
        return None
    return np.array(representations[0]["embedding"], dtype="float32")


class FaceVerifier:
    MODEL_THRESHOLDS = {
        "ArcFace": 0.68,
        "Facenet": 0.40,
        "VGG-Face": 0.68,
        "OpenFace": 0.10,
        "DeepFace": 0.23,
    }

    def __init__(
        self,
        users_dir: str = "data/users",
        model_name: str = "ArcFace",
        metric: str = "cosine",
        threshold: Optional[float] = None,
        save_crops: bool = True,
    ):
        self.users_dir = users_dir
        self.model_name = model_name
        self.metric = metric
        self.threshold = threshold if threshold is not None else self.MODEL_THRESHOLDS.get(model_name, 0.68)
        self.save_crops = save_crops
        os.makedirs(self.users_dir, exist_ok=True)

    def _emb_path(self, name: str) -> str:
        return os.path.join(self.users_dir, f"{name}.npy")

    def _crop_path(self, name: str) -> str:
        return os.path.join(self.users_dir, f"{name}.jpg")

    def _load_image(self, image_path_or_bgr: Union[str, np.ndarray]) -> np.ndarray:
        if not isinstance(image_path_or_bgr, str):
            return image_path_or_bgr

        image = cv2.imread(image_path_or_bgr)
        if image is None:
            raise ValueError(f"Could not read image: {image_path_or_bgr}")
        return image

    def _preprocess_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        from app.biometrics.alignment import FaceAligner
        from app.biometrics.landmarks import FaceLandmarkExtractor
        from app.biometrics.quality_assessment import FaceQualityAssessor

        extractor = FaceLandmarkExtractor(static_mode=True, refine=True)
        landmarks_list = extractor.extract(image_bgr)
        if not landmarks_list:
            print("No facial landmarks detected.")
            return None

        aligner = FaceAligner(desired_face_width=300)
        face_aligned, _ = aligner.align(image_bgr, landmarks_list[0])

        quality_result = FaceQualityAssessor().evaluate(face_aligned)
        if quality_result["total"] < 0.7:
            print(f"Face rejected due to low quality ({quality_result['total']:.2f}).")
            return None

        return face_aligned

    def register_user(self, name: str, image_path_or_bgr: Union[str, np.ndarray], preprocess: bool = True) -> dict:
        image = self._load_image(image_path_or_bgr)
        face_ready = self._preprocess_face(image) if preprocess else image
        if face_ready is None:
            raise ValueError("Could not prepare face for registration.")

        embedding = compute_embedding_from_aligned_face(face_ready, model_name=self.model_name)
        if embedding is None:
            raise ValueError("Could not compute face embedding.")

        np.save(self._emb_path(name), embedding)
        if self.save_crops:
            cv2.imwrite(self._crop_path(name), face_ready)

        return {"user": name, "embedding_dim": embedding.shape[0], "path": self._emb_path(name)}

    def verify_user(self, image_path_or_bgr: Union[str, np.ndarray], name: str, preprocess: bool = True) -> dict:
        ref_path = self._emb_path(name)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"User '{name}' is not registered.")

        reference_embedding = np.load(ref_path).astype("float32")
        image = self._load_image(image_path_or_bgr)
        face_ready = self._preprocess_face(image) if preprocess else image
        if face_ready is None:
            return {"user": name, "match": False, "reason": "no_face_detected"}

        test_embedding = compute_embedding_from_aligned_face(face_ready, model_name=self.model_name)
        if test_embedding is None:
            return {"user": name, "match": False, "reason": "embedding_failed"}

        if self.metric == "cosine":
            dist = float(distance.cosine(reference_embedding, test_embedding))
        else:
            dist = float(np.linalg.norm(reference_embedding - test_embedding))

        confidence = float(max(0.0, 1.0 - dist / max(self.threshold, 1e-6)))
        return {
            "user": name,
            "match": dist < self.threshold,
            "distance": dist,
            "confidence": confidence,
            "threshold": self.threshold,
            "metric": self.metric,
            "model": self.model_name,
        }
