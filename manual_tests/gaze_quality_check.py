from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.biometrics.detection import FaceDetector
from app.biometrics.gaze_estimator import GazeEstimator
from app.biometrics.quality_assessment import FaceQualityAssessor


WINDOW_NAME = "Gaze & Face Quality Check"
QUALITY_HISTORY = 5


def main():
    gaze_estimator = GazeEstimator()
    detector = FaceDetector()
    quality_assessor = FaceQualityAssessor()
    quality_history = deque(maxlen=QUALITY_HISTORY)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    try:
        print("Camera running. Press 'q' to exit, or close the window.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            detections = detector.detect(frame)
            if detections:
                x, y, width, height, _conf = max(detections, key=lambda item: item[4])
                face_crop = frame[y:y + height, x:x + width]

                quality_result = quality_assessor.evaluate(face_crop)
                quality_history.append(quality_result["total"])
                avg_quality = np.mean(quality_history)

                gaze_result = gaze_estimator.estimate(frame)
                label = gaze_result.get("label", "N/A") if gaze_result else "N/A"

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"Gaze: {label}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                color = (0, 255, 0) if quality_result["decision"] == "accept" else (0, 0, 255)
                cv2.putText(frame, f"Quality: {avg_quality:.2f} ({quality_result['decision']})",
                            (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(frame, "No face detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
