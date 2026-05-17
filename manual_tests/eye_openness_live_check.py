from collections import deque
from pathlib import Path
import sys
import tkinter as tk
from tkinter import Label

import cv2
import numpy as np
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.biometrics.detection import FaceDetector
from app.biometrics.eye_openness_detector import EyeOpennessDetector
from app.biometrics.landmarks import FaceLandmarkExtractor


MODEL_PATH = PROJECT_ROOT / "models" / "best_eye_detection_model_tf"
HISTORY_FRAMES = 10
THRESHOLD = 0.35


class EyeOpennessLiveCheck:
    def __init__(self):
        self.detector = FaceDetector()
        self.landmarks_extractor = FaceLandmarkExtractor(static_mode=False, refine=True)
        self.eye_detector = EyeOpennessDetector(model_path=str(MODEL_PATH), threshold=THRESHOLD)
        self.left_history = deque(maxlen=HISTORY_FRAMES)
        self.right_history = deque(maxlen=HISTORY_FRAMES)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.running = True
        self.after_id = None

        self.window = tk.Tk()
        self.window.title("Eye Openness Live Detector")
        self.window.geometry("950x700")

        self.video_label = Label(self.window)
        self.video_label.pack()
        self.status_label = Label(self.window, text="Initializing...", font=("Helvetica", 16))
        self.status_label.pack(pady=10)

        self.window.bind("<q>", self.close)
        self.window.bind("<Q>", self.close)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.draw_frame(frame)

        if self.running and self.window.winfo_exists():
            self.after_id = self.window.after(10, self.update_frame)

    def draw_frame(self, frame):
        detections = self.detector.detect(frame)
        if detections:
            landmarks_list = self.landmarks_extractor.extract(frame)
            if landmarks_list:
                self.draw_eye_state(frame, landmarks_list[0])
            else:
                self.status_label.config(text="Face detected but no landmarks", fg="orange")
        else:
            self.status_label.config(text="No face detected", fg="red")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.video_label.imgtk = image
        self.video_label.configure(image=image)

    def draw_eye_state(self, frame, landmarks):
        results = self.eye_detector.predict_eyes(frame, landmarks)
        left_prob = results["left"]["prob_open"]
        right_prob = results["right"]["prob_open"]

        if left_prob is not None:
            self.left_history.append(left_prob)
        if right_prob is not None:
            self.right_history.append(right_prob)

        left_avg = np.mean(self.left_history) if self.left_history else 0
        right_avg = np.mean(self.right_history) if self.right_history else 0
        left_state = "open" if left_avg >= THRESHOLD else "closed"
        right_state = "open" if right_avg >= THRESHOLD else "closed"

        cv2.putText(frame, f"Left Eye: {left_state.upper()} ({left_avg:.2f})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if left_state == "open" else (0, 0, 255), 2)
        cv2.putText(frame, f"Right Eye: {right_state.upper()} ({right_avg:.2f})",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if right_state == "open" else (0, 0, 255), 2)

        self.status_label.config(
            text=f"Eyes detected - L:{left_state} / R:{right_state}",
            fg="green" if left_state == right_state == "open" else "orange",
        )

    def close(self, _event=None):
        if not self.running:
            return

        self.running = False
        if self.after_id is not None:
            try:
                self.window.after_cancel(self.after_id)
            except tk.TclError:
                pass
        if self.cap.isOpened():
            self.cap.release()
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def run(self):
        try:
            self.update_frame()
            self.window.mainloop()
        finally:
            self.close()


if __name__ == "__main__":
    EyeOpennessLiveCheck().run()
