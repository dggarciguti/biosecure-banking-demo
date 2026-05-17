from pathlib import Path
import sys
import tkinter as tk
from tkinter import Label

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.biometrics.detection import FaceDetector
from app.biometrics.quality_assessment import FaceQualityAssessor


class DetectionQualityCheck:
    def __init__(self):
        self.detector = FaceDetector()
        self.quality = FaceQualityAssessor()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.running = True
        self.after_id = None

        self.window = tk.Tk()
        self.window.title("Face Detection & Quality Assessment")
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
            self.draw_frame(frame)

        if self.running and self.window.winfo_exists():
            self.after_id = self.window.after(10, self.update_frame)

    def draw_frame(self, frame):
        detections = self.detector.detect(frame)
        if detections:
            x, y, width, height, conf = max(detections, key=lambda item: item[4])
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            face_crop = frame[y:y + height, x:x + width]
            if face_crop.size > 0:
                result = self.quality.evaluate(face_crop)
                text = f"Detected ({conf:.2f}) - Quality: {result['total']:.2f} ({result['decision']})"
                color = "green" if result["decision"] == "accept" else "orange"
                self.status_label.config(text=text, fg=color)
            else:
                self.status_label.config(text="Face detected but crop invalid", fg="orange")
        else:
            self.status_label.config(text="Not detected", fg="red")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.video_label.imgtk = image
        self.video_label.configure(image=image)

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
    DetectionQualityCheck().run()
