import threading

import cv2
from PIL import Image, ImageTk


class CameraFeed:
    def __init__(self, label_widget):
        self.cap = None
        self.label_widget = label_widget
        self.running = False
        self.last_frame = None

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not access the camera.")
            return

        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            self.last_frame = frame.copy()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(Image.fromarray(image))
            if self.label_widget.winfo_exists():
                self.label_widget.configure(image=image)
                self.label_widget.image = image

    def get_last_frame(self):
        return self.last_frame

    def capture(self):
        return self.last_frame.copy() if self.last_frame is not None else None

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.label_widget and self.label_widget.winfo_exists():
            self.label_widget.configure(image="")
