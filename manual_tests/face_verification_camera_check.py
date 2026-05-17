from pathlib import Path
import sys
import tkinter as tk
from tkinter import Button, Entry, Label, Toplevel, messagebox

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.biometrics.face_verification import FaceVerifier


USERS_DIR = PROJECT_ROOT / "data" / "users"
verifier = FaceVerifier(users_dir=str(USERS_DIR))

root = tk.Tk()
root.title("Face Verification")
root.geometry("500x340")

Label(root, text="Face Verification", font=("Helvetica", 18, "bold")).pack(pady=20)
Label(root, text="User name:", font=("Helvetica", 13)).pack()
name_entry = Entry(root, font=("Helvetica", 13), width=30)
name_entry.pack(pady=8)
status_label = Label(root, text="Select an action", font=("Helvetica", 12))
status_label.pack(pady=10)

open_dialogs = []


class CameraDialog:
    def __init__(self, mode: str, user_name: str):
        self.mode = mode
        self.user_name = user_name
        self.cap = None
        self.after_id = None
        self.running = True

        self.dialog = Toplevel(root)
        self.dialog.title("Camera")
        self.dialog.geometry("800x600")
        self.dialog.grab_set()

        info_text = "Capture for REGISTRATION" if mode == "register" else "Capture for LOGIN"
        Label(self.dialog, text=info_text, font=("Helvetica", 14)).pack(pady=10)
        self.video_label = Label(self.dialog)
        self.video_label.pack(pady=10)
        Label(self.dialog, text="Press Capture to take a picture. ESC or q closes the window.",
              font=("Helvetica", 10), fg="#666").pack(pady=5)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            self.close()
            return

        self.dialog.bind("<Key>", self.on_key)
        self.dialog.protocol("WM_DELETE_WINDOW", self.close)

        button_row = tk.Frame(self.dialog)
        button_row.pack(pady=10)
        Button(button_row, text="Capture", font=("Helvetica", 12),
               bg="#4CAF50", fg="white", width=12,
               command=self.capture_and_process).grid(row=0, column=0, padx=6)
        Button(button_row, text="Cancel", font=("Helvetica", 12),
               bg="#999999", fg="white", width=12,
               command=self.close).grid(row=0, column=1, padx=6)

        open_dialogs.append(self)
        self.update_preview()

    def on_key(self, event):
        if event.keysym == "Escape" or (event.char and event.char.lower() == "q"):
            self.close()

    def update_preview(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.video_label.imgtk = image
            self.video_label.configure(image=image)

        if self.running and self.dialog.winfo_exists():
            self.after_id = self.dialog.after(16, self.update_preview)

    def capture_and_process(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not capture image.")
            return

        try:
            if self.mode == "register":
                info = verifier.register_user(self.user_name, frame, preprocess=True)
                messagebox.showinfo(
                    "Registration complete",
                    f"User '{self.user_name}' registered.\nEmbedding: {info['embedding_dim']} dim.",
                )
                status_label.config(text=f"Registered: {self.user_name}", fg="green")
            else:
                result = verifier.verify_user(frame, self.user_name, preprocess=True)
                if result.get("match", False):
                    messagebox.showinfo(
                        "Verification complete",
                        f"Matched '{self.user_name}'.\nConfidence: {result['confidence'] * 100:.1f}%",
                    )
                    status_label.config(text=f"Verified: {self.user_name}", fg="green")
                else:
                    messagebox.showwarning("No match", f"No match for '{self.user_name}'.")
                    status_label.config(text=f"Verification failed: {self.user_name}", fg="red")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            status_label.config(text="Operation failed", fg="red")
        finally:
            self.close()

    def close(self):
        if not self.running:
            return

        self.running = False
        if self.after_id is not None:
            try:
                self.dialog.after_cancel(self.after_id)
            except tk.TclError:
                pass
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self in open_dialogs:
            open_dialogs.remove(self)
        try:
            self.dialog.grab_release()
        except tk.TclError:
            pass
        try:
            self.dialog.destroy()
        except tk.TclError:
            pass


def open_camera_dialog(mode: str, user_name: str):
    if not user_name:
        messagebox.showwarning("Warning", "Enter a user name.")
        return
    CameraDialog(mode, user_name)


def on_register():
    open_camera_dialog("register", name_entry.get().strip())


def on_login():
    open_camera_dialog("login", name_entry.get().strip())


def close_root():
    for dialog in list(open_dialogs):
        dialog.close()
    try:
        root.destroy()
    except tk.TclError:
        pass


Button(root, text="Register", font=("Helvetica", 14),
       bg="#673AB7", fg="white", width=16, command=on_register).pack(pady=10)
Button(root, text="Login", font=("Helvetica", 14),
       bg="#2196F3", fg="white", width=16, command=on_login).pack(pady=6)

root.protocol("WM_DELETE_WINDOW", close_root)
root.mainloop()
