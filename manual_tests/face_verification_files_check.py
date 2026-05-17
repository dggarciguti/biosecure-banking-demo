from pathlib import Path
import sys
import tkinter as tk
from tkinter import Button, Label, filedialog, messagebox

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.biometrics.face_verification import FaceVerifier


USERS_DIR = PROJECT_ROOT / "data" / "users"
TEMP_USER = "temp_manual_check"
verifier = FaceVerifier(users_dir=str(USERS_DIR), model_name="ArcFace")

root = tk.Tk()
root.title("Face Verification From Files")
root.geometry("420x200")


def cleanup_temp_user():
    for suffix in [".npy", ".jpg", ".json"]:
        path = USERS_DIR / f"{TEMP_USER}{suffix}"
        if path.exists():
            path.unlink()


def select_and_verify():
    messagebox.showinfo("Select images", "First select the reference image.")
    ref_path = filedialog.askopenfilename(
        title="Select reference image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")],
    )
    if not ref_path:
        return

    messagebox.showinfo("Select images", "Now select the image to compare.")
    test_path = filedialog.askopenfilename(
        title="Select test image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")],
    )
    if not test_path:
        return

    try:
        verifier.register_user(TEMP_USER, ref_path, preprocess=True)
        result = verifier.verify_user(test_path, TEMP_USER, preprocess=True)

        ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        test_img = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ref_img)
        axes[0].set_title("Reference")
        axes[0].axis("off")
        axes[1].imshow(test_img)
        axes[1].set_title("MATCH" if result["match"] else "NO MATCH")
        axes[1].axis("off")
        plt.suptitle(
            f"Distance: {result['distance']:.3f}",
            fontsize=12,
            color="green" if result["match"] else "red",
        )
        plt.show(block=False)

        messagebox.showinfo(
            "Verification result",
            f"Result: {'MATCH' if result['match'] else 'NO MATCH'}\n"
            f"Distance: {result['distance']:.3f}\n"
            f"Confidence: {result['confidence'] * 100:.1f}%",
        )
    except Exception as exc:
        messagebox.showerror("Verification error", str(exc))
    finally:
        cleanup_temp_user()


def close_root():
    cleanup_temp_user()
    plt.close("all")
    try:
        root.destroy()
    except tk.TclError:
        pass


Label(root, text="Face Verification", font=("Helvetica", 18, "bold")).pack(pady=20)
Button(root, text="Select images and verify",
       font=("Helvetica", 13), bg="#4CAF50", fg="white",
       width=28, height=2, command=select_and_verify).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", close_root)
root.mainloop()
