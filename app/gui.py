import tkinter as tk

from app.camera_utils import CameraFeed
from app.logic import (
    get_user_funds,
    login_user,
    register_user,
    set_initial_funds,
    transfer_funds,
    user_exists,
)


class BankAppGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("App Bancaria Segura")
        self.root.geometry("400x720")
        self.root.resizable(False, False)

        self.bg_color = "#D6DADF"
        self.primary_color = "#0078D7"
        self.success_color = "#34A853"
        self.secondary_color = "#6C757D"
        self.error_color = "#D93025"
        self.text_dark = "#2D2D2D"
        self.text_soft = "#555"

        self.root.configure(bg=self.bg_color)
        self.root.option_add("*Font", ("Segoe UI", 11))
        self.root.option_add("*Button.Font", ("Segoe UI", 11, "bold"))
        self.root.option_add("*Label.Font", ("Segoe UI", 11))
        self.root.option_add("*Button.relief", "flat")
        self.root.option_add("*Button.activeForeground", "white")

        self.current_frame = None
        self.camera = None
        self.pending_transfer = None
        self.username_var = tk.StringVar()
        self.message_var = tk.StringVar()

        self._build_home_screen()

    def _add_hover_effect(self, button, normal_color, hover_color):
        button.bind("<Enter>", lambda _event: button.config(background=hover_color))
        button.bind("<Leave>", lambda _event: button.config(background=normal_color))

    def _clear_screen(self):
        if self.current_frame:
            self.current_frame.destroy()

    def _button(self, parent, text, color, command, width=20, height=1):
        button = tk.Button(
            parent,
            text=text,
            bg=color,
            fg="white",
            width=width,
            height=height,
            command=command,
        )
        hover = "#005A9E" if color == self.primary_color else "#2C8C46"
        if color == self.secondary_color:
            hover = "#5a6268"
        self._add_hover_effect(button, color, hover)
        return button

    def _build_home_screen(self):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both")

        tk.Label(
            self.current_frame,
            text="Banco Seguro",
            font=("Segoe UI", 24, "bold"),
            bg=self.bg_color,
            fg=self.text_dark,
        ).pack(pady=(50, 5))
        tk.Label(
            self.current_frame,
            text="Simulacion biometrica facial",
            font=("Segoe UI", 12),
            bg=self.bg_color,
            fg=self.text_soft,
        ).pack(pady=(0, 25))

        self._button(self.current_frame, "Iniciar sesion", self.primary_color, self._build_login_name_screen, 18).pack(pady=20)
        self._button(self.current_frame, "Registrar usuario", self.success_color, self._build_register_name_screen, 18).pack(pady=10)

    def _build_register_name_screen(self):
        self._build_name_screen(
            title="Registro de usuario",
            button_text="Continuar",
            button_color=self.success_color,
            action=self._go_to_register_photo,
        )

    def _build_login_name_screen(self):
        self._build_name_screen(
            title="Inicio de sesion",
            button_text="Continuar",
            button_color=self.primary_color,
            action=self._go_to_login_photo,
        )

    def _build_name_screen(self, title, button_text, button_color, action):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both")

        tk.Label(
            self.current_frame,
            text=title,
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.text_dark,
        ).pack(pady=40)

        entry = tk.Entry(self.current_frame, textvariable=self.username_var, font=("Segoe UI", 13), relief="flat", justify="center")
        entry.pack(pady=10, ipadx=10, ipady=6)

        tk.Label(
            self.current_frame,
            text="Introduce tu nombre y pulsa continuar.",
            bg=self.bg_color,
            fg=self.text_soft,
        ).pack(pady=5)

        self._button(self.current_frame, button_text, button_color, action).pack(pady=15)
        self._button(self.current_frame, "Volver", self.secondary_color, self._back_to_home).pack(pady=10)

    def _go_to_register_photo(self):
        if not self.username_var.get().strip():
            self._show_message("Por favor, introduce un nombre de usuario.", error=True)
            return
        self._build_camera_screen(mode="register")

    def _go_to_login_photo(self):
        user = self.username_var.get().strip()
        if not user:
            self._show_message("Por favor, introduce un nombre de usuario.", error=True)
            return
        if not user_exists(user):
            self._show_message("El usuario no existe. Registrate antes de iniciar sesion.", error=True)
            self.username_var.set("")
            return

        self.message_var.set("")
        self._build_camera_screen(mode="login")

    def _build_camera_screen(self, mode="register"):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both")

        title = "Registrar usuario" if mode == "register" else "Verificacion de usuario"
        action_text = "Capturar y registrar" if mode == "register" else "Verificar acceso"
        action_color = self.success_color if mode == "register" else self.primary_color
        action = self._capture_register if mode == "register" else self._capture_verify

        tk.Label(
            self.current_frame,
            text=title,
            font=("Segoe UI", 20, "bold"),
            bg=self.bg_color,
            fg=self.text_dark,
        ).pack(pady=(25, 10))
        tk.Label(
            self.current_frame,
            text=f"Usuario: {self.username_var.get()}",
            bg=self.bg_color,
            font=("Segoe UI", 13, "bold"),
            fg=self.text_soft,
        ).pack(pady=(0, 15))

        self.video_label = tk.Label(self.current_frame, bg="black", width=360, height=300)
        self.video_label.pack(pady=(5, 25))
        self.camera = CameraFeed(self.video_label)
        self.camera.start()

        self._button(self.current_frame, action_text, action_color, action, width=22, height=2).pack(pady=(5, 15))
        self._button(self.current_frame, "Volver", self.secondary_color, self._back_to_home, width=22, height=2).pack(pady=(10, 15))

        self.message_label = tk.Label(self.current_frame, textvariable=self.message_var, bg=self.bg_color, fg=self.text_soft, font=("Segoe UI", 12))
        self.message_label.pack(pady=10)

    def _capture_register(self):
        photo = self.camera.capture()
        user = self.username_var.get().strip()

        if photo is None:
            self._show_message("No se pudo capturar la imagen.", error=True)
            return

        self.camera.stop()
        if register_user(user, photo):
            self._show_message("Usuario registrado. Introduce tus fondos iniciales.")
            self._build_funds_input_screen()
        else:
            self._build_register_failed_screen("No se pudo registrar el usuario. Intentalo de nuevo.")

    def _build_register_failed_screen(self, message="Error al registrar usuario"):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both", padx=30, pady=30)

        tk.Label(
            self.current_frame,
            text="Registro fallido",
            font=("Segoe UI", 20, "bold"),
            bg=self.bg_color,
            fg=self.error_color,
        ).pack(pady=(80, 10))
        tk.Label(
            self.current_frame,
            text=message,
            font=("Segoe UI", 12),
            bg=self.bg_color,
            fg=self.text_soft,
            wraplength=340,
            justify="center",
        ).pack(pady=(0, 40))
        tk.Label(
            self.current_frame,
            text="Volviendo al menu...",
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg=self.text_soft,
        ).pack()

        self.root.after(2000, self._back_to_home)

    def _build_funds_input_screen(self):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both")

        tk.Label(self.current_frame, text="Fondos iniciales", font=("Segoe UI", 18, "bold"), bg=self.bg_color, fg=self.text_dark).pack(pady=30)
        tk.Label(self.current_frame, text=f"Usuario: {self.username_var.get()}", bg=self.bg_color, fg=self.text_soft, font=("Segoe UI", 12)).pack(pady=10)
        tk.Label(self.current_frame, text="Introduce fondos iniciales (EUR):", bg=self.bg_color, fg=self.text_soft).pack(pady=10)

        amount_var = tk.StringVar()
        entry = tk.Entry(self.current_frame, textvariable=amount_var, font=("Segoe UI", 13), relief="flat", justify="center")
        entry.pack(pady=10, ipadx=10, ipady=6)

        def save_funds():
            try:
                amount = float(amount_var.get())
            except ValueError:
                self._show_message("Introduce una cantidad valida.", error=True)
                return

            set_initial_funds(self.username_var.get(), amount)
            self._show_message(f"Fondos iniciales: {amount:.2f} EUR guardados correctamente.")
            self._build_account_screen(self.username_var.get())

        self._button(self.current_frame, "Guardar", self.success_color, save_funds, 18).pack(pady=15)
        self._button(self.current_frame, "Cancelar", self.secondary_color, self._back_to_home, 18).pack(pady=10)

    def _capture_verify(self):
        photo = self.camera.capture()
        user = self.username_var.get().strip()
        if photo is None:
            self._show_message("No se pudo capturar la imagen.", error=True)
            return

        self.camera.stop()
        if not user_exists(user):
            self._show_message("El usuario no existe. Registrate antes de iniciar sesion.", error=True)
            return

        self._build_verifying_screen(user, photo)

    def _build_verifying_screen(self, username, photo):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both")

        tk.Label(
            self.current_frame,
            text="Verificando identidad...",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.text_dark,
        ).pack(pady=80)
        tk.Label(
            self.current_frame,
            text="Procesando...",
            font=("Segoe UI", 13),
            bg=self.bg_color,
            fg=self.text_soft,
        ).pack(pady=10)

        # Delay lets Tkinter render the processing screen before running inference.
        self.root.after(500, lambda: self._do_verification(username, photo))

    def _do_verification(self, username, photo):
        verified = login_user(username, photo)
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both", padx=30, pady=30)

        message = "Verificacion completada correctamente." if verified else "Acceso denegado. Intenta de nuevo."
        color = self.success_color if verified else self.error_color
        tk.Label(
            self.current_frame,
            text=message,
            font=("Segoe UI", 16, "bold"),
            bg=self.bg_color,
            fg=color,
            wraplength=340,
            justify="center",
        ).pack(pady=(100, 40))

        if verified:
            tk.Label(
                self.current_frame,
                text="Redirigiendo a tu cuenta...",
                font=("Segoe UI", 12),
                bg=self.bg_color,
                fg=self.text_soft,
            ).pack()
            self.root.after(2000, lambda: self._build_account_screen(username))
        else:
            self._button(self.current_frame, "Volver", self.secondary_color, self._back_to_home, 18, 2).pack(pady=30)

    def _build_account_screen(self, username):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both")

        funds = get_user_funds(username)
        tk.Label(self.current_frame, text=f"Bienvenido, {username}", font=("Segoe UI", 18, "bold"), bg=self.bg_color, fg=self.text_dark).pack(pady=20)
        tk.Label(self.current_frame, text=f"Fondos actuales: {funds:.2f} EUR", font=("Segoe UI", 15), bg=self.bg_color, fg=self.text_soft).pack(pady=10)

        self._button(self.current_frame, "Enviar fondos", self.primary_color, lambda: self._build_transfer_screen(username), 20).pack(pady=20)
        self._button(self.current_frame, "Cerrar sesion", self.secondary_color, self._back_to_home, 20).pack(pady=10)

        self.message_label = tk.Label(self.current_frame, textvariable=self.message_var, bg=self.bg_color, fg=self.text_soft, font=("Segoe UI", 12))
        self.message_label.pack(pady=15)

    def _build_transfer_screen(self, username):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both")

        tk.Label(self.current_frame, text=f"Enviar fondos desde: {username}", font=("Segoe UI", 16, "bold"), bg=self.bg_color, fg=self.text_dark).pack(pady=20)
        tk.Label(self.current_frame, text="Destinatario:", bg=self.bg_color, fg=self.text_soft).pack()

        to_var = tk.StringVar()
        tk.Entry(self.current_frame, textvariable=to_var, font=("Segoe UI", 13), relief="flat", justify="center").pack(pady=5, ipadx=8, ipady=5)
        tk.Label(self.current_frame, text="Cantidad (EUR):", bg=self.bg_color, fg=self.text_soft).pack()

        amount_var = tk.StringVar()
        tk.Entry(self.current_frame, textvariable=amount_var, font=("Segoe UI", 13), relief="flat", justify="center").pack(pady=5, ipadx=8, ipady=5)

        def execute_transfer():
            try:
                amount = float(amount_var.get())
            except ValueError:
                self._show_message("Introduce una cantidad valida.", error=True)
                return

            to_user = to_var.get().strip()
            if not to_user:
                self._show_message("Introduce un destinatario valido.", error=True)
                return
            if not user_exists(to_user):
                self._show_message(f"El usuario '{to_user}' no existe.", error=True)
                return
            if amount <= 0:
                self._show_message("Introduce una cantidad positiva.", error=True)
                return

            available = get_user_funds(username)
            if amount > available:
                self._show_message(f"Fondos insuficientes. Solo tienes {available:.2f} EUR disponibles.", error=True)
                return

            self.pending_transfer = (username, to_user, amount)
            self._build_security_tests_screen()

        self._button(self.current_frame, "Enviar", self.primary_color, execute_transfer, 18).pack(pady=15)
        self._button(self.current_frame, "Cancelar", self.secondary_color, lambda: self._build_account_screen(username), 18).pack(pady=10)

    def _show_message(self, text, error=False):
        color = self.error_color if error else self.success_color
        self.message_var.set(text)
        try:
            if hasattr(self, "message_label") and self.message_label.winfo_exists():
                self.message_label.config(fg=color, text=text)
            else:
                self.message_label = tk.Label(self.current_frame, text=text, fg=color, bg=self.bg_color, font=("Segoe UI", 12))
                self.message_label.pack(pady=15)
        except tk.TclError:
            pass

    def _build_security_tests_screen(self):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both", padx=20, pady=20)

        tk.Label(
            self.current_frame,
            text="Selecciona tipo de verificacion",
            font=("Segoe UI", 15, "bold"),
            bg=self.bg_color,
            fg=self.text_dark,
        ).pack(pady=(40, 30))

        self._button(self.current_frame, "Test de mirada", self.primary_color, self._build_gaze_test_screen, 24, 2).pack(pady=15)
        self._button(self.current_frame, "Test de apertura ocular", self.success_color, self._build_eye_test_screen, 24, 2).pack(pady=15)
        self._button(self.current_frame, "Cancelar", self.secondary_color, lambda: self._build_account_screen(self.username_var.get()), 20, 2).pack(pady=(40, 10))

        tk.Label(
            self.current_frame,
            text="Elige uno de los metodos de verificacion para continuar la transferencia.",
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg=self.text_soft,
            wraplength=340,
            justify="center",
        ).pack(pady=(20, 5))

    def _build_gaze_test_screen(self):
        from app.gaze_test import GazeTest

        self._build_test_screen("Verificacion de atencion")

        def finish(success=True):
            self._finish_security_test(success)

        self.gaze_test = GazeTest(
            camera_feed=self.camera,
            callback_on_finish=finish,
            root=self.root,
            username=self.username_var.get(),
        )
        self.gaze_test.start(
            update_instruction_callback=self._update_instruction,
            update_message_callback=self._update_message,
        )

    def _build_eye_test_screen(self):
        from app.eye_test import EyeTest

        self._build_test_screen("Verificacion de apertura ocular")

        def finish(success=True):
            self._finish_security_test(success)

        self.eye_test = EyeTest(
            camera_feed=self.camera,
            callback_on_finish=finish,
            root=self.root,
            flip_for_mirror=True,
            username=self.username_var.get(),
        )
        self.eye_test.prepare(
            update_instruction_callback=self._update_instruction,
            update_message_callback=self._update_message,
        )

    def _build_test_screen(self, title):
        self._clear_screen()
        self.current_frame = tk.Frame(self.root, bg=self.bg_color)
        self.current_frame.pack(expand=True, fill="both", padx=20, pady=20)

        tk.Label(self.current_frame, text=title, bg=self.bg_color, fg=self.text_dark, font=("Segoe UI", 18, "bold")).pack(pady=(10, 20))
        self.camera_label = tk.Label(self.current_frame, bg="black", width=360, height=300)
        self.camera_label.pack(pady=(5, 20))

        self.camera = CameraFeed(self.camera_label)
        self.camera.start()

        self.instruction_var = tk.StringVar(value="Preparando prueba...")
        self.message_var = tk.StringVar(value="")

        tk.Label(
            self.current_frame,
            textvariable=self.instruction_var,
            bg=self.bg_color,
            fg=self.text_dark,
            font=("Segoe UI", 14, "bold"),
            wraplength=360,
            justify="center",
            height=2,
        ).pack(pady=(10, 8))
        tk.Label(
            self.current_frame,
            textvariable=self.message_var,
            bg=self.bg_color,
            fg=self.text_soft,
            font=("Segoe UI", 12),
            wraplength=360,
            justify="center",
            height=2,
        ).pack(pady=(0, 10))

        self._button(self.current_frame, "Cancelar", self.secondary_color, lambda: self._build_account_screen(self.username_var.get()), 20, 2).pack(pady=(20, 10))

    def _update_instruction(self, text):
        self.instruction_var.set(text)
        self.root.update_idletasks()

    def _update_message(self, text):
        self.message_var.set(text)
        self.root.update_idletasks()

    def _finish_security_test(self, success=True):
        if self.camera:
            self.camera.stop()

        if success and self.pending_transfer:
            from_user, to_user, amount = self.pending_transfer
            transfer_success, message = transfer_funds(from_user, to_user, amount)
            self._show_message(message, error=not transfer_success)

        self._build_account_screen(self.username_var.get())

    def _back_to_home(self):
        if self.camera:
            self.camera.stop()
        self.pending_transfer = None
        self.username_var.set("")
        self.message_var.set("")
        self._build_home_screen()

    def run(self):
        self.root.mainloop()
