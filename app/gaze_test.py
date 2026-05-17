import os
import random
import threading
import time

import pygame

from app.biometrics.detection import FaceDetector
from app.biometrics.face_verification import FaceVerifier
from app.biometrics.gaze_estimator import GazeEstimator


class GazeTest:
    def __init__(self, camera_feed, callback_on_finish, root, username=None):
        self.camera_feed = camera_feed
        self.callback_on_finish = callback_on_finish
        self.root = root
        self.username = username

        self.gaze_estimator = GazeEstimator()
        self.face_detector = FaceDetector(min_conf=0.6)
        self.verifier = FaceVerifier(
            users_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users")
        )

        self.directions = ["izquierda", "derecha", "arriba", "abajo", "centro"]
        self.dir_map = {
            "izquierda": "left",
            "derecha": "right",
            "arriba": "up",
            "abajo": "down",
            "centro": "middle",
        }

        self.required_hold_time = 3.0
        self.timeout_per_test = 10.0
        self.tests_needed = 4
        self.completed = 0
        self.running = False
        self.no_face_frames = 0
        self.max_no_face_frames = 10

        pygame.mixer.init()
        audio_dir = os.path.join(os.path.dirname(__file__), "audios")
        self.success_sound = os.path.join(audio_dir, "ding.wav")
        self.audio_files = {
            "comienzo": os.path.join(audio_dir, "comienzo.mp3"),
            "prueba_superada": os.path.join(audio_dir, "prueba_superada.mp3"),
            "prueba_fallida": os.path.join(audio_dir, "prueba_fallida.mp3"),
            "fallo_rostro": os.path.join(audio_dir, "fallo_rostro.mp3"),
            "fallo_verificacion": os.path.join(audio_dir, "fallo_verificacion.mp3"),
            "fallo_multiple": os.path.join(audio_dir, "fallo_multiple.mp3"),
            "fallo_direccion": os.path.join(audio_dir, "fallo_direccion.mp3"),
            "mira_izquierda": os.path.join(audio_dir, "mira_izquierda.mp3"),
            "mira_derecha": os.path.join(audio_dir, "mira_derecha.mp3"),
            "mira_arriba": os.path.join(audio_dir, "mira_arriba.mp3"),
            "mira_abajo": os.path.join(audio_dir, "mira_abajo.mp3"),
            "mira_centro": os.path.join(audio_dir, "mira_centro.mp3"),
        }

    def _speak(self, key):
        def run_audio():
            try:
                path = self.audio_files.get(key)
                if not path or not os.path.exists(path):
                    print(f"[WARN] Missing audio file for '{key}'")
                    return

                if not pygame.mixer.get_init():
                    pygame.mixer.init()

                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as exc:
                print(f"[WARN] Could not play audio '{key}': {exc}")

        threading.Thread(target=run_audio, daemon=True).start()

    def _play_success(self):
        try:
            pygame.mixer.music.load(self.success_sound)
            pygame.mixer.music.play()
        except Exception as exc:
            print(f"[WARN] Could not play success sound: {exc}")

    def start(self, update_instruction_callback, update_message_callback):
        self.update_instruction_callback = update_instruction_callback
        self.update_message_callback = update_message_callback
        self.running = True
        self.completed = 0
        threading.Thread(target=self._delayed_start, daemon=True).start()

    def _delayed_start(self):
        self._speak("comienzo")
        time.sleep(3)
        self._run_tests()

    def stop(self):
        self.running = False

    def _build_direction_sequence(self):
        sequence = []
        core_directions = ["izquierda", "derecha", "arriba", "abajo"]
        center_index = random.randrange(self.tests_needed)
        previous_direction = None

        for index in range(self.tests_needed):
            if index == center_index:
                sequence.append("centro")
                previous_direction = "centro"
                continue

            choices = [direction for direction in core_directions if direction != previous_direction]
            selected = random.choice(choices)
            sequence.append(selected)
            previous_direction = selected

        return sequence

    def _run_tests(self):
        failure_reason = None
        sequence = self._build_direction_sequence()

        for current_direction in sequence:
            if not self.running or self.completed >= self.tests_needed:
                break

            expected_label = self.dir_map[current_direction]
            self.update_instruction_callback(f"Mira hacia {current_direction.upper()}")
            self.update_message_callback(f"Esperando que mires hacia {current_direction}...")
            self._speak(f"mira_{current_direction}")
            time.sleep(0.5)

            start_time = time.time()
            hold_start = None
            success = False

            while self.running and (time.time() - start_time < self.timeout_per_test):
                frame = self.camera_feed.get_last_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                faces = self.face_detector.detect(frame)
                if not faces:
                    self.no_face_frames += 1
                    if self.no_face_frames >= self.max_no_face_frames:
                        failure_reason = "fallo_rostro"
                        break
                    time.sleep(0.1)
                    continue
                if len(faces) > 1:
                    failure_reason = "fallo_multiple"
                    break

                self.no_face_frames = 0
                label = self.gaze_estimator.estimate(frame).get("label", "")

                if label == expected_label:
                    if hold_start is None:
                        hold_start = time.time()
                        self.update_message_callback("Direccion correcta, manten la mirada...")
                    elif time.time() - hold_start >= self.required_hold_time:
                        if current_direction == "centro" and self.username:
                            verify_result = self.verifier.verify_user(frame, self.username)
                            if not verify_result.get("match", False):
                                failure_reason = "fallo_verificacion"
                                break

                        self.completed += 1
                        self._play_success()
                        self.update_message_callback(
                            f"Direccion '{current_direction}' superada ({self.completed}/{self.tests_needed})"
                        )
                        success = True
                        break
                else:
                    hold_start = None

                time.sleep(0.1)

            if not success:
                failure_reason = failure_reason or "fallo_direccion"
                break

            time.sleep(1)

        self._finish(self.completed >= self.tests_needed, failure_reason)

    def _finish(self, success, failure_reason=None):
        if success:
            self.update_instruction_callback("Prueba superada.")
            self.update_message_callback("Transaccion realizada correctamente.")
            self._speak("prueba_superada")
            self._play_success()
        else:
            self.update_instruction_callback("Prueba fallida.")
            self.update_message_callback("Transaccion cancelada.")
            self._speak(failure_reason or "prueba_fallida")

        time.sleep(1.5)
        self.root.after(0, lambda: self.callback_on_finish(success))
