import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import pygame

from app.biometrics.detection import FaceDetector
from app.biometrics.eye_openness_detector import EyeOpennessDetector
from app.biometrics.face_verification import FaceVerifier
from app.biometrics.gaze_estimator import GazeEstimator
from app.biometrics.landmarks import FaceLandmarkExtractor


class EyeTest:
    def __init__(self, camera_feed=None, callback_on_finish=None, root=None, flip_for_mirror=True, username=None):
        self.camera_feed = camera_feed
        self.callback_on_finish = callback_on_finish
        self.root = root
        self.flip_for_mirror = flip_for_mirror
        self.username = username

        self.required_hold_time = 1.0
        self.timeout_per_test = 10.0
        self.tests_needed = 4
        self.completed = 0
        self.running = False

        self.face_detector = FaceDetector(min_conf=0.6)
        self.landmarks_extractor = FaceLandmarkExtractor(static_mode=False, refine=True)

        project_root = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(project_root, "models", "best_eye_detection_model_tf")
        users_dir = os.path.join(project_root, "data", "users")

        self.threshold = 0.30
        self.eye_detector = EyeOpennessDetector(model_path=model_path, threshold=self.threshold)
        self.gaze_estimator = GazeEstimator()
        self.verifier = FaceVerifier(users_dir=users_dir)

        history_frames = 10
        self.left_history = deque(maxlen=history_frames)
        self.right_history = deque(maxlen=history_frames)

        self.no_face_frames = 0
        self.max_no_face_frames = 10

        pygame.mixer.init()
        audio_dir = os.path.join(os.path.dirname(__file__), "audios")
        self.success_sound = os.path.join(audio_dir, "ding.wav")
        self.audio_files = {
            "cierra_izq": os.path.join(audio_dir, "cierra_izq.mp3"),
            "cierra_der": os.path.join(audio_dir, "cierra_der.mp3"),
            "cierra_ambos": os.path.join(audio_dir, "cierra_ambos.mp3"),
            "abre_ambos": os.path.join(audio_dir, "abre_ambos.mp3"),
            "comienzo": os.path.join(audio_dir, "comienzo.mp3"),
            "prueba_completada": os.path.join(audio_dir, "prueba_completada.mp3"),
            "prueba_fallida": os.path.join(audio_dir, "prueba_fallida.mp3"),
            "fallo_rostro": os.path.join(audio_dir, "fallo_rostro.mp3"),
            "fallo_verificacion": os.path.join(audio_dir, "fallo_verificacion.mp3"),
            "fallo_accion": os.path.join(audio_dir, "fallo_accion.mp3"),
        }

        self.instruction_text = {
            "cierra_izq": "Cierra el ojo izquierdo",
            "cierra_der": "Cierra el ojo derecho",
            "cierra_ambos": "Cierra ambos ojos",
            "abre_ambos": "Abre los dos ojos y mira al centro",
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

    def prepare(self, update_instruction_callback, update_message_callback):
        self.update_instruction_callback = update_instruction_callback
        self.update_message_callback = update_message_callback

        self.update_instruction_callback("Preparando la prueba de deteccion ocular...")
        self.update_message_callback("Por favor, mantente frente a la camara.")
        threading.Thread(target=self._delayed_start, daemon=True).start()

    def _delayed_start(self):
        self.running = True
        self._speak("comienzo")
        time.sleep(3)
        self._run_tests()

    def stop(self):
        self.running = False

    def _eyes_state_ok(self, left_state, right_state, instruction):
        expected_states = {
            "cierra_izq": ("closed", "open"),
            "cierra_der": ("open", "closed"),
            "cierra_ambos": ("closed", "closed"),
            "abre_ambos": ("open", "open"),
        }
        return (left_state, right_state) == expected_states.get(instruction)

    def _run_tests(self):
        sequence = self._build_sequence()
        face_verified = False

        for instruction in sequence:
            if not self.running:
                break

            phrase = self.instruction_text[instruction]
            self.update_instruction_callback(phrase.upper())
            self.update_message_callback(f"Esperando: {phrase} ...")
            self._speak(instruction)
            time.sleep(0.8)

            hold_start = None
            success = False
            start_time = time.time()
            self.no_face_frames = 0

            while self.running and (time.time() - start_time < self.timeout_per_test):
                frame = self.camera_feed.get_last_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                if self.flip_for_mirror:
                    frame = cv2.flip(frame, 1)

                detections = self.face_detector.detect(frame)
                if not detections:
                    self.no_face_frames += 1
                    if self.no_face_frames >= self.max_no_face_frames:
                        self._fail_test("No se detecta tu rostro", "fallo_rostro")
                        return
                    time.sleep(0.1)
                    continue

                self.no_face_frames = 0
                landmarks_list = self.landmarks_extractor.extract(frame)
                if not landmarks_list:
                    self.update_message_callback("Rostro detectado pero sin landmarks.")
                    time.sleep(0.1)
                    continue

                landmarks = landmarks_list[0]
                results = self.eye_detector.predict_eyes(frame, landmarks)
                left_prob = results["right"]["prob_open"]
                right_prob = results["left"]["prob_open"]
                if left_prob is None or right_prob is None:
                    time.sleep(0.1)
                    continue

                self.left_history.append(left_prob)
                self.right_history.append(right_prob)
                left_state = "open" if np.mean(self.left_history) >= self.threshold else "closed"
                right_state = "open" if np.mean(self.right_history) >= self.threshold else "closed"

                if instruction == "abre_ambos" and not face_verified:
                    success = self._handle_final_eye_step(frame, left_state, right_state)
                    if not self.running:
                        return
                    if success:
                        face_verified = True
                        break
                    time.sleep(0.15)
                    continue

                if self._eyes_state_ok(left_state, right_state, instruction):
                    if hold_start is None:
                        hold_start = time.time()
                    elif time.time() - hold_start >= self.required_hold_time:
                        self._mark_step_success()
                        success = True
                        break
                else:
                    hold_start = None

                time.sleep(0.15)

            if not success:
                self._fail_test("No realizaste la accion indicada", "fallo_accion")
                return

            time.sleep(1.0)

        self._finish_success()

    def _handle_final_eye_step(self, frame, left_state, right_state):
        if left_state != "open" or right_state != "open":
            return False

        gaze = self.gaze_estimator.estimate(frame)
        if gaze.get("label") != "middle":
            return False

        self.update_message_callback("Mirada centrada. Verificando identidad...")
        if self.username:
            result = self.verifier.verify_user(frame, self.username)
            if not result.get("match", False):
                self._fail_test("Verificacion facial fallida", "fallo_verificacion")
                return False

        self._mark_step_success("Verificacion completada correctamente.")
        return True

    def _mark_step_success(self, message=None):
        self.completed += 1
        self._play_success()
        self.update_message_callback(message or f"Accion superada ({self.completed}/{self.tests_needed})")

    def _fail_test(self, reason, audio_key="prueba_fallida"):
        self.update_instruction_callback("Prueba fallida.")
        self.update_message_callback(f"{reason}. Transaccion cancelada.")
        self._speak(audio_key)
        self.root.after(0, lambda: self.callback_on_finish(False))
        self.running = False

    def _finish_success(self):
        self.update_instruction_callback("Prueba completada.")
        self.update_message_callback("Transaccion aprobada correctamente.")
        self._speak("prueba_completada")
        self._play_success()
        self.root.after(0, lambda: self.callback_on_finish(True))
        self.running = False

    def _build_sequence(self):
        sequence = ["cierra_izq", "cierra_der", "cierra_ambos"]
        np.random.shuffle(sequence)
        sequence.append("abre_ambos")
        return sequence
