import cv2
import numpy as np


class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height if desired_face_height is not None else desired_face_width

    def align(self, image, landmarks):
        left_eye_points = [33, 133]
        right_eye_points = [362, 263]

        left_eye_center = np.mean([(landmarks[index][0], landmarks[index][1]) for index in left_eye_points], axis=0)
        right_eye_center = np.mean([(landmarks[index][0], landmarks[index][1]) for index in right_eye_points], axis=0)

        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        desired_dist = (1.0 - 2 * self.desired_left_eye[0]) * self.desired_face_width
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        scale = desired_dist / dist

        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) / 2,
            (left_eye_center[1] + right_eye_center[1]) / 2,
        )
        matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        target_x = self.desired_face_width * 0.5
        target_y = self.desired_face_height * self.desired_left_eye[1]
        matrix[0, 2] += target_x - eyes_center[0]
        matrix[1, 2] += target_y - eyes_center[1]

        output = cv2.warpAffine(
            image,
            matrix,
            (self.desired_face_width, self.desired_face_height),
            flags=cv2.INTER_CUBIC,
        )
        return output, angle
