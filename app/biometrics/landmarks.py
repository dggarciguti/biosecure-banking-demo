import cv2, mediapipe as mp

class FaceLandmarkExtractor:
    def __init__(self, static_mode=False, max_faces=1, refine=True,
                 min_det=0.6, min_trk=0.6):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk)

    def extract(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        all_pts=[]
        if res.multi_face_landmarks:
            h,w = bgr.shape[:2]
            for fl in res.multi_face_landmarks:
                pts=[(int(lm.x*w), int(lm.y*h), lm.z) for lm in fl.landmark]
                all_pts.append(pts)
        return all_pts
