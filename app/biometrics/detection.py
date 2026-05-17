import cv2, mediapipe as mp

class FaceDetector:
    def __init__(self, model_selection=0, min_conf=0.6):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_conf)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)
        dets=[]
        if res.detections:
            h,w = bgr.shape[:2]
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x,y = int(bb.xmin*w), int(bb.ymin*h)
                ww,hh = int(bb.width*w), int(bb.height*h)
                dets.append((x,y,ww,hh,d.score[0]))
        return dets



