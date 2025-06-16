import cv2
import os


class NumberPlateDetector:
    def __init__(self):
        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_russian_plate_number.xml')
        self.cascade_classifier = cv2.CascadeClassifier(cascade_path)

    def detect_plate(self, frame):
        plates = self.cascade_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        for (x, y, w, h) in plates:
            plate_img = frame[y:y+h, x:x+w]
            return plate_img, (x, y, w, h)
        return None, None