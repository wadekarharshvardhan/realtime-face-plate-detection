import cv2

class Camera:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def __del__(self):
        self.video_capture.release()

    def get_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return None
        return frame