import cv2
import threading


class Camera:

    def __init__(self, config):
        self.config = config
        self.cam = cv2.VideoCapture(config['index_or_path'], cv2.CAP_AVFOUNDATION)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
        self.cam.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.frame_lock = threading.Lock()
        self.cam_ready = threading.Event()
        self.thread = threading.Thread(target=self.cam_loop_, name='camera_loop', daemon=True)
        self.thread.start()
        self.cam_ready.wait()

    def cam_loop_(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue
            with self.frame_lock:
                self.frame = frame
            self.cam_ready.set()

    def get_frame(self):
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()