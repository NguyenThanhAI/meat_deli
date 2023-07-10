import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(THIS_DIR)

from retinaface import RetinaFace

THRESHOLD = 0.8
GPUID = 0  # -1 to use CPU


class RetinaFaceDetector:
    def __init__(self, gpuid=GPUID, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(THIS_DIR, 'models')
        MODEL_PREFIX = os.path.join(model_dir, 'mnet.25', 'mnet.25')
        self.detector = RetinaFace(MODEL_PREFIX, 0, gpuid, 'net3')

    def detect(self, img, thresh=THRESHOLD):
        return self.detector.detect(img, thresh)
