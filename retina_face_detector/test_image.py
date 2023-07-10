import os
import sys

import cv2
import numpy as np

from retinaface import RetinaFace

THRESHOLD = 0.8
GPUID = 0  # -1 to use CPU
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PREFIX = os.path.join(THIS_DIR, 'models', 'mnet.25', 'mnet.25')

MAX_WIDTH = 1920
MAX_HEIGHT = 1080


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 test_image.py <img1> [img2 ...]')
        sys.exit(0)

    detector = RetinaFace(MODEL_PREFIX, 0, GPUID, 'net3')

    for fn in sys.argv[1:]:
        img = cv2.imread(fn)

        height, width = img.shape[:2]
        if height > MAX_HEIGHT or width > MAX_WIDTH:
            scales = [min(MAX_HEIGHT / height, MAX_WIDTH / width)]
        else:
            scales = [1.0]

        faces, landmarks = detector.detect(img, THRESHOLD, scales=scales)

        if faces is not None:
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

            filename = fn + '_out.png'
            cv2.imwrite(filename, img)
