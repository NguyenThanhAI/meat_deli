"""Motion detection modules."""

import cv2

import imutils

MIN_SIZE = (30, 30)


class MotionDetection:
    def __init__(self, min_size=MIN_SIZE):
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.min_size = min_size

    def get_motion_region(self, frame):
        """Return region with motion."""
        orig_size = frame.shape[1]
        new_size = min(orig_size, 200)
        scale = float(orig_size) / new_size

        if orig_size != new_size:
            frame_local = imutils.resize(frame, width=new_size)

        frame_local = cv2.blur(frame_local, (3, 3))

        mask = self.mog.apply(frame_local, learningRate=0.00075)
        mask = cv2.dilate(mask, None, iterations=1)

        # cv2.imshow('Mask', mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        t = mask.shape[0]
        b = 0
        l = mask.shape[1]
        r = 0
        ret = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            t = min(t, y)
            b = max(b, y + h)
            l = min(l, x)
            r = max(r, x + w)

        t = int(scale * t - 1)
        b = int(scale * b + 1)
        l = int(scale * l - 1)
        r = int(scale * r + 1)

        t = max(t, 0)
        b = min(b, frame.shape[0])
        l = max(l, 0)
        r = min(r, frame.shape[1])

        if l < r and t < b and (r - l) > self.min_size[0] and (b - t) > self.min_size[1]:
            ret.append((l, t, r - l, b - t))
        return ret
