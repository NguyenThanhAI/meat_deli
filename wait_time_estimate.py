import time

import cv2
import numpy as np


class WaitTimeEstimator:
    def __init__(self):
        self.in_payment = dict()  # id -> (last_time, total_time)
        self.count = 0
        self.total_time = 0
        self.payment_area = []
        self.payment_area_mask = None

    def update(self, detections, ids, alive_ids, counted_ids, payment_area, img_w, img_h):
        if self.payment_area != payment_area or self.payment_area_mask is None:
            self.payment_area = payment_area
            self.payment_area_mask = np.zeros((img_h, img_w), 'uint8')
            list_points = []
            for p in payment_area:
                x = p[0] * img_w
                y = p[1] * img_h
                list_points.append((x, y))
            points = np.array([list_points], dtype='int32')
            cv2.fillPoly(self.payment_area_mask, points, (255,))

        for pid, box in zip(ids, detections):
            if pid == -1:  # the detection has not confirmed as a valid track yet
                continue
            if self._is_in_payment_area(box):
                if pid not in self.in_payment.keys():  # first time
                    self.in_payment[pid] = [time.time(), 0]
                else:  # next time
                    now = time.time()
                    last_time = self.in_payment[pid][0]
                    if last_time is not None:
                        self.in_payment[pid][1] += now - last_time
                    self.in_payment[pid][0] = now
            else:  # not in the area
                if pid in self.in_payment.keys():
                    self.in_payment[pid][0] = None

        tracking_ids = list(self.in_payment.keys())
        for pid in tracking_ids:
            if pid not in ids:  # not found
                if pid not in alive_ids and pid in counted_ids:  # dead and counted
                    self.count += 1
                    self.total_time += self.in_payment[pid][1]
                    del self.in_payment[pid]

    def get_total_wait_time(self):
        return self.total_time

    def get_count(self):
        return self.count

    def get_avg_wait_time(self):
        return self.total_time / self.count

    def reset_all(self):
        self.in_payment = dict()
        self.count = 0
        self.total_time = 0

    def reset_count(self):
        self.count = 0
        self.total_time = 0

    def _is_in_payment_area(self, box):
        l, t, r, b = box
        row = int((t + b) / 2)
        col = int((l + r) / 2)
        if self.payment_area_mask[row, col] > 0:
            return True
        else:
            return False
