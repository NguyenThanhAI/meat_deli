import cv2
import numpy as np

RATIO = 0.2


class HeatMap:
    def __init__(self):
        self.heatmap = None
        self.count = 0

    def update(self, detections, frame_size):
        # detections: locations of people in the frame, [(left, top, right, bottom), ...]
        # frame_size: (row, col)
        nrow, ncol = frame_size
        if self.heatmap is None:
            map_row = int(nrow * RATIO)
            map_col = int(ncol * RATIO)
            self.heatmap = np.zeros((map_row, map_col))
            self.count = 0

        self.count += 1

        for box in detections:
            kernel = HeatMap._create_gaussian_kernel(box)
            krow, kcol = kernel.shape
            l, t, _, _ = box
            l = int(l * RATIO)
            t = int(t * RATIO)
            # may cause overflow, but very unlikely. For a demo, this is fine
            self.heatmap[t:t+krow, l:l+kcol] += kernel

    def get_heatmap(self):
        if self.heatmap is not None and self.count != 0:
            return self.heatmap / self.count
        else:
            return None

    def reset(self):
        self.heatmap = None
        self.count = 0

    @staticmethod
    def _create_gaussian_kernel(box):
        l, t, r, b = box
        h_kernel_size = int((r - l) * RATIO)
        h_kernel_size -= (h_kernel_size + 1) % 2
        v_kernel_size = int((b - t) * RATIO)
        v_kernel_size -= (v_kernel_size + 1) % 2
        h_kernel = cv2.getGaussianKernel(h_kernel_size, 0)
        v_kernel = cv2.getGaussianKernel(v_kernel_size, 0)
        kernel = np.matmul(v_kernel, h_kernel.transpose())
        kernel = kernel / kernel.max()
        return kernel
