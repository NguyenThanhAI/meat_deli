import os
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import numpy as np
import cv2


class MarkAreaWindow(QtWidgets.QDialog):
    update_list_points = QtCore.pyqtSignal(list)

    def __init__(self, img, list_points, parent=None):
        super(MarkAreaWindow, self).__init__(parent, QtCore.Qt.Window)
        uifile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'area.ui')
        uic.loadUi(uifile, self)
        self.img = img
        img_h, img_w = img.shape[:2]
        self.list_points = []
        for x, y in list_points:
            self.list_points.append((x * img_w, y * img_h))
        self.draw_points()

    def mousePressEvent(self, e):
        x = e.x()
        y = e.y()
        x_label = self.label.x()
        y_label = self.label.y()
        x = x - x_label
        y = y - y_label
        self.list_points.append((x, y))
        self.draw_points()
        self.update()

    def draw_points(self):
        bg = np.zeros_like(self.img, dtype='uint8')
        if len(self.list_points) >= 3:
            points = np.array([self.list_points], dtype='int32')
            cv2.fillPoly(bg, points, (0, 255, 255))
        img = (0.5 * bg + 0.5 * self.img).astype('uint8')
        for p in self.list_points:
            x = int(p[0])
            y = int(p[1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        img_h, img_w = img.shape[:2]
        qimg = QtGui.QImage(img, img_w, img_h, img_w * 3,
                            QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap(qimg))

    @QtCore.pyqtSlot()
    def on_clear_button_clicked(self):
        self.list_points = []
        self.draw_points()

    @QtCore.pyqtSlot()
    def on_cancel_button_clicked(self):
        self.close()

    @QtCore.pyqtSlot()
    def on_ok_button_clicked(self):
        img_h, img_w = self.img.shape[:2]
        ret = []
        for x, y in self.list_points:
            ret.append((x / img_w, y / img_h))
        self.update_list_points.emit(ret)
        self.close()
