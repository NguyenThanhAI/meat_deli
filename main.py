import datetime
import os
import pickle
import random
import sys
import tempfile
import time
from calendar import monthrange

import cv2
import numpy as np
import xlsxwriter
from PyQt5 import QtChart, QtCore, QtGui, QtWidgets, uic

import config_main
import utils_main
from face_age_gender.age_gender_estimator import AgeGenderEstimator
from faceid import FaceIDManager
from heatmap import HeatMap
from heatmap_updater import HeatmapUpdater
from human_detection import HumanDetector
from mark_payment_area import MarkAreaWindow
from motion_detection import MotionDetection
from retina_face_detector.face_detector import RetinaFaceDetector
from storage import DataStorage
from storage_update import StorageUpdater
# from tracker_2 import Tracker
from tracker_3 import SimpleTracker
from videostream import QueuedStream

# from wait_time_estimate import WaitTimeEstimator

# Do not run performance tests to find the best convolution algorithm
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

storage_updater = StorageUpdater()
heatmap_updater = HeatmapUpdater()
heatmap_updater.start()


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        uifile = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'main.ui')
        uic.loadUi(uifile, self)
        self.frame = None
        self.face_frame = None
        self.heatmap = None
        self.storage = DataStorage()
        self.human_process = HumanProcess()
        self.human_process.updateFrame.connect(self.updateFrame)
        self.human_process.updateHeatmap.connect(self.updateHeatmap)
        self.human_process.finished.connect(self.human_process_finished)
        self.face_process = FaceProcess()
        self.face_process.updateFrame.connect(self.updateFaceFrame)
        self.face_process.finished.connect(self.face_process_finished)
        self.face_process.updateFaceID.connect(self.updateFaceID)
        self.camera_video_label.setText('No information')
        self.face_video_label.setText('No information')

        self.day_calendar_widget.selectionChanged.connect(self.load_day_data)
        self.tabWidget.currentChanged.connect(self.changedTab)

        self.month_comboBox.currentIndexChanged.connect(self.load_month_data)
        self.year_comboBox.currentIndexChanged.connect(self.load_month_data)

        self.exportButton.pressed.connect(self.export_excel)

        self.day_calendar_widget.setSelectedDate(QtCore.QDate.currentDate())
        self._set_month_combobox()

        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

    def closeEvent(self, event):
        self.stop()
        event.accept()

    @QtCore.pyqtSlot(np.ndarray)
    def updateFrame(self, frame):
        self.frame = frame
        if self.tabWidget.currentIndex() == 0:
            self._set_video_frame()
            self.update()

    @QtCore.pyqtSlot(np.ndarray)
    def updateFaceFrame(self, frame):
        self.face_frame = frame
        if self.tabWidget.currentIndex() == 1:
            img_h, img_w = self.face_frame.shape[:2]
            qimg = QtGui.QImage(self.face_frame, img_w, img_h, img_w * 3,
                                QtGui.QImage.Format_RGB888).rgbSwapped()
            self.face_video_label.setPixmap(QtGui.QPixmap(qimg))
            self.update()

    @QtCore.pyqtSlot(np.ndarray)
    def updateHeatmap(self, hmap):
        self.heatmap = hmap

    @QtCore.pyqtSlot()
    def human_process_finished(self):
        self.camera_video_label.clear()
        self.camera_video_label.setText('No information')
        self.update()

    @QtCore.pyqtSlot()
    def face_process_finished(self):
        self.face_video_label.clear()
        self.face_video_label.setText('No information')
        self.update()

    def start(self):
        self.human_process.start()
        self.face_process.start()

    def stop(self):
        self.human_process.stop()
        self.human_process.wait()
        self.face_process.stop()
        self.face_process.wait()
        self.frame = None
        self.heatmap = None
        storage_updater.flush()
        heatmap_updater.stop()

    def _set_video_frame(self):
        img_h, img_w = self.frame.shape[:2]
        if self.camera_heatmap_checkBox.isChecked() and self.heatmap is not None:
            power = self.camera_heatmap_slider.value()
            power = (21 - power) / 11
            out = utils_main.blend_heatmap(self.frame, self.heatmap, power)
        else:
            out = self.frame
        qimg = QtGui.QImage(out, img_w, img_h, img_w * 3,
                            QtGui.QImage.Format_RGB888).rgbSwapped()
        self.camera_video_label.setPixmap(QtGui.QPixmap(qimg))

    @QtCore.pyqtSlot()
    def load_day_data(self):
        selected_date = self.day_calendar_widget.selectedDate()
        year = selected_date.year()
        month = selected_date.month()
        day = selected_date.day()
        data = self.storage.get_day_data(year, month, day)
        if len(data) == 0:
            self.day_statistic_label.setText('No information')
            self.day_chart_view.setChart(QtChart.QChart())
        else:
            total_count = 0
            total_wait_time = 0
            total_stay_time = 0
            max_count = 0
            max_wait_time = 0
            max_stay_time = 0
            counts = [0 for _ in range(24)]
            wait_time = [0 for _ in range(24)]
            stay_time = [0 for _ in range(24)]
            self.heatmap_hour = [None for _ in range(24)]
            self.frame_hour = [None for _ in range(24)]
            for row in data:
                h = row[3]
                c = row[4]
                w = row[5]
                s = row[6]
                counts[h] = c
                wait_time[h] = w
                stay_time[h] = s
                self.heatmap_hour[h] = row[13]
                self.frame_hour[h] = row[14]
                total_count += c
                total_wait_time += c * w
                total_stay_time += c * s
                max_count = max(max_count, c)
                max_wait_time = max(max_wait_time, w)
                max_stay_time = max(max_stay_time, s)
            if total_count != 0:
                avg_wait_time = total_wait_time / total_count
                avg_stay_time = total_stay_time / total_count
            else:
                avg_wait_time = 0
                avg_stay_time = 0
            self.day_statistic_label.setText('Tổng khách hàng: ' + str(
                total_count) + '\nThời gian chờ trung bình: ' + str(
                int(avg_wait_time)) + '\nThời gian mua sắm: ' + str(
                int(avg_stay_time)))
            bar_set = QtChart.QBarSet('Số khách hàng')
            for c in counts:
                bar_set.append(c)
            count_series = QtChart.QBarSeries()
            count_series.append(bar_set)
            count_series.hovered.connect(self._set_hour_heatmap)

            wait_time_series = QtChart.QLineSeries()
            wait_time_series.setName('Thời gian chờ')
            for i, t in enumerate(wait_time):
                wait_time_series.append(i, t)

            stay_time_series = QtChart.QLineSeries()
            stay_time_series.setName('Thời gian mua sắm')
            for i, t in enumerate(stay_time):
                stay_time_series.append(i, t)

            chart = QtChart.QChart()
            chart.addSeries(count_series)
            chart.addSeries(wait_time_series)
            chart.addSeries(stay_time_series)
            chart.setTitle(
                'Khách hàng theo từng giờ, thời gian mua hàng và thời gian chờ thanh toán')
            chart.legend().setVisible(True)
            chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)

            axisX = QtChart.QValueAxis()
            axisX.setRange(0, 23)
            axisX.setTickCount(24)
            axisX.setLabelFormat('%d')
            chart.addAxis(axisX, QtCore.Qt.AlignBottom)
            count_series.attachAxis(axisX)

            axisY = QtChart.QValueAxis()
            axisY.setTitleText('Khách')
            axisY.setRange(0, max(50, max_count + 1))
            chart.addAxis(axisY, QtCore.Qt.AlignLeft)
            count_series.attachAxis(axisY)

            axisY = QtChart.QValueAxis()
            axisY.setTitleText('Phút')
            axisY.setRange(0, max(20, max_wait_time + 1, max_stay_time + 1))
            chart.addAxis(axisY, QtCore.Qt.AlignRight)
            wait_time_series.attachAxis(axisY)
            stay_time_series.attachAxis(axisY)

            self.day_chart_view.setChart(chart)
        self.update()

    @QtCore.pyqtSlot(int)
    def changedTab(self, index):
        if index == 2:
            self.load_day_data()
        elif index == 3:
            self.load_month_data()

    def _set_hour_heatmap(self, over, index, barset):
        if over:
            max_width = self.day_heatmap_label.width() - 10
            max_height = self.day_heatmap_label.height() - 10
            heatmap = self.heatmap_hour[index]
            frame = self.frame_hour[index]
            if heatmap is None or frame is None:
                self.day_heatmap_label.clear()
                self.day_heatmap_label.setText('No heatmap data')
            else:
                heatmap = heatmap / heatmap.max()
                frame = utils_main.resize_max_size(frame, max_width, max_height)
                out = utils_main.blend_heatmap(frame, heatmap, 1)
                img_h, img_w = out.shape[:2]
                qimg = QtGui.QImage(out, img_w, img_h, img_w * 3,
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
                self.day_heatmap_label.setPixmap(QtGui.QPixmap(qimg))
        else:
            self.day_heatmap_label.clear()

    def _set_month_combobox(self):
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        list_years = self.storage.get_distinct_years()
        list_years.append(year)
        list_years = sorted(list(set(list_years)))
        for year in list_years:
            self.year_comboBox.addItem(str(year), year)
        self.year_comboBox.setCurrentIndex(list_years.index(year))
        self.month_comboBox.setCurrentIndex(month - 1)

    @QtCore.pyqtSlot()
    def load_month_data(self):
        month = int(self.month_comboBox.currentText())
        year = int(self.year_comboBox.currentText())
        month_statistic = self.storage.get_month_statistic(year, month)
        chart = QtChart.QChart()
        chart_age = QtChart.QChart()
        chart_gender = QtChart.QChart()
        if not month_statistic:
            self.label_4.setText('No information')
            chart.setTitle('No information')
            chart_age.setTitle('No information')
            chart_gender.setTitle('No information')
        else:
            nday = monthrange(year, month)[1]
            total_count = 0
            total_wait_time = 0
            total_stay_time = 0
            max_count = 0
            max_avg_wait_time = 0
            max_avg_stay_time = 0
            counts = [0 for _ in range(nday)]
            wait_time = [0 for _ in range(nday)]
            stay_time = [0 for _ in range(nday)]
            male_count = 0
            female_count = 0
            age_group_1 = 0
            age_group_2 = 0
            age_group_3 = 0
            age_group_4 = 0
            for row in month_statistic:
                row = [f if f else 0 for f in row]
                d = row[0]
                c = row[1]
                w = row[2]
                s = row[3]
                male_count += row[4]
                female_count += row[5]
                age_group_1 += row[6]
                age_group_2 += row[7]
                age_group_3 += row[8]
                age_group_4 += row[9]
                counts[d] = c
                if c != 0:
                    wait_time[d] = w / c
                    stay_time[d] = s / c
                else:
                    wait_time[d] = 0
                    stay_time[d] = 0
                total_count += c
                total_wait_time += w
                total_stay_time += s
                max_count = max(max_count, c)
                max_avg_wait_time = max(max_avg_wait_time, wait_time[d])
                max_avg_stay_time = max(max_avg_stay_time, stay_time[d])
            if total_count != 0:
                avg_wait_time = total_wait_time / total_count
                avg_stay_time = total_stay_time / total_count
            else:
                avg_wait_time = 0
                avg_stay_time = 0
            self.label_4.setText('Tổng khách hàng: ' +
                                 str(total_count) + '. Thời gian chờ trung bình: ' +
                                 str(int(avg_wait_time)) + ' phút' +
                                 '. Thời gian mua sắm trung bình: ' +
                                 str(int(avg_stay_time)) + ' phút')
            bar_set = QtChart.QBarSet('Số  khách hàng')
            for c in counts:
                bar_set.append(c)
            count_series = QtChart.QBarSeries()
            count_series.append(bar_set)

            wait_time_series = QtChart.QLineSeries()
            wait_time_series.setName('Thời gian chờ')
            for i, t in enumerate(wait_time):
                wait_time_series.append(i, t)

            stay_time_series = QtChart.QLineSeries()
            stay_time_series.setName('Thời gian mua')
            for i, t in enumerate(stay_time):
                stay_time_series.append(i, t)

            gender_pie = QtChart.QPieSeries()
            gender_pie.append('Nam', male_count)
            gender_pie.append('Nữ', female_count)

            age_pie = QtChart.QPieSeries()
            age_pie.append('0-25', age_group_1)
            age_pie.append('25-34', age_group_2)
            age_pie.append('35-54', age_group_3)
            age_pie.append('>54', age_group_4)

            chart = QtChart.QChart()
            chart.addSeries(count_series)
            chart.addSeries(wait_time_series)
            chart.addSeries(stay_time_series)
            chart.setTitle(
                'Khách hàng theo từng ngày, thời gian mua sắm và thời gian chờ')
            chart.legend().setVisible(True)
            chart.setAnimationOptions(QtChart.QChart.SeriesAnimations)

            chart_gender = QtChart.QChart()
            chart_gender.addSeries(gender_pie)
            chart_gender.setTitle('Tỉ lệ theo giới tính')
            chart_gender.legend().setVisible(True)
            chart_gender.setAnimationOptions(QtChart.QChart.SeriesAnimations)

            chart_age = QtChart.QChart()
            chart_age.addSeries(age_pie)
            chart_age.setTitle('Tỉ lệ theo nhóm tuổi')
            chart_age.legend().setVisible(True)
            chart_age.setAnimationOptions(QtChart.QChart.SeriesAnimations)

            axisX = QtChart.QValueAxis()
            axisX.setRange(1, nday)
            axisX.setTickCount(nday)
            axisX.setLabelFormat('%d')
            chart.addAxis(axisX, QtCore.Qt.AlignBottom)
            count_series.attachAxis(axisX)

            axisY = QtChart.QValueAxis()
            axisY.setTitleText('Khách')
            axisY.setRange(0, max(500, max_count + 1))
            axisY.setLabelFormat('%d')
            chart.addAxis(axisY, QtCore.Qt.AlignLeft)
            count_series.attachAxis(axisY)

            axisY = QtChart.QValueAxis()
            axisY.setTitleText('Phút')
            axisY.setRange(
                0, max(20, max_avg_wait_time + 1, max_avg_stay_time + 1))
            chart.addAxis(axisY, QtCore.Qt.AlignRight)
            wait_time_series.attachAxis(axisY)
            stay_time_series.attachAxis(axisY)
        self.month_chart_view.setChart(chart)
        self.age_chart_view.setChart(chart_age)
        self.gender_chart_view.setChart(chart_gender)

    @QtCore.pyqtSlot()
    def export_excel(self):
        month = int(self.month_comboBox.currentText())
        year = int(self.year_comboBox.currentText())
        month_statistic = self.storage.get_month_statistic(year, month)
        if not month_statistic:
            message_box = QtWidgets.QMessageBox()
            message_box.setWindowTitle('Thông tin')
            message_box.setText('Không có dữ liệu để trích xuất')
            message_box.exec()
        else:
            nday = monthrange(year, month)[1]
            counts = [None for _ in range(nday)]
            wait_time = [None for _ in range(nday)]
            stay_time = [None for _ in range(nday)]
            male_count = [None for _ in range(nday)]
            female_count = [None for _ in range(nday)]
            age_group_1 = [None for _ in range(nday)]
            age_group_2 = [None for _ in range(nday)]
            age_group_3 = [None for _ in range(nday)]
            age_group_4 = [None for _ in range(nday)]
            for row in month_statistic:
                d = row[0]
                c = row[1]
                w = row[2]
                s = row[3]
                counts[d] = c
                if c != 0:
                    wait_time[d] = w / c
                    stay_time[d] = s / c
                else:
                    wait_time[d] = 0
                    stay_time[d] = 0
                male_count[d] = row[4]
                female_count[d] = row[5]
                age_group_1[d] = row[6]
                age_group_2[d] = row[7]
                age_group_3[d] = row[8]
                age_group_4[d] = row[9]
            fileName, ftype = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save as", os.path.expanduser('~'), "Excel 2007 (*.xlsx *.XLSX)")
            if fileName:
                name, ext = os.path.splitext(fileName)
                if not ext:
                    ext = '.xlsx'
                fileName = name + ext
                workbook = xlsxwriter.Workbook(fileName)
                format = workbook.add_format()
                format.set_font_size(10)
                worksheet = workbook.add_worksheet(
                    "Thống kê tháng " + str(month) + "-" + str(year))
                worksheet.write(0, 0, 'Ngày', format)
                worksheet.write(0, 1, 'Lượng khách', format)
                worksheet.write(0, 2, 'Thời gian chờ (phút)', format)
                worksheet.write(0, 3, 'Thời gian mua (phút)', format)
                worksheet.write(0, 4, 'Số khách nam', format)
                worksheet.write(0, 5, 'Số khách nữ', format)
                worksheet.write(0, 6, '0-25 tuổi', format)
                worksheet.write(0, 7, '25-34 tuổi', format)
                worksheet.write(0, 8, '35-54 tuổi', format)
                worksheet.write(0, 9, 'Từ 55 tuổi', format)
                for i in range(len(counts)):
                    worksheet.write(i + 1, 0, str(i+1) + '/' +
                                    str(month) + '/' + str(year), format)
                    worksheet.write(i + 1, 1, counts[i], format)
                    worksheet.write(i + 1, 2, wait_time[i], format)
                    worksheet.write(i + 1, 3, stay_time[i], format)
                    worksheet.write(i + 1, 4, male_count[i], format)
                    worksheet.write(i + 1, 5, female_count[i], format)
                    worksheet.write(i + 1, 6, age_group_1[i], format)
                    worksheet.write(i + 1, 7, age_group_2[i], format)
                    worksheet.write(i + 1, 8, age_group_3[i], format)
                    worksheet.write(i + 1, 9, age_group_4[i], format)
                # sheet2
                worksheet_2 = workbook.add_worksheet('Charts')
                # month chart
                img = QtGui.QImage(1600, 400, QtGui.QImage.Format_RGB888)
                img.fill(QtCore.Qt.white)
                self.month_chart_view.render(QtGui.QPainter(img))
                pathname = os.path.join(tempfile.gettempdir(), 'chart.jpg')
                img.save(pathname)
                worksheet_2.insert_image('A1', pathname)
                # age chart
                img = QtGui.QImage(600, 500, QtGui.QImage.Format_RGB888)
                img.fill(QtCore.Qt.white)
                self.age_chart_view.render(QtGui.QPainter(img))
                pathname = os.path.join(tempfile.gettempdir(), 'chart_2.jpg')
                img.save(pathname)
                worksheet_2.insert_image('A30', pathname)
                # gender chart
                img = QtGui.QImage(600, 500, QtGui.QImage.Format_RGB888)
                img.fill(QtCore.Qt.white)
                self.gender_chart_view.render(QtGui.QPainter(img))
                pathname = os.path.join(tempfile.gettempdir(), 'chart_3.jpg')
                img.save(pathname)
                worksheet_2.insert_image('K30', pathname)
                workbook.close()

    @QtCore.pyqtSlot()
    def on_pick_area_button_clicked(self):
        # we should NOT access human_process's attribute this way, but I am in hurry ;( TODO
        area_window = MarkAreaWindow(
            self.frame, self.human_process.payment_area, self)
        area_window.update_list_points.connect(
            self.human_process.set_payment_area)
        area_window.open()

    @QtCore.pyqtSlot(int, np.ndarray, float, int)
    def updateFaceID(self, cid, face, timestamp, count):
        print('Customer ID:', cid)
        self.tableWidget.removeRow(49)
        self.tableWidget.insertRow(0)
        self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem(str(cid)))
        face = cv2.resize(face, (95, 95))
        row, col = face.shape[:2]
        qimg = QtGui.QImage(face, col, row, col * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
        img_item = QtWidgets.QTableWidgetItem()
        img_item.setData(QtCore.Qt.DecorationRole, QtGui.QPixmap(qimg))
        self.tableWidget.setItem(0, 1, img_item)
        datetime_dt = datetime.datetime.fromtimestamp(timestamp)
        self.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem(str(datetime_dt)))
        self.tableWidget.setItem(0, 3, QtWidgets.QTableWidgetItem(str(count)))


class HumanProcess(QtCore.QThread):
    updateFrame = QtCore.pyqtSignal(np.ndarray)
    updateHeatmap = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(HumanProcess, self).__init__()
        self.stopped = False
        self.payment_area = []
        path = os.path.join(config_main.DATABASE_DIR, 'payment_area.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as fi:
                self.payment_area = pickle.load(fi)

    def run(self):
        video_uri = config_main.data['VIDEO_URI']
        # object detector, that run on a different process
        if config_main.data['HUMAN_DETECTION'] is True:
            human_detector = HumanDetector()
            human_detector.start()
        else:
            human_detector = None
        # motion detection, we need this to save computing power when the scene is static
        motion_detector = MotionDetection()
        # heatmap
        heatmap = HeatMap()
        # tracker
        # tracker = Tracker()
        # wait time estimator
        # wait_time_estimator = WaitTimeEstimator()
        # video stream object, that run on a different thread
        video = QueuedStream(video_uri, True, 25)
        video.start()
        if not video.isOpened():
            print("Can not open video")
            return
        # read some config
        max_process_w = config_main.data['PROCESS_WIDTH']
        max_process_h = config_main.data['PROCESS_HEIGHT']
        max_display_w = config_main.data['DISPLAY_WIDTH']
        max_display_h = config_main.data['DISPLAY_HEIGHT']
        #
        now = datetime.datetime.now()
        last_time = (now.year, now.month, now.day, now.hour)
        last_heatmap = None
        last_heatmap_count = 0
        # processing loop
        while not self.stopped:
            ret, frame, frame_id = video.read()
            # time_stamp = time.time()
            if not ret:
                break

            frame_process = utils_main.resize_max_size(
                frame, max_process_w, max_process_h)

            # scale_row = frame.shape[0] / frame_process.shape[0]
            # scale_col = frame.shape[1] / frame_process.shape[1]

            if len(motion_detector.get_motion_region(frame_process)) > 0:
                # put the current frame to the tracking list
                # tracker.update_frame(frame_id, time_stamp, frame)
                # put the current frame to the detection queue, discard all other frames that are currently in the queue
                if config_main.data['HUMAN_DETECTION'] is True:
                    human_detector.put_frame(frame_id, frame_process)
                    # get detection result of the old frame if result is available
                    # detection_result = human_detector.get_result(block=False)
                    detection_result = human_detector.get_result(block=False)
                else:
                    detection_result = None
                if detection_result is not None:
                    frame_id, _, _, boxes = detection_result
                    # update heatmap
                    heatmap.update(boxes, frame_process.shape[:2])
                    # update tracker
                    # boxes_correct_size = []
                    # for l, t, r, b in boxes:
                    #     l = l * scale_col
                    #     t = t * scale_row
                    #     r = r * scale_col
                    #     b = b * scale_row
                    #     boxes_correct_size.append((l, t, r, b))
                    # ids, counted_ids = tracker.update_detection_result(frame_id, boxes_correct_size)
                    # update wait_time_estimator
                    # wait_time_estimator.update(boxes, ids, tracker.alive_ids(), counted_ids, self.payment_area, img_w, img_h)

                # if frame is not None:
                if frame is not None:
                    frame_display = utils_main.resize_max_size(frame, max_display_w, max_display_h)
                    self.updateFrame.emit(frame_display)
                hmap = heatmap.get_heatmap()
                if hmap is not None:
                    hmap = utils_main.resize_max_size(hmap, max_display_w, max_display_h)
                    self.updateHeatmap.emit(hmap)
            # save data to database if needed
            now = datetime.datetime.now()
            current_time = (now.year, now.month, now.day, now.hour)
            if last_time != current_time:
                hmap = heatmap.heatmap
                count = heatmap.count
                if last_heatmap is not None:
                    diff_hmap = hmap - last_heatmap
                    diff_count = count - last_heatmap_count
                else:
                    diff_hmap = hmap
                    diff_count = count
                if diff_count != 0:
                    hour_heatmap = diff_hmap / diff_count
                else:
                    hour_heatmap = None
                data = dict()
                data['heatmap'] = hour_heatmap
                data['frame'] = frame
                storage_updater.update(*last_time, data)
                last_heatmap = hmap
                last_heatmap_count = count
                # reset heatmap if need
                if last_time[2] != current_time[2]:  # differ in date
                    heatmap.reset()
                last_time = current_time

        video.release()
        if human_detector is not None:
            human_detector.stop()

    def stop(self):
        self.stopped = True

    @QtCore.pyqtSlot(list)
    def set_payment_area(self, list_points):
        self.payment_area = list_points
        path = os.path.join(config_main.DATABASE_DIR, 'payment_area.pkl')
        with open(path, 'wb') as fo:
            pickle.dump(self.payment_area, fo)


class FaceProcess(QtCore.QThread):
    updateFrame = QtCore.pyqtSignal(np.ndarray)
    updateFaceID = QtCore.pyqtSignal(int, np.ndarray, float, int)

    def __init__(self):
        super(FaceProcess, self).__init__()
        self.stopped = False

    def run(self):
        video_uri = config_main.data['VIDEO_URI_FACE']
        face_detector = RetinaFaceDetector(gpuid=config_main.data['FACE_DETECTION_GPU'])
        ag_estimator = AgeGenderEstimator(gpuid=config_main.data['AGE_GENDER_GPU'])
        # motion detection, we need this to save computing power when the scene is static
        motion_detector = MotionDetection()
        # tracker
        tracker = SimpleTracker()
        # faceid manager
        faceid = FaceIDManager()
        faceid.updateFaceID.connect(self.updateFaceID)
        faceid.start()
        # video stream object, that run on a different thread
        video = QueuedStream(video_uri, True, 25)
        video.start()
        if not video.isOpened():
            print("Can not open video")
            return
        # read some config
        max_process_w = config_main.data['PROCESS_WIDTH_FACE']
        max_process_h = config_main.data['PROCESS_HEIGHT_FACE']
        max_display_w = config_main.data['DISPLAY_WIDTH_FACE']
        max_display_h = config_main.data['DISPLAY_HEIGHT_FACE']
        #
        now = datetime.datetime.now()
        last_time = (now.year, now.month, now.day, now.hour)
        count = 0
        male_count = 0
        female_count = 0
        age1_count = 0
        age2_count = 0
        age3_count = 0
        age4_count = 0
        tracking_list = {}
        # processing loop
        while not self.stopped:
            ret, frame, frame_id = video.read()
            if not ret:
                break

            frame_process = utils_main.resize_max_size(
                frame, max_process_w, max_process_h)

            if len(motion_detector.get_motion_region(frame_process)) > 0:
                detection_result = face_detector.detect(frame_process)
                if detection_result is not None:
                    recs, points = detection_result
                    # ignore small face
                    next_recs = []
                    next_points = []
                    for rec, p in zip(recs, points):
                        l, t, r, b = rec[:4]
                        if (b - t + r - l) / 2 > config_main.data['MIN_FACE_SIZE']:
                            next_recs.append(rec)
                            next_points.append(p)
                    recs = np.array(next_recs)
                    points = np.array(next_points)

                    # predict age and gender
                    list_age = []
                    list_gender = []
                    for i in range(len(recs)):
                        g, a = ag_estimator.predict(frame_process, recs[i], points[i])
                        a += 5  # Asian guys alway look young =))
                        list_age.append(a)
                        list_gender.append(g)
                    # update tracker
                    recs = recs.astype('int')
                    if len(recs) > 0:
                        recs = recs[:, :4]  # ignore score column
                    points = points.astype('int')
                    ids, counted_ids = tracker.update(recs)
                    # update count
                    count += len(counted_ids)
                    for i in counted_ids:
                        print('Tracking ID go out of scene:', i)
                        # update age count
                        avg_age = sum(
                            tracking_list[i]['age']) / len(tracking_list[i]['age'])
                        if avg_age < 25:
                            age1_count += 1
                        elif avg_age < 35:
                            age2_count += 1
                        elif avg_age < 55:
                            age3_count += 1
                        else:
                            age4_count += 1
                        # update gender count
                        avg_gender = sum(
                            tracking_list[i]['gender']) / len(tracking_list[i]['gender'])
                        if avg_gender < 0.5:
                            female_count += 1
                        else:
                            male_count += 1
                        # pass the data to the faceid manager
                        if len(tracking_list[i]['face']) > 5:
                            tracking_list[i]['face'] = random.sample(tracking_list[i]['face'], 5)
                        faceid.put_data(tracking_list[i])
                    # add to tracking_list
                    for i, a, g, rec, point in zip(ids, list_age, list_gender, recs, points):
                        if i not in tracking_list.keys():
                            tracking_list[i] = dict()
                            tracking_list[i]['timestamp'] = time.time()
                            tracking_list[i]['age'] = []
                            tracking_list[i]['gender'] = []
                            tracking_list[i]['face'] = []
                        tracking_list[i]['age'].append(a)
                        tracking_list[i]['gender'].append(g)
                        aligned_face = faceid.get_aligned_face(frame_process, rec, point)
                        tracking_list[i]['face'].append(aligned_face)
                        if len(tracking_list[i]['face']) > 20:
                            del tracking_list[i]['face'][0]
                            # del tracking_list[i]['age'][0]
                            # del tracking_list[i]['gender'][0]
                    # remove dead id
                    alives = tracker.alive_ids()
                    keys = list(tracking_list.keys())
                    for k in keys:
                        if k not in alives:
                            del tracking_list[k]

                    if frame is not None:
                        frame_display = utils_main.resize_max_size(frame, max_display_w, max_display_h)
                        if len(recs) > 0:
                            scale_row = frame_display.shape[0] / frame_process.shape[0]
                            scale_col = frame_display.shape[1] / frame_process.shape[1]
                            for pid, box in zip(ids, recs):
                                l, t, r, b = box
                                l = int(l * scale_col)
                                t = int(t * scale_row)
                                r = int(r * scale_col)
                                b = int(b * scale_row)
                                cv2.rectangle(frame_display, (l, t), (r, b), (0, 0, 255), 2)
                                # utils_main.putTextLabel(frame_display, (l, t), str(pid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 255), thickness=1, bottom=True)
                        self.updateFrame.emit(frame_display)
            # save data if needed
            now = datetime.datetime.now()
            current_time = (now.year, now.month, now.day, now.hour)
            if last_time != current_time:
                data = dict()
                data['count'] = count
                data['wait_time'] = 0
                data['stay_time'] = 0
                data['male'] = male_count
                data['female'] = female_count
                data['age1'] = age1_count
                data['age2'] = age2_count
                data['age3'] = age3_count
                data['age4'] = age4_count
                storage_updater.update(*last_time, data)
                last_time = current_time
                count = 0
                male_count = 0
                female_count = 0
                age1_count = 0
                age2_count = 0
                age3_count = 0
                age4_count = 0
        faceid.stop()
        video.release()

    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.start()
    widget.show()
    sys.exit(app.exec_())
