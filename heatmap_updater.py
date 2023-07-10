import datetime
import os
import tempfile
import threading
import time
import urllib.parse

import cv2
import requests

import config_main
import utils_main
from storage import DataStorage

TEMP_DIR = tempfile.gettempdir()


class HeatmapUpdater:

    def __init__(self):
        self.stopped = True
        self.thread = None

    def start(self):
        if self.thread is not None:
            print('HeatmapUpdater: Already running')
            return
        self.stopped = False
        self.thread = threading.Thread(target=self._run_func)
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.thread = None

    def _run_func(self):
        self._create_missing_heatmap()
        now = datetime.datetime.now()
        last_time = (now.year, now.month, now.day)
        while self.stopped is False:
            time.sleep(1)
            now = datetime.datetime.now()
            curr_time = (now.year, now.month, now.day)
            if curr_time != last_time:
                self._create_missing_heatmap()
                last_time = curr_time

    def _create_missing_heatmap(self):
        storage = DataStorage()
        header = dict()
        header['Store-Id'] = config_main.data['STORE_ID']
        header['Store-Api-Key'] = config_main.data['API_KEY']
        post_url = urllib.parse.urljoin(config_main.data['API_URL'], 'api/store-batch/heatmap')
        # daily heatmap
        this_day = datetime.datetime.now()
        oneday = datetime.timedelta(days=1)
        for i in range(1, 32):
            this_day = this_day - oneday
            year = this_day.year
            month = this_day.month
            day = this_day.day
            hmap_data = storage.get_heatmap_day(year, month, day)
            if len(hmap_data) == 0:
                day_data = storage.get_day_data(year, month, day)
                if len(day_data) == 0:
                    continue
                hmap = None
                count = 0
                for row in day_data:
                    if row[-2] is not None:
                        if hmap is None:
                            hmap = row[-2]
                            count += 1
                        else:
                            hmap += row[-2]
                            count += 1
                if count != 0 and hmap is not None:
                    hmap = hmap / count
                    storage.set_heatmap_day(year, month, day, hmap)
                    hmap_min = hmap.min()
                    hmap_max = hmap.max()
                    hmap = (hmap - hmap_min) / (hmap_max - hmap_min)
                    frame = day_data[-1][-1]
                    send_img = utils_main.blend_heatmap(frame, hmap, 1)
                    filename = os.path.join(TEMP_DIR, 'heatmap_day.jpg')
                    cv2.imwrite(filename, send_img)
                    data_to_send = dict()
                    data_to_send['year'] = year
                    data_to_send['month'] = month
                    data_to_send['day'] = day
                    with open(filename, 'rb') as fi:
                        files = {'heatmap_img': fi}
                        r = requests.post(post_url, data=data_to_send, headers=header, files=files, timeout=30)
                        if r.status_code >= 300:
                            print('Send heatmap to server. Error code:', r.status_code)
                        else:
                            print('Send heatmap to server: Done. Status code:', r.status_code)
                else:
                    storage.set_heatmap_day(year, month, day, None)
        # monthly heatmap
        this_day = datetime.datetime.now()
        year = this_day.year
        month = this_day.month
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        hmap_data = storage.get_heatmap_month(year, month)
        if len(hmap_data) == 0:
            month_data = storage.get_month_statistic(year, month)
            if len(month_data) != 0:
                count = 0
                hmap = None
                for row in month_data:
                    day = row[0]
                    day_hmap = storage.get_heatmap_day(year, month, day)
                    if len(day_hmap) > 0 and day_hmap[0][0] is not None:
                        day_hmap = day_hmap[0][0]
                        if hmap is None:
                            hmap = day_hmap
                            count += 1
                        else:
                            hmap += day_hmap
                            count += 1
                if count != 0 and hmap is not None:
                    hmap = hmap / count
                    storage.set_heatmap_month(year, month, hmap)
                    hmap_min = hmap.min()
                    hmap_max = hmap.max()
                    hmap = (hmap - hmap_min) / (hmap_max - hmap_min)
                    # find a frame to blend with heatmap
                    frame = None
                    for i in range(1, 32):
                        this_day = this_day - oneday
                        year_ = this_day.year
                        month_ = this_day.month
                        day_ = this_day.day
                        day_data = storage.get_day_data(year_, month_, day_)
                        for row in day_data:
                            if row[-1] is not None:
                                frame = row[-1]
                                break
                        if frame is not None:
                            break
                    if frame is not None:
                        send_img = utils_main.blend_heatmap(frame, hmap, 1)
                        filename = os.path.join(TEMP_DIR, 'heatmap_month.jpg')
                        cv2.imwrite(filename, send_img)
                        data_to_send = dict()
                        data_to_send['year'] = year
                        data_to_send['month'] = month
                        with open(filename, 'rb') as fi:
                            files = {'heatmap_img': fi}
                            r = requests.post(post_url, data=data_to_send, headers=header, files=files, timeout=30)
                            if r.status_code >= 300:
                                print('Send heatmap to server. Error code:', r.status_code)
                            else:
                                print('Send heatmap to server: Done. Status code:', r.status_code)
                else:
                    storage.set_heatmap_month(year, month, None)
