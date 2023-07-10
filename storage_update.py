import os
import tempfile
import threading
import traceback
import urllib.parse

import cv2
import numpy as np
import requests

import config_main
from heatmap import RATIO
from storage import DataStorage

TEMP_DIR = tempfile.gettempdir()


class StorageUpdater:
    def __init__(self):
        self.data_save = dict()
        self.data_send = dict()
        self.save_thread = None
        self.send_thread = None
        self.save_data_lock = threading.Lock()
        self.send_data_lock = threading.Lock()

    def update(self, year, month, day, hour, data):
        key = (year, month, day, hour)

        self.save_data_lock.acquire()
        if key not in self.data_save:
            self.data_save[key] = dict()
        self.data_save[key].update(data)
        self.save_data_lock.release()
        self._save_data()

        self.send_data_lock.acquire()
        if key not in self.data_send:
            self.data_send[key] = dict()
        self.data_send[key].update(data)
        self.send_data_lock.release()
        self._check_and_send()

    def flush(self):
        self._save_data()
        self._send_data()
        if self.save_thread is not None and self.save_thread.is_alive():
            self.save_thread.join()
        if self.send_thread is not None and self.send_thread.is_alive():
            self.send_thread.join()

    def _save_data(self):
        data_copy = dict()
        self.save_data_lock.acquire()
        for key in list(self.data_save.keys()):
            if len(self.data_save[key]) == 11:
                data_copy[key] = self.data_save[key]
                del self.data_save[key]
        self.save_data_lock.release()
        if len(data_copy) > 0:
            if (self.save_thread is None or self.save_thread.is_alive() is False):
                self.save_thread = threading.Thread(target=StorageUpdater._save_thread, args=(data_copy,))
                self.save_thread.start()
            else:
                # put data back
                self.save_data_lock.acquire()
                for key in list(data_copy.keys()):
                    self.data_save[key] = data_copy[key]
                self.save_data_lock.release()

    def _check_and_send(self):
        self.send_data_lock.acquire()
        valid_count = 0
        for key in list(self.data_send.keys()):
            if len(self.data_send[key]) == 11:
                valid_count += 1
        self.send_data_lock.release()
        if valid_count >= config_main.data['SEND_DATA_INTERVAL']:
            self._send_data()

    def _send_data(self):
        data_copy = dict()
        self.send_data_lock.acquire()
        for key in list(self.data_send.keys()):
            if len(self.data_send[key]) == 11:
                data_copy[key] = self.data_send[key]
                del self.data_send[key]
        self.send_data_lock.release()
        if len(data_copy) > 0:
            if (self.send_thread is None or self.send_thread.is_alive() is False):
                self.send_thread = threading.Thread(target=StorageUpdater._send_thread, args=(data_copy,))
                self.send_thread.start()
            else:
                # put data back
                self.send_data_lock.acquire()
                for key in list(data_copy.keys()):
                    self.data_send[key] = data_copy[key]
                self.send_data_lock.release()

    @staticmethod
    def _save_thread(data):
        storage = DataStorage()
        for key in list(data.keys()):
            storage.insert_into_hour_data(*key, data[key])

    @staticmethod
    def _send_thread(data):
        try:
            header = dict()
            header['Store-Id'] = config_main.data['STORE_ID']
            header['Store-Api-Key'] = config_main.data['API_KEY']
            post_url = urllib.parse.urljoin(config_main.data['API_URL'], 'api/store-batch/hour-data')
            for key in list(data.keys()):
                data_to_send = dict()
                data_to_send['year'] = key[0]
                data_to_send['month'] = key[1]
                data_to_send['day'] = key[2]
                data_to_send['hour'] = key[3]
                # data_to_send['count'] = data[key]['count']  # server-side does not accept this field
                data_to_send['wait_time'] = data[key]['wait_time']
                data_to_send['stay_time'] = data[key]['stay_time']
                data_to_send['male'] = data[key]['male']
                data_to_send['female'] = data[key]['female']
                data_to_send['age_group_1'] = data[key]['age1']
                data_to_send['age_group_2'] = data[key]['age2']
                data_to_send['age_group_3'] = data[key]['age3']
                data_to_send['age_group_4'] = data[key]['age4']
                # save frame
                frame_file_name = os.path.join(TEMP_DIR, 'frame.jpg')
                cv2.imwrite(frame_file_name, data[key]['frame'])
                # save heatmap as a image
                if data[key]['heatmap'] is None:
                    nrow, ncol = data[key]['frame'].shape[:2]
                    map_row = int(nrow * RATIO)
                    map_col = int(ncol * RATIO)
                    data[key]['heatmap'] = np.zeros((map_row, map_col))
                heatmap = data[key]['heatmap'] * 255
                heatmap = heatmap.astype('uint8')
                heatmap_file_name = os.path.join(TEMP_DIR, 'heatmap.jpg')
                cv2.imwrite(heatmap_file_name, heatmap)
                with open(frame_file_name, 'rb') as f1:
                    with open(heatmap_file_name, 'rb') as f2:
                        files = {'frame_img': f1, 'heatmap_img': f2}
                        r = requests.post(post_url, data=data_to_send, headers=header, files=files, timeout=30)
                        if r.status_code >= 300:
                            print('Send hour data to server. Error code:', r.status_code)
                        else:
                            print('Send hour data to server: Done. Status code:', r.status_code)
        except Exception as e:
            print(e)
            traceback.print_exc()
