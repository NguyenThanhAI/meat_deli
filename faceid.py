import datetime
import os
import queue
import random
import tempfile
import threading
import urllib.parse

import cv2
import numpy as np
import requests
from PyQt5 import QtCore

import config_main
from arcface_embedding.arcface_embedding import ArcfaceEmbedding
from storage import DataStorage

TEMP_DIR = tempfile.gettempdir()
MAX_FACE = 5


class FaceIDManager(QtCore.QObject):
    """Query for FaceID and send update to the server. Yes, I should not design this class like this, but I am in hurry, sorry :(("""
    # TODO Re-design this class
    updateFaceID = QtCore.pyqtSignal(int, np.ndarray, float, int)

    def __init__(self):
        super(FaceIDManager, self).__init__()
        self.emb_model = ArcfaceEmbedding()
        self.in_queue = queue.Queue(maxsize=500)
        self.stopped = False
        self.th = None

    def start(self):
        self.stopped = False
        self.th = threading.Thread(target=self._run)
        self.th.start()

    def stop(self):
        self.stopped = True
        if self.th is not None:
            self.th.join()
            self.th = None

    def get_aligned_face(self, img, bbox, points):
        return self.emb_model._get_input(img, bbox, points)

    def put_data(self, data):
        try:
            self.in_queue.put(data, block=False)
        except Exception:
            pass

    def _run(self):
        storage = DataStorage()
        while self.stopped is False:
            try:
                data = self.in_queue.get(block=True, timeout=0.25)
                if len(data['face']) > 5:
                    face_crop = random.sample(data['face'], 5)
                else:
                    face_crop = data['face']
                vectors = np.zeros((len(face_crop), 512))
                for i, img in enumerate(face_crop):
                    emb = self.emb_model._get_feature(img)
                    vectors[i, :] = emb
                post_url = urllib.parse.urljoin(config_main.data['FACEID_API_URL'], 'faceid')
                data_to_send = {'vectors': vectors.tolist()}
                print('Request FaceID')
                r = requests.post(post_url, json=data_to_send, timeout=10)
                print('Request FaceID DONE')
                if r.status_code != 200:
                    print('Error calling FaceID API:', r.status_code)
                else:
                    ret_data = r.json()
                    cid = ret_data['cid']
                    old_data = storage.get_from_customer_data(cid)
                    if len(old_data) > 0:
                        _, age, gender, count = old_data[0]
                        age *= count
                        gender *= count
                        avg_age = sum(data['age']) / len(data['age'])
                        avg_gender = sum(data['gender']) / len(data['gender'])
                        count += 1
                        age = (age + avg_age) / count
                        gender = (gender + avg_gender) / count
                    else:
                        age = sum(data['age']) / len(data['age'])
                        gender = sum(data['gender']) / len(data['gender'])
                        count = 1
                    storage.insert_into_customer_data(cid, age, gender, count)
                    storage.insert_into_visit_data(cid, data['timestamp'])
                    avatar = np.transpose(face_crop[0], (1, 2, 0))
                    avatar = cv2.cvtColor(avatar, cv2.COLOR_RGB2BGR)
                    self.updateFaceID.emit(cid, avatar, data['timestamp'], count)
                    # send customer data to server
                    header = dict()
                    header['Store-Id'] = config_main.data['STORE_ID']
                    header['Store-Api-Key'] = config_main.data['API_KEY']
                    post_url = urllib.parse.urljoin(config_main.data['API_URL'], 'api/store-batch/customer')
                    data_to_send = dict()
                    data_to_send['customer_code'] = cid
                    data_to_send['sex'] = 0 if gender > 0.5 else 1
                    data_to_send['age'] = int(age)
                    avatar_filename = os.path.join(TEMP_DIR, 'avatar.jpg')
                    cv2.imwrite(avatar_filename, avatar)
                    with open(avatar_filename, 'rb') as fi:
                        files = {'avatar': fi}
                        r = requests.post(post_url, data=data_to_send, headers=header, files=files, timeout=30)
                        if r.status_code >= 300:
                            print('Send customer data to server. Error code:', r.status_code)
                        else:
                            print('Send customer data to server: Done. Status code:', r.status_code)
                    # send visit data to server
                    post_url = urllib.parse.urljoin(config_main.data['API_URL'], 'api/store-batch/customer-data')
                    data_to_send = dict()
                    data_to_send['customer_code'] = cid
                    data_to_send['visited_time'] = datetime.datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    r = requests.post(post_url, data=data_to_send, headers=header, timeout=30)
                    if r.status_code >= 300:
                        print('Send visit data to server. Error code:', r.status_code)
                    else:
                        print('Send visit data to server: Done. Status code:', r.status_code)
            except queue.Empty:
                pass
            except Exception as e:
                print(e)
