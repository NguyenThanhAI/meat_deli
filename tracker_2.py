import os
import random
import sys

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T

import config_main

sys.path.append(os.path.join(config_main.THIS_DIR, 'reid'))

from config import cfg
from data.transforms import build_transforms
from modeling import build_model
from utils.re_ranking import re_ranking

MAX_BUFFER_SIZE = 100
MAX_AGE = 5 * 60  # 5 minutes
MAX_GALLERY_SIZE = 30
THRESHOLD = 0.45
MIN_MATCH_COUNT = 5  # minimum number of match to be considered as a valid id


class Tracker:
    def __init__(self):
        cfg.merge_from_file(os.path.join(config_main.THIS_DIR, 'reid', 'configs', 'softmax_triplet_with_center.yml'))
        cfg.MODEL.DEVICE_ID = '0'
        cfg.TEST.NECK_FEAT = 'after'
        cfg.TEST.FEAT_NORM = 'yes'
        cfg.MODEL.PRETRAIN_CHOICE = 'self'
        cfg.TEST.RE_RANKING = 'yes'
        cfg.TEST.WEIGHT = os.path.join(config_main.THIS_DIR, 'data_main', 'market_resnet50_model_120_rank1_945.pth')
        cfg.freeze()
        if cfg.MODEL.DEVICE == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

        num_classes = 2  # just a boilerplate
        self.model = build_model(cfg, num_classes)
        self.model.load_param(cfg.TEST.WEIGHT)

        self.device = cfg.MODEL.DEVICE
        if self.device:
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
        self.model.eval()
        # save frame to retrieve later when the detection result is available
        self.frame_buffer = dict()
        self.cfg = cfg
        val_transforms = build_transforms(cfg, is_train=False)
        self.val_transforms = T.Compose([T.ToPILImage(), val_transforms])
        self.tracking = dict()
        self.next_id = 0
        self.total_sequence = 0  # total sequence ever live, except that are currently alive
        self.total_live_time = 0

    def update_frame(self, frame_id, time_stamp, frame):
        '''Add a new frame to the tracking list'''
        self.frame_buffer[frame_id] = (time_stamp, frame)
        if len(self.frame_buffer) > MAX_BUFFER_SIZE:
            keys = sorted(self.frame_buffer.keys())
            for k in keys[:-MAX_BUFFER_SIZE]:
                del self.frame_buffer[k]

    def update_detection_result(self, frame_id, detections):
        '''Update detection result for an old frame'''
        keys = list(self.frame_buffer.keys())
        for k in keys:
            if k < frame_id:
                del self.frame_buffer[k]
        if frame_id not in keys:
            print('frame_id not found. Increase buffer size')
            return None
        time_stamp, frame = self.frame_buffer[frame_id]
        if len(detections) > 0:
            feat = self.get_features(frame, detections)
            gallery = self.get_gallery()
            if gallery is not None:
                distmat = re_ranking(feat, gallery, k1=20, k2=6, lambda_value=0.3)
                matched_det, matched_id = self.match_id(distmat)
            else:
                matched_det = []
                matched_id = []
            # update matched track
            ret = [-1 for _ in range(len(detections))]
            for det, tid in zip(matched_det, matched_id):
                self.tracking[tid]['match_count'] += 1
                self.tracking[tid]['last_time'] = time_stamp
                self.tracking[tid]['vectors'].append(feat[det])
                gallery_size = len(self.tracking[tid]['vectors'])
                if gallery_size > MAX_GALLERY_SIZE:
                    remove_index = random.randint(0, gallery_size - 1)
                    del self.tracking[tid]['vectors'][remove_index]
                if self.tracking[tid]['match_count'] >= MIN_MATCH_COUNT:
                    ret[det] = tid
            # create new id for unmatched detections
            for det in range(len(detections)):
                if det not in matched_det:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracking[tid] = dict()
                    self.tracking[tid]['match_count'] = 1
                    self.tracking[tid]['update_count'] = 0
                    self.tracking[tid]['first_time'] = time_stamp
                    self.tracking[tid]['last_time'] = time_stamp
                    self.tracking[tid]['vectors'] = [feat[det]]

        # remove dead track and update 'update_count'
        counted_id = []
        keys = list(self.tracking.keys())
        for k in keys:
            self.tracking[tid]['update_count'] += 1
            if time_stamp - self.tracking[k]['last_time'] > MAX_AGE:
                if self.tracking[tid]['last_time'] - self.tracking[tid]['first_time'] > 3 * 60:
                    if self.tracking[tid]['match_count'] / self.tracking[tid]['update_count'] > 0.1:
                        self.total_live_time += self.tracking[k]['last_time'] - self.tracking[k]['first_time']
                        self.total_sequence += 1
                        counted_id.append(k)
                del self.tracking[k]

        return ret, counted_id

    def total_live_time(self):
        return self.total_live_time

    def avg_alive_time(self):
        if self.total_sequence == 0:
            return 0
        return self.total_live_time / self.total_sequence

    def total_sequence_count(self):
        return self.total_sequence

    def alive_ids(self):
        keys = list(self.tracking.keys())
        ret = []
        for k in keys:
            if self.tracking[k]['match_count'] >= MIN_MATCH_COUNT:
                ret.append(k)
        return ret

    def get_features(self, frame, detections):
        list_crop = []
        for l, t, r, b in detections:
            l = int(l)
            t = int(t)
            r = int(r)
            b = int(b)
            crop = frame[t:b, l:r]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = self.val_transforms(crop)
            list_crop.append(crop)
        data = torch.stack(list_crop)
        with torch.no_grad():
            data = data.to(self.device) if torch.cuda.device_count() >= 1 else data
            feats = self.model(data)
        return feats

    def get_gallery(self):
        tids = sorted(self.tracking.keys())
        list_vectors = []
        self.gallery_info = []
        for tid in tids:
            vectors = self.tracking[tid]['vectors']
            list_vectors.extend(vectors)
            self.gallery_info.append((tid, len(vectors)))
        if len(list_vectors) > 0:
            gallery = torch.stack(list_vectors)
        else:
            gallery = None
        return gallery

    def match_id(self, distmat):
        pairs = []
        nrow, ncol = distmat.shape
        for i in range(nrow):
            for j in range(ncol):
                pairs.append((distmat[i][j], i, j))
        pairs.sort()
        matched_row = []
        matched_id = []
        for dis, row, col in pairs:
            if dis > THRESHOLD:
                break
            tid = self._index_to_id(col)
            if row not in matched_row and tid not in matched_id:
                matched_row.append(row)
                matched_id.append(tid)
        return matched_row, matched_id

    def _index_to_id(self, index):
        for tid, length in self.gallery_info:
            if index < length:
                return tid
            else:
                index -= length
