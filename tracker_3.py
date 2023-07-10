import time

import config_main
import utils_main

MIN_LIVE_TIME = config_main.data['MIN_LIVE_TIME']
DIS_THRESHOLD = config_main.data['MAX_DISTANCE']


def box_dis(box1, box2):
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    size_1 = ((b1 - t1) + (r1 - l1)) / 2
    size_2 = ((b2 - t2) + (r2 - l2)) / 2
    cx1 = (l1 + r1) / 2
    cy1 = (t1 + b1) / 2
    cx2 = (l2 + r2) / 2
    cy2 = (t2 + b2) / 2
    return utils_main.distance((cx1, cy1), (cx2, cy2)) / ((size_1 + size_2) / 2)


class SimpleTracker:
    def __init__(self, max_age=2.0):
        self.tracked_item = []
        self.max_age = max_age  # in seconds
        self.total_sequence = 0

    def alive_items(self):
        return self.tracked_item

    def alive_ids(self):
        return [item['id'] for item in self.tracked_item]

    def total_sequence_count(self):
        return self.total_sequence

    def update(self, list_boxes):
        ret_ids = [None for _ in range(len(list_boxes))]  # id for each box in list_boxes
        list_tracked_boxes = [item['box'] for item in self.tracked_item]

        overlap_list = []
        for i in range(len(list_boxes)):
            for j in range(len(list_tracked_boxes)):
                dis = box_dis(list_boxes[i], list_tracked_boxes[j])
                if dis < DIS_THRESHOLD:
                    overlap_list.append((dis, i, j))
        overlap_list = sorted(overlap_list)

        list_match = []
        list_tracked_match = []
        new_track_list = []
        for overlap, i, j in overlap_list:
            if i not in list_match and j not in list_tracked_match:
                item = self.tracked_item[j]
                if time.time() - item['last_time'] < self.max_age:
                    list_match.append(i)
                    list_tracked_match.append(j)
                    item['last_time'] = time.time()
                    item['box'] = list_boxes[i]
                    new_track_list.append(item)
                    ret_ids[i] = item['id']
        # new item
        for i in range(len(list_boxes)):
            if i not in list_match:
                item = dict()
                item['id'] = item['first_time'] = item['last_time'] = time.time()
                item['box'] = list_boxes[i]
                new_track_list.append(item)
                ret_ids[i] = item['id']
        # keep old item
        counted_ids = []
        for j in range(len(self.tracked_item)):
            if j not in list_tracked_match:
                item = self.tracked_item[j]
                if time.time() - item['last_time'] < self.max_age:
                    new_track_list.append(item)
                else:
                    if item['last_time'] - item['first_time'] > MIN_LIVE_TIME:
                        self.total_sequence += 1
                        counted_ids.append(item['id'])

        self.tracked_item = new_track_list

        return ret_ids, counted_ids
