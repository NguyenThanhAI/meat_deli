import logging
import math
import random
import string

import cv2
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from numba import jit

import string_int_label_map_pb2


def iou(l1, t1, r1, b1, l2, t2, r2, b2):
    """Intersection over union"""
    l = max(l1, l2)
    r = min(r1, r2)
    t = max(t1, t2)
    b = min(b1, b2)
    if l > r or t > b:
        return 0
    s_i = (r - l) * (b - t)
    s_u = (b1 - t1) * (r1 - l1) + (b2 - t2) * (r2 - l2) - s_i
    return float(s_i) / s_u


def any_overlap(box, list_boxes, threshold=0.5):
    l, t, r, b = box
    for l2, t2, r2, b2 in list_boxes:
        if iou(l, t, r, b, l2, t2, r2, b2) > threshold:
            return True
    return False


def _validate_label_map(label_map):
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if (item.id == 0 and item.name != 'background' and
                item.display_name != 'background'):
            raise ValueError('Label map id 0 is reserved for the background label')


def load_labelmap(path):
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({
                'id': class_id + label_id_offset,
                'name': 'category_{}'.format(class_id + label_id_offset)
            })
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested '
                         'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories


def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def random_id():
    return ''.join(random.choice(string.digits) for _ in range(5))


def putTextLabel(img, orig, text, font_face, font_scale, text_color, bg_color, thickness=1, bottom=False):
    size = cv2.getTextSize(text, font_face, font_scale, thickness)
    x, y = orig
    if bottom:
        cv2.rectangle(img, (x, y - size[0][1] - 6), (x + size[0][0] + 6, y), bg_color, cv2.FILLED)
        cv2.putText(img, text, (x + 3, y - 3), font_face, font_scale, text_color, thickness)
    else:
        cv2.rectangle(img, (x, y), (x + size[0][0] + 6, y + size[0][1] + 6), bg_color, cv2.FILLED)
        cv2.putText(img, text, (x + 3, y + size[0][1] + 3), font_face, font_scale, text_color, thickness)


def resize_by_height(img, new_height):
    row, col = img.shape[:2]
    new_width = int(col * new_height / row)
    return cv2.resize(img, (new_width, new_height))


def resize_by_width(img, new_width):
    row, col = img.shape[:2]
    new_height = int(row * new_width / col)
    return cv2.resize(img, (new_width, new_height))


def resize_max_size(img, max_w, max_h):
    img_h, img_w = img.shape[:2]
    if img_w <= max_w and img_h <= max_h:
        return img
    else:
        max_w = min(max_w, img_w)
        max_h = min(max_h, img_h)
        ratio_w = max_w / img_w
        ratio_h = max_h / img_h
        if ratio_w < ratio_h:
            return resize_by_width(img, max_w)
        else:
            return resize_by_height(img, max_h)


def blend_heatmap(frame, hmap, power):
    if power != 1:
        hmap = np.power(hmap, power)
    hmap_hsv = 120 - hmap * 120
    hsv_bg = np.full((hmap.shape[0], hmap.shape[1], 3), 255.0)
    hsv_bg[:, :, 0] = hmap_hsv
    bgr_bg = cv2.cvtColor(hsv_bg.astype('uint8'), cv2.COLOR_HSV2BGR)
    hmap = cv2.resize(hmap, (frame.shape[1], frame.shape[0]))
    bgr_bg = cv2.resize(bgr_bg, (frame.shape[1], frame.shape[0]))
    ret = _blend(frame, bgr_bg, hmap)
    return ret.astype('uint8')


@jit
def _blend(frame, bgr_bg, hmap):
    ret = np.zeros_like(frame)
    nrow, ncol = hmap.shape[:2]
    for i in range(nrow):
        for j in range(ncol):
            ret[i, j, 0] = frame[i, j, 0] * (1 - hmap[i, j]) + bgr_bg[i, j, 0] * hmap[i, j]
            ret[i, j, 1] = frame[i, j, 1] * (1 - hmap[i, j]) + bgr_bg[i, j, 1] * hmap[i, j]
            ret[i, j, 2] = frame[i, j, 2] * (1 - hmap[i, j]) + bgr_bg[i, j, 2] * hmap[i, j]
    return ret


def distance(p1, p2):
    d = (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
    return math.sqrt(d)
