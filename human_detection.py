import os
from ctypes import c_bool
from multiprocessing import Process, Queue, Value

import numpy as np
import tensorflow as tf

import config_main
import utils_main

MODEL_NAME = config_main.data['DETECTION_MODEL']
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CKPT = os.path.join(THIS_DIR, 'data_main',
                            MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(
    THIS_DIR, 'data_main', config_main.data['DETECTION_MAP_FILE'])

NUM_CLASSES = config_main.data['NUM_CLASSES']
THRESHOLD = config_main.data['DETECTION_THRESHOLD']
OVERLAP_THRESHOLD = config_main.data['DETECTION_OVERLAP_THRESHOLD']

KEEP_CLASSES = {'person'}


class HumanDetector:

    def __init__(self):
        self.in_queue = Queue(maxsize=1)
        self.out_queue = Queue(maxsize=1)
        self.stopped = Value(c_bool, False)

    @staticmethod
    def _run_function(in_queue, out_queue, stopped):
        if config_main.data['HUMAN_DETECTION_GPU'] < 0:
            device = '/device:CPU:0'
            my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
            tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            device = '/GPU:' + str(config_main.data['HUMAN_DETECTION_GPU'])
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        with tf.device(device):
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            label_map = utils_main.load_labelmap(PATH_TO_LABELS)
            categories = utils_main.convert_label_map_to_categories(
                label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = utils_main.create_category_index(categories)

            with detection_graph.as_default():
                with tf.Session(graph=detection_graph, config=config) as sess:
                    # Get input and output tensors
                    image_tensor = detection_graph.get_tensor_by_name(
                        'image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name(
                        'detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name(
                        'detection_scores:0')
                    classes = detection_graph.get_tensor_by_name(
                        'detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name(
                        'num_detections:0')

                    while not stopped.value:
                        fid, frame = in_queue.get()
                        if fid is None:
                            break

                        image_np_expanded = np.expand_dims(frame, axis=0)

                        # Actual detection.
                        (boxes_res, scores_res, classes_res, num_detections_res) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                        # post-processing of detection output
                        boxes_res = np.squeeze(boxes_res)
                        scores_res = np.squeeze(scores_res)
                        classes_res = np.squeeze(classes_res)

                        im_height, im_width = frame.shape[:2]

                        # Output will be stored in these lists
                        output_scores = []
                        output_cls = []
                        output_boxes = []

                        for i, score in enumerate(scores_res):
                            # Stop when score is lower than threshold since the
                            # score is sorted.
                            if score < THRESHOLD:
                                break
                            class_name = category_index[classes_res[i]]['name']

                            if class_name not in KEEP_CLASSES:
                                continue

                            ymin, xmin, ymax, xmax = boxes_res[i]
                            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                          ymin * im_height, ymax * im_height)

                            if utils_main.any_overlap((left, top, right, bottom), output_boxes, OVERLAP_THRESHOLD):
                                continue

                            # Append the detections to list
                            output_scores.append(score)
                            output_cls.append(class_name)
                            output_boxes.append((left, top, right, bottom))
                        try:
                            out_queue.get(False)
                        except Exception:
                            pass
                        out_queue.put(
                            (fid, output_scores, output_cls, output_boxes))

    def start(self):
        # we need to run detector in another process to avoid Python's Global Interpreter Lock
        self.process = Process(target=HumanDetector._run_function,
                               args=(self.in_queue, self.out_queue, self.stopped))
        self.process.daemon = True
        self.process.start()

    def stop(self):
        if not self.stopped.value:
            self.stopped.value = True
            try:
                self.in_queue.put((None, None), False)
            except Exception:
                pass
            self.process.join()

    def put_frame(self, frame_id, img):
        """Send a image to the detection process"""
        try:
            self.in_queue.get(False)
        except Exception:
            pass
        self.in_queue.put((frame_id, img))

    def get_result(self, block=True):
        if block:
            return self.out_queue.get()
        else:
            try:
                res = self.out_queue.get(False)
            except Exception:
                return None
            return res

    def warm_up(self):
        """Feed a empty image through the network to warm it up"""
        img = np.zeros((512, 512, 3), np.int8)
        self.put_frame(-1, img)
