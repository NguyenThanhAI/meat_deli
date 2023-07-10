class Tracker:
    def __init__(self):
        pass

    def update_frame(self, frame_id, time_stamp, frame):
        '''Add a new frame to the tracking list'''
        pass

    def update_detection_result(self, frame_id, detections):
        '''Update detection result for an old frame'''
        pass

    def avg_alive_time(self):
        pass

    def total_sequence_count(self):
        pass

    def alive_idxs(self):
        pass
