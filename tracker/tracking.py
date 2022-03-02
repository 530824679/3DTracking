
import numpy as np
from utils.utils import iou3d
from tracker.kalman_filter import KalmanTracker
from utils.utils import convert_3dbox_to_8corner
from scipy.optimize import linear_sum_assignment

class MOT3D(object):
    # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
    def __init__(self, max_age=2, min_hits=3):
        """
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]

    def update(self, dets_all, seq, frame):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.seq = seq
        self.frame = frame

        dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
        dets = dets[:, self.reorder]
        self.frame_count += 1

        prev_trks = np.zeros((len(self.trackers), 7))
        for t, trk in enumerate(prev_trks):
            pos = self.trackers[t].kf.x.reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
        prev_trks = np.ma.compress_rows(np.ma.masked_invalid(prev_trks))

        # N x 7 , #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            dets_8corner, trks_8corner, dets, prev_trks, mu=1.0)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[
                    0], 0]     # a list of index
                trk.update(dets[d, :][0], info[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:        # a scalar of index
            trk = KalmanTracker(dets[i, :], info[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()      # bbox location
            d = d[self.reorder_back]

            if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate(
                    (d, [trk.id + 1], trk.info)).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            # x, y, z, theta, l, w, h, ID, other info, confidence
            return np.concatenate(ret)
        return np.empty((0, 15))

    def associate_detections_to_trackers(self, detections, trackers, dets, trks, mu=1.0, iou_threshold=0.1):
        # def associate_detections_to_trackers(detections,trackers,iou_threshold=0.01):     # ablation study
        # def
        # associate_detections_to_trackers(detections,trackers,iou_threshold=0.25):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        detections:  N x 8 x 3
        trackers:    M x 8 x 3

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                # det: 8 x 3, trk: 8 x 3
                iou_matrix[d, t] = iou3d(det, trk)[0]

        # deep features
        # boxes1, boxes2 = [], []
        # for d in dets:
        #     for t in trks:
        #         boxes1.append(d)
        #         boxes2.append(t)

        if mu != 1.0:
            sim = self.feature_model.compute_similarity(self.seq, self.frame, dets, trks)
            sim_matrix = sim.reshape(len(dets), len(trks))
        else:
            sim_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        cost_matrix = -(mu * iou_matrix + (1. - mu) * sim_matrix)

        # matched_indices = linear_assignment(-iou_matrix)      # hougarian algorithm
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        matched_indices = np.stack([row_inds, col_inds], axis=1)


        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if(len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)