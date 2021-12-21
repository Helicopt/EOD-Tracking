import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .utils.kalman_filter import KalmanFilter
from .utils import matching
from .basetrack import BaseTrack
from .no_tracking import NoTracking, TrackState
from .utils.tracklet import STrack

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY


@MODULE_ZOO_REGISTRY.register('bytetracker')
class BYTETracker(NoTracking):
    def __init__(self, cfg, frame_rate=30):
        super(BYTETracker, self).__init__()
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        
        self.cfg = cfg
        # self.det_thresh = args.track_thresh
        self.track_thresh = self.cfg.get('track_thresh', 0.6)
        self.track_buffer = self.cfg.get('track_buffer', 30)
        self.conf_thresh = self.cfg.get('conf_thresh', 0.1)
        self.match_thresh = self.cfg.get('match_thresh', 0.9)
        
        
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
        self.device = None
        
    def initialize(self, state):
        super().initialize(state)
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0

    def finalize(self, state):
        tag = 'data'
        return NotImplemented    
    
    # def get_gt(self, image_id):
    #     from senseTk.common import TrackSet
    #     frame_id = int(os.path.basename(image_id).split('.')[0])
    #     seq_dir = os.path.dirname(os.path.dirname(image_id))
    #     gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
    #     seq = os.path.basename(seq_dir)
    #     if self.use_gt and os.path.exists(gt_file):
    #         if not hasattr(self, 'seq_name') or self.seq_name != seq:
    #             self.seq_name = seq
    #             self.gt = TrackSet(gt_file)
    #         return seq, self.gt[frame_id]
    #     else:
    #         return seq, None
    
    def preprocess(self, bboxes, info):
        raw_bboxes = bboxes
        bboxes = bboxes.clone()
        # embeds = F.normalize(embeds, dim=1)
        scale_h, scale_w = info[2]
        pad_h, pad_w = info[6], info[7]
        bboxes[:, [0, 2]] -= pad_w
        bboxes[:, [0, 2]] /= scale_w
        bboxes[:, [1, 3]] -= pad_h
        bboxes[:, [1, 3]] /= scale_h
        return raw_bboxes, bboxes
    
    def forward(self, state, inputs):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        output_results, real_output_results = self.preprocess(inputs['dt_bboxes'], inputs['image_info'])
        # seq, gt = self.get_gt(inputs['image_id'])
        self.device = output_results.device
        
        total_scores = real_output_results[:, 4]
        total_bboxes = real_output_results[:, :4]
        total_cls = real_output_results[:, 5]
        
        remain_inds = total_scores > self.track_thresh
        
        # still
        remain_inds = remain_inds.cpu().numpy()
        scores = total_scores.cpu().numpy()
        bboxes = total_bboxes.cpu().numpy()
        total_cls = total_cls.cpu().numpy()
        
        inds_low = scores > self.conf_thresh
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        clss = total_cls[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_second = total_cls[inds_second]
        

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, clss)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.cfg.get('mot20', False):
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets_second, scores_second, cls_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.cfg.get('mot20', False):
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # transfer format ['image_info', 'dt_bboxes', 'image_id']
        output_targets = np.array([track.dt_bboxes for track in output_stracks])
        # output_dets = torch.cat([bboxes, ids], dim=1)
        output_targets = output_targets[output_targets[:, -1] > 0]
        
        dt_boxes = torch.from_numpy(output_targets)
        raw_dt_boxes = self.postprocess(dt_boxes, inputs['image_info'])
        inputs['dt_bboxes'] = raw_dt_boxes
        
        return inputs

    def postprocess(self, bboxes, info):
        # raw_bboxes = bboxes
        _bboxes = bboxes.clone()
        # embeds = F.normalize(embeds, dim=1)
        scale_h, scale_w = info[2]
        pad_h, pad_w = info[6], info[7]
        _bboxes[:, [0, 2]] *= scale_w
        _bboxes[:, [0, 2]] += pad_w
        _bboxes[:, [1, 3]] *= scale_h
        _bboxes[:, [1, 3]] += pad_h
        
        return _bboxes

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
