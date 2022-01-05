import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .utils import matching
from .byte_tracker_misc.basetrack import BaseTrack
from .no_tracking import NoTracking, TrackState
from .utils.tracklet import STrack

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY


@MODULE_ZOO_REGISTRY.register('simple_tracker')
class SimpleTracker(NoTracking):
    def __init__(self, cfg, frame_rate=30):
        super(SimpleTracker, self).__init__()
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

        #TODO: add

        # return None

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
