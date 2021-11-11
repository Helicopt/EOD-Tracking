from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .no_tracking import NoTracking
from eod.utils.general.log_helper import default_logger as logger
import torch
import numpy as np
from ..utils.debug import info_debug
from ..utils.matching import bbox_overlaps

__all__ = ['MotionAppearanceOnlineTracker']


class Tracklet(object):

    def __init__(self, det, uid, last):
        self.det = det
        self.uid = uid
        self.last = last


@MODULE_ZOO_REGISTRY.register('ma_online')
class MotionAppearanceOnlineTracker(NoTracking):

    def __init__(self, output_thr=0.5, **kwargs):
        super().__init__()
        self.output_thr = output_thr

    def initialize(self, state):
        super().initialize(state)
        state.tracklets = []
        self.id_cnt = 0
        self.fr = 0

    @property
    def next_id(self):
        self.id_cnt += 1
        return self.id_cnt

    def forward(self, state, inputs):
        self.fr += 1
        bboxes = inputs['dt_bboxes']
        self.device = bboxes.device
        keep = bboxes[:, 4] > self.output_thr
        bboxes = bboxes[keep]
        dets = [trk.det for trk in state.tracklets]
        if len(dets) > 0:
            dets = torch.stack(dets)
            ious = bbox_overlaps(dets[:, :4], bboxes[:, :4])
            mx, inds = ious.max(dim=1)
        valid = {}
        for i in range(len(dets)):
            matched = int(inds[i])
            if mx[i] > 0.5 and valid.get(matched, -1) < 0:
                state.tracklets[i].det = bboxes[matched]
                state.tracklets[i].last = self.fr
                valid[matched] = i
        ids = []
        for j in range(bboxes.size(0)):
            if j not in valid:
                nid = self.next_id
                state.tracklets.append(Tracklet(bboxes[j], nid, self.fr))
            else:
                nid = state.tracklets[valid[j]].uid
            ids.append((int(j not in valid), nid))
        new_tracklets = []
        for i, t in enumerate(state.tracklets):
            if self.fr - t.last < 20:
                new_tracklets.append(t)
        state.tracklets = new_tracklets
        ids = torch.from_numpy(np.array(ids, dtype=np.float)).to(bboxes.device)
        output_dets = torch.cat([bboxes, ids], dim=1)
        inputs['dt_bboxes'] = output_dets
        return inputs
