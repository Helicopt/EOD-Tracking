import os
import json
from collections import deque
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .no_tracking import NoTracking
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.env.dist_helper import env
import torch
import torch.nn.functional as F
import numpy as np
from ..utils.debug import info_debug, logger_print
from ..utils.matching import bbox_overlaps, bbox_dist
try:
    import xgboost as xgb
    xgb_flag = True
except:
    xgb_flag = False
from .online_trackers import MotionAppearanceOnlineTracker, to_json, Tracklet

__all__ = ['CrowdedMAOnlineTracker']


builtin_print = print
print = logger_print


@MODULE_ZOO_REGISTRY.register('crowded_ma_online')
class CrowdedMAOnlineTracker(MotionAppearanceOnlineTracker):

    def predict(self, data):
        ious = data['mat']['ious']
        dists = data['mat']['dist']
        sims = data['mat']['sims']
        skipped_frs = data['left']['skipped']
        occluders = data['left']['occluder']
        gt_sims = data['mat'].get('gt_sims', None)
        scores = ious.new_zeros(ious.shape)
        if isinstance(occluders, list):
            is_occluded = torch.from_numpy(np.array(occluders)).to(ious.device)
        is_occluded = is_occluded >= 0
        is_occluded = is_occluded.unsqueeze(-1).repeat(1, ious.size(1))
        mask = (ious > 0.5) & (dists < 0.2) & (~ is_occluded)
        scores[mask] = ious[mask]
        mask = (ious > 0.5) & (dists < 0.2) & (is_occluded)
        scores[mask] = ious[mask] - 0.185 + sims[mask] / 10.
        if isinstance(skipped_frs, list):
            skipped_frs = torch.from_numpy(np.array(skipped_frs)).to(ious.device)
        skipped_frs = skipped_frs.unsqueeze(-1).repeat(1, ious.size(1))
        mask = (sims > self.high_sim_thr) & ((dists / skipped_frs) < 0.15) & (~is_occluded)
        scores[mask & (scores < 0.5)] = 0.5 + sims[mask & (scores < 0.5)] / 10
        # ious[sims < self.low_sim_thr] -= 0.3
        # ious = (gt_sims & gt_masks).float()
        if False and gt_sims is not None:
            scores = gt_sims.float()
        o_gids = torch.from_numpy(np.array(data['left']['uids'])).to(ious.device)
        # mask = o_gids == 147
        if env.rank == 3 and False:
            # for u, v, w in zip(o_gids, ooids, uids):
            #     builtin_print(u, v, w)
            # builtin_print(ooids)
            builtin_print(data['real_frame'])
            builtin_print(ious[mask])
            builtin_print(scores[mask])
        return scores

    def decay(self, score, left):
        # skipped = left['skipped']
        # if skipped > 1:
        #     return score / 10.
        return score

    def prepare_data(self, real_frame, dets, dfeats, bboxes, embeds, tracklets, o_gids=None, gids=None):
        data = super().prepare_data(real_frame, dets, dfeats, bboxes, embeds, tracklets, o_gids=o_gids, gids=gids)
        occluder = [int(trk.occluder.uid) if trk.occluder is not None else -1 for trk in tracklets]
        data['left']['occluder'] = occluder
        return data

    def forward(self, state, inputs):
        ret = super().forward(state, inputs)
        ndets = [trk.det for trk in state.tracklets]
        ndets = torch.stack(ndets)
        ious = bbox_overlaps(ndets, ndets)
        if ious.numel() > 0:
            mx, inds = ious.topk(2, dim=1)
            mx = mx[:, 1:]
            inds = inds[:, 1:]
            for i, t in enumerate(state.tracklets):
                if getattr(t, 'occluder', None) is None:
                    t.occluder = None
                    ind = int(inds[i])
                    to = state.tracklets[ind]
                    if i != ind and mx[i] > 0.5 and to.y2 > t.y2 and self.fr == to.last and getattr(to, 'occluder', None) is None:
                        t.occluder = to
                else:
                    to = t.occluder
                    if to.last != self.fr or t.last == self.fr and t.iou(to) < 0.5:
                        t.occluder = None
        # for i, trk in enumerate(state.tracklets):
        #     print(trk.uid, int(trk.gid), ('(%d:%d)' % (trk.occluder.uid, trk.occluder.gid))
        #           if trk.occluder is not None else 'None')
        return ret
