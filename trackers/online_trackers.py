import os

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

__all__ = ['MotionAppearanceOnlineTracker', 'XGBMotionAppearanceTracker', 'PureAppearanceTracker', 'PureMotionTracker']


builtin_print = print
print = logger_print


class Tracklet(object):

    def __init__(self, det, feat, uid, last):
        self.det = det
        self.feat = feat
        self.uid = uid
        self.last = last


@MODULE_ZOO_REGISTRY.register('ma_online')
class MotionAppearanceOnlineTracker(NoTracking):

    def __init__(self, output_thr=0.5, interval=30, reset_id=True, save_feats=False, save_feats_dir='/home/toka/code/EOD/data/', **kwargs):
        super().__init__()
        if isinstance(output_thr, float):
            self.output_thr = {'default': output_thr}
        else:
            self.output_thr = output_thr
        self.interval = interval
        self.high_sim_thr = 0.9
        self.low_sim_thr = 0.5
        self.use_gt = True
        self.reset_id = reset_id
        self.verbose = False
        self.save_feats = save_feats
        self.save_feats_dir = save_feats_dir
        logger.info('Setting up %s' % (self.__class__))

    def initialize(self, state):
        super().initialize(state)
        state.tracklets = []
        if self.save_feats:
            state.cache_items = []
        if self.reset_id:
            self.id_cnt = 0
        self.fr = 0

    def finalize(self, state):
        tag = 'data'
        if self.save_feats:
            os.makedirs(self.save_feats_dir, exist_ok=True)
            torch.save(state.cache_items, os.path.join(self.save_feats_dir, 'tracker.%s.%d.%s.%d.pkl' %
                       (tag, env.rank, self.seq_name, self.id_cnt)))
        del state.cache_items
        return NotImplemented

    @property
    def next_id(self):
        self.id_cnt += 1
        if env.distributed:
            return self.id_cnt * env.world_size + env.rank
        return self.id_cnt

    def get_gt(self, image_id):
        import os
        from senseTk.common import TrackSet
        frame_id = int(os.path.basename(image_id).split('.')[0])
        seq_dir = os.path.dirname(os.path.dirname(image_id))
        gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
        seq = os.path.basename(seq_dir)
        if self.use_gt and os.path.exists(gt_file):
            if not hasattr(self, 'seq_name') or self.seq_name != seq:
                self.seq_name = seq
                self.gt = TrackSet(gt_file)
            return seq, self.gt[frame_id]
        else:
            return seq, None

    def collect(self, frame_row, device='cpu'):
        bboxes = []
        for d in frame_row:
            bboxes.append([d.x1, d.y1, d.x2, d.y2, d.uid])
        bboxes = np.array(bboxes)
        bboxes = torch.from_numpy(bboxes).to(device)
        return bboxes

    def preprocess(self, bboxes, embeds, info):
        raw_bboxes = bboxes
        bboxes = bboxes.clone()
        embeds = F.normalize(embeds, dim=1)
        scale_h, scale_w = info[2]
        pad_h, pad_w = info[6], info[7]
        bboxes[:, [0, 2]] -= pad_w
        bboxes[:, [0, 2]] /= scale_w
        bboxes[:, [1, 3]] -= pad_h
        bboxes[:, [1, 3]] /= scale_h
        return raw_bboxes, bboxes, embeds

    def forward(self, state, inputs):
        self.fr += 1
        bboxes, real_bboxes, embeds = self.preprocess(inputs['dt_bboxes'], inputs['id_embeds'], inputs['image_info'])
        seq, gt = self.get_gt(inputs['image_id'])
        self.device = bboxes.device
        keep = bboxes[:, 4] > 0.1
        bboxes = bboxes[keep]
        embeds = embeds[keep]
        real_bboxes = real_bboxes[keep]
        if gt is not None:
            gtbboxes = self.collect(gt, device=self.device)
            gt_ious = bbox_overlaps(real_bboxes[:, :4], gtbboxes[:, :4])
            g, gi = gt_ious.max(dim=1)
            gids = gtbboxes[:, 4][gi]
            gids[g < 0.5] = -1
            raw_old_gids = [int(t.gid) for t in state.tracklets]
            old_gids = set(raw_old_gids)
            new_gids = {*[int(gid) for gid in gids]}
        dets = [trk.det for trk in state.tracklets]
        dfeats = [trk.feat for trk in state.tracklets]
        valid = {}
        if len(dets) > 0 and len(bboxes) > 0:
            data = [self.fr, real_bboxes, embeds, [(self.fr - trk.last) for trk in state.tracklets]]
            dets = torch.stack(dets)
            ious = bbox_overlaps(dets[:, :4], bboxes[:, :4])
            ndist = bbox_dist(dets[:, :4], bboxes[:, :4])
            dfeats = torch.stack(dfeats)
            sims = torch.matmul(dfeats, embeds.T)
            if gt is not None:
                o_gids = torch.from_numpy(np.array(raw_old_gids)).to(gids.device)
                gt_sims = o_gids.reshape(-1, 1) == gids.reshape(1, -1)
                gt_masks = (o_gids >= 0).reshape(-1, 1) | (gids >= 0).reshape(1, -1)
                data.append(ious)
                data.append(ndist)
                data.append(sims)
                data.append(gt_sims)
                data.append(gt_masks)
            # # print(embeds.shape, bboxes.shape)
            data = list(map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, data))
            if self.save_feats:
                state.cache_items.append(data)
            ious[sims > self.high_sim_thr] += 0.6
            ious[sims < self.low_sim_thr] -= 0.3
            # ious = (gt_sims & gt_masks).float()
            mx, inds = ious.max(dim=1)
            sinds = sorted(list(range(len(dets))),
                           key=lambda x: (mx[x]) if (state.tracklets[x].last == self.fr - 1) else (mx[x] / 10.), reverse=True)
            # sinds = sorted(list(range(len(dets))),
            #                key=lambda x: (mx[x]), reverse=True)
            for i in range(len(dets)):
                i = sinds[i]
                matched = int(inds[i])
                if mx[i] > 0.4 and valid.get(matched, -1) < 0:
                    state.tracklets[i].det = bboxes[matched]
                    state.tracklets[i].feat = embeds[matched]
                    if gt is not None:
                        if self.verbose and state.tracklets[i].gid != gids[matched]:
                            print(seq, self.fr, 'miss_met', self.fr - state.tracklets[i].last, (int(state.tracklets[i].gid), int(gids[matched])), mx[i],
                                  ious[i, matched], ndist[i, matched], sims[i, matched])
                        state.tracklets[i].gid = gids[matched]
                    valid[matched] = i
                    state.tracklets[i].last = self.fr
                elif gt is not None:
                    for mm, k in enumerate(gids):
                        if k == state.tracklets[i].gid:
                            break
                    if self.verbose and self.fr - state.tracklets[i].last > 5 and int(state.tracklets[i].gid) in new_gids and int(state.tracklets[i].gid) >= 0:
                        print(seq, self.fr, 'unmet', self.fr - state.tracklets[i].last, (int(state.tracklets[i].gid), valid.get(mm, '---')),
                              ious[i, matched], ndist[i, matched], sims[i, matched])
        ids = []
        thr = self.output_thr.setdefault(seq, self.output_thr['default'])
        for j in range(bboxes.size(0)):
            if j not in valid:
                if bboxes[j, 4] > thr:
                    nid = self.next_id
                    t = Tracklet(bboxes[j], embeds[j], nid, self.fr)
                    if gt is not None:
                        t.gid = gids[j]
                        if self.verbose and int(gids[j]) in old_gids and int(gids[j]) >= 0:
                            for i in range(len(raw_old_gids)):
                                to = state.tracklets[i]
                                if raw_old_gids[i] == t.gid:
                                    matched = i
                            print(seq, self.fr, 'wrongly ascended: ', self.fr - state.tracklets[matched].last,
                                  gids[j], ious[matched, j], sims[matched, j])
                    state.tracklets.append(t)
                else:
                    nid = -1
            else:
                nid = state.tracklets[valid[j]].uid
            ids.append((int(j not in valid), nid))
        new_tracklets = []
        for i, t in enumerate(state.tracklets):
            if self.fr - t.last < self.interval:
                new_tracklets.append(t)
        state.tracklets = new_tracklets
        if len(ids) > 0:
            ids = torch.from_numpy(np.array(ids, dtype=np.float)).to(bboxes.device)
        else:
            ids = bboxes.new_zeros((bboxes.size(0), 2))
        output_dets = torch.cat([bboxes, ids], dim=1)
        output_dets = output_dets[output_dets[:, -1] > 0]
        inputs['dt_bboxes'] = output_dets
        # print(output_dets[:10])
        return inputs


@MODULE_ZOO_REGISTRY.register('xgb_ma')
class XGBMotionAppearanceTracker(MotionAppearanceOnlineTracker):

    def __init__(self, xgb_model=None, **kwargs):
        super().__init__(**kwargs)
        self.xgb_model = xgb.Booster({'nthread': 4})
        self.xgb_model.load_model(xgb_model)
        self.use_gt = False
        assert xgb_flag, 'xgboost import failed'

    def finalize(self, state):
        return NotImplemented

    def xgb_predict(self, intvals, ious, ndists, sims):
        device = ious.device
        ious = ious.cpu().numpy()
        ndists = ndists.cpu().numpy()
        sims = sims.cpu().numpy()
        intvls = np.array(intvals).reshape(-1, 1).repeat(ious.shape[1], 1)
        X = np.stack([intvls.astype(np.float32), ious.astype(np.float32), ndists.astype(np.float32),
                      sims.astype(np.float32)]).transpose((1, 2, 0))
        N, M, C = X.shape
        X = X.reshape(N * M, C)
        Y = self.xgb_model.predict(xgb.DMatrix(X))
        Y = Y.reshape(N, M)
        # print(Y[:3])

        return torch.from_numpy(Y).to(device)

    def forward(self, state, inputs):
        self.fr += 1
        bboxes, real_bboxes, embeds = self.preprocess(inputs['dt_bboxes'], inputs['id_embeds'], inputs['image_info'])
        seq, gt = self.get_gt(inputs['image_id'])
        self.device = bboxes.device
        keep = bboxes[:, 4] > 0.1
        bboxes = bboxes[keep]
        embeds = embeds[keep]
        real_bboxes = real_bboxes[keep]
        if gt is not None:
            gtbboxes = self.collect(gt, device=self.device)
            gt_ious = bbox_overlaps(real_bboxes[:, :4], gtbboxes[:, :4])
            g, gi = gt_ious.max(dim=1)
            gids = gtbboxes[:, 4][gi]
            gids[g < 0.5] = -1
            raw_old_gids = [int(t.gid) for t in state.tracklets]
            old_gids = set(raw_old_gids)
            new_gids = {*[int(gid) for gid in gids]}
        dets = [trk.det for trk in state.tracklets]
        dfeats = [trk.feat for trk in state.tracklets]
        valid = {}
        if len(dets) > 0 and len(bboxes) > 0:
            data = [self.fr, real_bboxes, embeds, [(self.fr - trk.last) for trk in state.tracklets]]
            dets = torch.stack(dets)
            ious = bbox_overlaps(dets[:, :4], bboxes[:, :4])
            ndists = bbox_dist(dets[:, :4], bboxes[:, :4])
            dfeats = torch.stack(dfeats)
            sims = torch.matmul(dfeats, embeds.T)
            if gt is not None:
                o_gids = torch.from_numpy(np.array(raw_old_gids)).to(gids.device)
                gt_sims = o_gids.reshape(-1, 1) == gids.reshape(1, -1)
                gt_masks = (o_gids >= 0).reshape(-1, 1) | (gids >= 0).reshape(1, -1)
                data.append(ious)
                data.append(sims)
                data.append(gt_sims)
                data.append(gt_masks)
            # # print(embeds.shape, bboxes.shape)
            # data = list(map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, data))
            # state.cache_items.append(data)
            # ious[sims > self.high_sim_thr] += 0.6
            # ious[sims < self.low_sim_thr] -= 0.3
            # ious = (gt_sims & gt_masks).float()
            xgbsims = self.xgb_predict([(self.fr - trk.last) for trk in state.tracklets], ious, ndists, sims)
            mx, inds = xgbsims.max(dim=1)
            # sinds = sorted(list(range(len(dets))),
            #                key=lambda x: (mx[x]) if (state.tracklets[x].last == self.fr) else (mx[x] / 10.), reverse=True)
            sinds = sorted(list(range(len(dets))), key=lambda x: mx[x], reverse=True)
            for i in range(len(dets)):
                i = sinds[i]
                matched = int(inds[i])
                if mx[i] > 0.1 and valid.get(matched, -1) < 0:
                    state.tracklets[i].det = bboxes[matched]
                    state.tracklets[i].feat = embeds[matched]
                    state.tracklets[i].last = self.fr
                    if gt is not None:
                        if self.verbose and state.tracklets[i].gid != gids[matched]:
                            print(seq, self.fr, 'miss_met', state.tracklets[i].gid, gids[matched], mx[i],
                                  sims[i, matched], ious[i], sims[i])
                        state.tracklets[i].gid = gids[matched]
                    valid[matched] = i
                elif gt is not None:
                    if self.verbose and int(state.tracklets[i].gid) in new_gids and int(state.tracklets[i].gid) >= 0:
                        print(seq, self.fr, 'unmet', state.tracklets[i].gid, mx[i],
                              ious[i, matched], sims[i, matched], ious[i], sims[i])
        ids = []
        thr = self.output_thr.setdefault(seq, self.output_thr['default'])
        for j in range(bboxes.size(0)):
            if j not in valid:
                if bboxes[j, 4] > thr:
                    nid = self.next_id
                    t = Tracklet(bboxes[j], embeds[j], nid, self.fr)
                    if gt is not None:
                        t.gid = gids[j]
                        if self.verbose and int(gids[j]) in old_gids and int(gids[j]) >= 0:
                            for i, to in enumerate(state.tracklets):
                                if raw_old_gids[i] == t.gid:
                                    matched = i
                            print(seq, self.fr, self.fr == state.tracklets[matched].last, 'wrongly ascended: ',
                                  gids[j], ious[matched, j], sims[matched, j], ious[:, j], sims[:, j])
                    state.tracklets.append(t)
                else:
                    nid = -1
            else:
                nid = state.tracklets[valid[j]].uid
            ids.append((int(j not in valid), nid))
        new_tracklets = []
        for i, t in enumerate(state.tracklets):
            if self.fr - t.last < self.interval:
                new_tracklets.append(t)
        state.tracklets = new_tracklets
        if len(ids) > 0:
            ids = torch.from_numpy(np.array(ids, dtype=np.float)).to(bboxes.device)
        else:
            ids = bboxes.new_zeros((bboxes.size(0), 2))
        output_dets = torch.cat([bboxes, ids], dim=1)
        output_dets = output_dets[output_dets[:, -1] > 0]
        inputs['dt_bboxes'] = output_dets
        # print(output_dets[:10])
        return inputs


@MODULE_ZOO_REGISTRY.register('pure_a')
class PureAppearanceTracker(NoTracking):

    def __init__(self, sim_thr=0.8, interval=30, **kwargs):
        super().__init__()
        self.output_thr = {'default': 0.01}
        self.sim_thr = sim_thr
        self.interval = interval
        self.use_gt = False
        self.reset_id = False
        self.id_cnt = 0

    def initialize(self, state):
        super().initialize(state)
        state.tracklets = []
        # state.cache_items = []
        self.fr = 0
        if self.reset_id:
            self.id_cnt = 0

    # def finalize(self, state):
    #     tag = 'train'
    #     torch.save(state.cache_items, '/home/toka/code/EOD/ptss/tracker.%s.%d.%s.%d.pkl' %
    #                (tag, env.rank, self.seq_name, self.id_cnt))
    #     del state.cache_items
    #     return NotImplemented

    @property
    def next_id(self):
        self.id_cnt += 1
        if env.distributed:
            return self.id_cnt * env.world_size + env.rank
        return self.id_cnt

    def get_gt(self, image_id):
        import os
        from senseTk.common import TrackSet
        frame_id = int(os.path.basename(image_id).split('.')[0])
        seq_dir = os.path.dirname(os.path.dirname(image_id))
        gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
        seq = os.path.basename(seq_dir)
        self.seq_name = seq
        if self.use_gt and os.path.exists(gt_file):
            if not hasattr(self, 'seq_name') or self.seq_name != seq:
                self.gt = TrackSet(gt_file)
            return seq, self.gt[frame_id]
        else:
            return seq, None

    def collect(self, frame_row, device='cpu'):
        bboxes = []
        for d in frame_row:
            bboxes.append([d.x1, d.y1, d.x2, d.y2, d.uid])
        bboxes = np.array(bboxes)
        bboxes = torch.from_numpy(bboxes).to(device)
        return bboxes

    def preprocess(self, bboxes, embeds, info):
        raw_bboxes = bboxes
        bboxes = bboxes.clone()
        embeds = F.normalize(embeds, dim=1)
        scale_h, scale_w = info[2]
        pad_h, pad_w = info[6], info[7]
        bboxes[:, [0, 2]] -= pad_w
        bboxes[:, [0, 2]] /= scale_w
        bboxes[:, [1, 3]] -= pad_h
        bboxes[:, [1, 3]] /= scale_h
        return raw_bboxes, bboxes, embeds

    def forward(self, state, inputs):
        self.fr += 1
        bboxes, real_bboxes, embeds = self.preprocess(inputs['dt_bboxes'], inputs['id_embeds'], inputs['image_info'])
        # state.cache_items.append((real_bboxes, embeds))
        seq, gt = self.get_gt(inputs['image_id'])
        self.device = bboxes.device
        keep = bboxes[:, 4] > self.output_thr.setdefault(seq, self.output_thr['default'])
        bboxes = bboxes[keep]
        embeds = embeds[keep]
        real_bboxes = real_bboxes[keep]
        if gt is not None:
            gtbboxes = self.collect(gt, device=self.device)
            gt_ious = bbox_overlaps(real_bboxes[:, :4], gtbboxes[:, :4])
            g, gi = gt_ious.max(dim=1)
            gids = gtbboxes[:, 4][gi]
            gids[g < 0.5] = -1
            raw_old_gids = [int(t.gid) for t in state.tracklets]
            old_gids = set(raw_old_gids)
            new_gids = {*[int(gid) for gid in gids]}
        dets = [trk.det for trk in state.tracklets]
        dfeats = [trk.feat for trk in state.tracklets]
        valid = {}
        if len(dets) > 0 and len(bboxes) > 0:
            dets = torch.stack(dets)
            # ious = bbox_overlaps(dets[:, :4], bboxes[:, :4])
            dfeats = torch.stack(dfeats)
            sims = torch.matmul(dfeats, embeds.T)
            # print(embeds.shape, bboxes.shape)
            # ious[sims > self.high_sim_thr] += 0.6
            # ious[sims < self.low_sim_thr] -= 0.3
            mx, inds = sims.max(dim=1)
            sinds = sorted(list(range(len(dets))),
                           key=lambda x: (mx[x]) if (state.tracklets[x].last == self.fr - 1) else (mx[x] - 1), reverse=True)
            for i in range(len(dets)):
                i = sinds[i]
                matched = int(inds[i])
                if mx[i] > self.sim_thr and valid.get(matched, -1) < 0:
                    state.tracklets[i].det = bboxes[matched]
                    state.tracklets[i].feat = embeds[matched]
                    state.tracklets[i].last = self.fr
                    if gt is not None:
                        if state.tracklets[i].gid != gids[matched]:
                            print(seq, self.fr, 'miss_met', state.tracklets[i].gid, gids[matched], mx[i],
                                  sims[i])
                        state.tracklets[i].gid = gids[matched]
                    valid[matched] = i
                elif gt is not None:
                    if int(state.tracklets[i].gid) in new_gids and int(state.tracklets[i].gid) >= 0:
                        print(seq, self.fr, 'unmet', state.tracklets[i].gid, mx[i], sims[i])
        ids = []
        for j in range(bboxes.size(0)):
            if j not in valid:
                nid = self.next_id
                t = Tracklet(bboxes[j], embeds[j], nid, self.fr)
                if gt is not None:
                    t.gid = gids[j]
                    if int(gids[j]) in old_gids and int(gids[j]) >= 0:
                        for i, to in enumerate(state.tracklets):
                            if raw_old_gids[i] == t.gid:
                                matched = i
                        print(seq, self.fr, self.fr == state.tracklets[matched].last, 'wrongly ascended: ',
                              gids[j], sims[matched, j], sims[:, j])
                state.tracklets.append(t)
            else:
                nid = state.tracklets[valid[j]].uid
            ids.append((int(j not in valid), nid))
        new_tracklets = []
        for i, t in enumerate(state.tracklets):
            if self.fr - t.last < self.interval:
                new_tracklets.append(t)
        state.tracklets = new_tracklets
        if len(ids) > 0:
            ids = torch.from_numpy(np.array(ids, dtype=np.float)).to(bboxes.device)
        else:
            ids = bboxes.new_zeros((bboxes.size(0), 2))
        output_dets = torch.cat([bboxes, ids], dim=1)
        inputs['dt_bboxes'] = output_dets
        # print(output_dets[:10])
        return inputs


@MODULE_ZOO_REGISTRY.register('pure_m')
class PureMotionTracker(NoTracking):

    def __init__(self, output_thr=0.01, sim_thr=0.8, interval=30, two_step=False, **kwargs):
        super().__init__()
        if isinstance(output_thr, float):
            self.output_thr = {'default': output_thr}
        else:
            self.output_thr = output_thr
        self.sim_thr = sim_thr
        self.interval = interval
        self.use_gt = False
        self.reset_id = False
        self.id_cnt = 0
        self.two_step = two_step

    def initialize(self, state):
        super().initialize(state)
        state.tracklets = []
        # state.cache_items = []
        self.fr = 0
        if self.reset_id:
            self.id_cnt = 0

    # def finalize(self, state):
    #     tag = 'train'
    #     torch.save(state.cache_items, '/home/toka/code/EOD/ptss/tracker.%s.%d.%s.%d.pkl' %
    #                (tag, env.rank, self.seq_name, self.id_cnt))
    #     del state.cache_items
    #     return NotImplemented

    @property
    def next_id(self):
        self.id_cnt += 1
        if env.distributed:
            return self.id_cnt * env.world_size + env.rank
        return self.id_cnt

    def get_gt(self, image_id):
        import os
        from senseTk.common import TrackSet
        frame_id = int(os.path.basename(image_id).split('.')[0])
        seq_dir = os.path.dirname(os.path.dirname(image_id))
        gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
        seq = os.path.basename(seq_dir)
        self.seq_name = seq
        if self.use_gt and os.path.exists(gt_file):
            if not hasattr(self, 'seq_name') or self.seq_name != seq:
                self.gt = TrackSet(gt_file)
            return seq, self.gt[frame_id]
        else:
            return seq, None

    def collect(self, frame_row, device='cpu'):
        bboxes = []
        for d in frame_row:
            bboxes.append([d.x1, d.y1, d.x2, d.y2, d.uid])
        bboxes = np.array(bboxes)
        bboxes = torch.from_numpy(bboxes).to(device)
        return bboxes

    def preprocess(self, bboxes, info):
        raw_bboxes = bboxes
        bboxes = bboxes.clone()
        scale_h, scale_w = info[2]
        pad_h, pad_w = info[6], info[7]
        bboxes[:, [0, 2]] -= pad_w
        bboxes[:, [0, 2]] /= scale_w
        bboxes[:, [1, 3]] -= pad_h
        bboxes[:, [1, 3]] /= scale_h
        return raw_bboxes, bboxes

    def forward(self, state, inputs):
        self.fr += 1
        bboxes, real_bboxes = self.preprocess(inputs['dt_bboxes'], inputs['image_info'])
        # state.cache_items.append((real_bboxes, embeds))
        seq, gt = self.get_gt(inputs['image_id'])
        self.device = bboxes.device
        if self.two_step:
            thr = 0.1
        else:
            thr = self.output_thr.setdefault(seq, self.output_thr['default'])
        # print(real_bboxes[:10])
        keep = bboxes[:, 4] > thr
        bboxes = bboxes[keep]
        real_bboxes = real_bboxes[keep]
        if gt is not None:
            gtbboxes = self.collect(gt, device=self.device)
            gt_ious = bbox_overlaps(real_bboxes[:, :4], gtbboxes[:, :4])
            g, gi = gt_ious.max(dim=1)
            gids = gtbboxes[:, 4][gi]
            gids[g < 0.5] = -1
            raw_old_gids = [int(t.gid) for t in state.tracklets]
            old_gids = set(raw_old_gids)
            new_gids = {*[int(gid) for gid in gids]}
        dets = [trk.det for trk in state.tracklets]
        # dfeats = [trk.feat for trk in state.tracklets]
        valid = {}
        if len(dets) > 0 and len(bboxes) > 0:
            dets = torch.stack(dets)
            ious = bbox_overlaps(dets[:, :4], bboxes[:, :4])
            # dfeats = torch.stack(dfeats)
            # sims = torch.matmul(dfeats, embeds.T)
            # print(embeds.shape, bboxes.shape)
            # ious[sims > self.high_sim_thr] += 0.6
            # ious[sims < self.low_sim_thr] -= 0.3
            # mx, inds = sims.max(dim=1)
            mx, inds = ious.max(dim=1)
            sinds = sorted(list(range(len(dets))),
                           key=lambda x: (mx[x]) if (state.tracklets[x].last == self.fr - 1) else (mx[x] / 10), reverse=True)
            for i in range(len(dets)):
                i = sinds[i]
                matched = int(inds[i])
                if (mx[i] > self.sim_thr or state.tracklets[i].last != self.fr - 1 and mx[i] * 10 > 0.2) and valid.get(matched, -1) < 0:
                    state.tracklets[i].det = bboxes[matched]
                    # state.tracklets[i].feat = embeds[matched]
                    state.tracklets[i].last = self.fr
                    if gt is not None:
                        if state.tracklets[i].gid != gids[matched]:
                            print(seq, self.fr, 'miss_met', state.tracklets[i].gid, gids[matched], mx[i],
                                  sims[i])
                        state.tracklets[i].gid = gids[matched]
                    valid[matched] = i
                elif gt is not None:
                    if int(state.tracklets[i].gid) in new_gids and int(state.tracklets[i].gid) >= 0:
                        print(seq, self.fr, 'unmet', state.tracklets[i].gid, mx[i], sims[i])
        ids = []
        thr = self.output_thr.setdefault(seq, self.output_thr['default'])
        for j in range(bboxes.size(0)):
            if j not in valid:
                if bboxes[j, 4] > thr or not self.two_step:
                    nid = self.next_id
                    t = Tracklet(bboxes[j], None, nid, self.fr)
                    if gt is not None:
                        t.gid = gids[j]
                        if int(gids[j]) in old_gids and int(gids[j]) >= 0:
                            for i, to in enumerate(state.tracklets):
                                if raw_old_gids[i] == t.gid:
                                    matched = i
                            print(seq, self.fr, self.fr == state.tracklets[matched].last, 'wrongly ascended: ',
                                  gids[j], sims[matched, j], sims[:, j])
                    state.tracklets.append(t)
                else:
                    nid = -1
            else:
                nid = state.tracklets[valid[j]].uid
            ids.append((int(j not in valid), nid))
        new_tracklets = []
        for i, t in enumerate(state.tracklets):
            if self.fr - t.last < self.interval:
                new_tracklets.append(t)
        state.tracklets = new_tracklets
        if len(ids) > 0:
            ids = torch.from_numpy(np.array(ids, dtype=np.float)).to(bboxes.device)
        else:
            ids = bboxes.new_zeros((bboxes.size(0), 2))
        output_dets = torch.cat([bboxes, ids], dim=1)
        output_dets = output_dets[output_dets[:, -1] > 0]
        inputs['dt_bboxes'] = output_dets
        # print(output_dets[:10])
        return inputs
