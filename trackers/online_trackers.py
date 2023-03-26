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
from ..data.mot_dataset import MultiFrameDataset
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


def to_json(data):
    if isinstance(data, (int, float, str, bool)):
        return data
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.bool8):
        return bool(data)
    if isinstance(data, np.str_):
        return str(data)
    if isinstance(data, torch.Tensor):
        return to_json(data.detach().cpu().numpy())
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return float(data)
        else:
            return [to_json(x) for x in data]
    if isinstance(data, (list, tuple)):
        return [to_json(x) for x in data]
    if isinstance(data, dict):
        return {k: to_json(v) for k, v in data.items()}
    return data


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, (list, tuple)):
        return [to_json(x) for x in data]
    if isinstance(data, dict):
        return {k: to_json(v) for k, v in data.items()}
    return data


class Tracklet(object):

    def __init__(self, det, feat, uid, last, cache_range=1, pred_method='linear', crowded_grouping=False, avg_method='mean'):
        self._det = det
        self._feat = feat
        self.uid = uid
        self.last = last
        self.cache_range = cache_range
        self.pred_method = pred_method
        self.crowded_grouping = crowded_grouping
        self.avg_method = avg_method
        self.pos_q = deque(maxlen=self.cache_range)
        self.apr_q = deque(maxlen=self.cache_range)
        self.frs = deque(maxlen=self.cache_range)

    @property
    def det(self):
        return self._det

    @property
    def feat(self):
        return self._feat

    @det.setter
    def det(self, d):
        self._det = d

    @feat.setter
    def feat(self, f):
        self._feat = f

    @property
    def y2(self):
        return float(self._det[3])

    @property
    def avg_feat(self):
        if self.avg_method == 'mean':
            all_feats = torch.stack([a for a in self.apr_q if a is not None])
            return all_feats.mean(dim=0)
        raise NotImplementedError(self.avg_method)

    def pred_det(self, fr=None):
        if self.crowded_grouping:
            occluder = getattr(self, 'occluder', None)
            if occluder is not None and fr - occluder.last == 1 and fr - self.last > 1:
                return occluder.pred_det(fr)
        if self.pred_method == 'linear':
            return self._det
        raise NotImplementedError(self.pred_method)

    def update(self, d, f=None, fr=None):
        if fr is None:
            fr = self.last + 1
        self._det = d
        self._feat = f
        self.last = fr
        self.apr_q.append(f)
        self.pos_q.append(f)
        self.frs.append(fr)

    def iou(self, o):
        return float(bbox_overlaps(self._det.reshape(1, -1), o._det.reshape(1, -1)))


@MODULE_ZOO_REGISTRY.register('ma_online')
class MotionAppearanceOnlineTracker(NoTracking):

    def __init__(self, sim_thr=0.4, output_thr=0.5, keep_thr=0.1, interval=30,
                 pred_det=False, avg_feat=False, reset_id=False, tracklet_cfg=dict(),
                 use_gt=False, save_feats=False, save_feats_dir='./data/',
                 save_feats_type='pkl', feats_tag='data', save_interval=0, **kwargs):
        super().__init__()
        self.sim_thr = sim_thr
        if isinstance(output_thr, (float, int)):
            self.output_thr = {'default': output_thr}
        else:
            self.output_thr = output_thr
        self.keep_thr = keep_thr
        self.interval = interval
        self.pred_det = pred_det
        self.avg_feat = avg_feat
        self.reset_id = reset_id
        self.tracklet_cfg = tracklet_cfg
        self.use_gt = use_gt
        self.save_feats = save_feats
        self.save_feats_dir = save_feats_dir
        self.save_feats_type = save_feats_type
        self.feats_tag = feats_tag
        self.save_interval = save_interval
        logger.info('Setting up %s with %s' % (self.__class__, str(self.tracklet_cfg)))
        self.verbose = False

        self.high_sim_thr = 0.9
        self.low_sim_thr = 0.5

    def initialize(self, state):
        super().initialize(state)
        state.tracklets = []
        if self.save_feats:
            state.cache_items = []
        if not hasattr(state, 'id_cnt'):
            state.id_cnt = 0
        if self.reset_id:
            state.id_cnt = 0
        state.fr = 0

    def save_data(self, state, finalize=False):
        tag = self.feats_tag
        if self.save_feats:
            seq_name = self.seq_name
            if self.save_interval > 0:
                if len(state.cache_items) < self.save_interval and not finalize or len(state.cache_items) == 0:
                    return
                seq_name = seq_name + '.%d_%d' % (self.fr - len(state.cache_items), self.fr)
            else:
                if not finalize:
                    return
            os.makedirs(self.save_feats_dir, exist_ok=True)
            if self.save_feats_type == 'assocjson':
                path = os.path.join(self.save_feats_dir, '%s.%d.%s.assocjson' %
                                    (tag, env.rank, seq_name))
                with open(path, 'w') as fd:
                    json.dump(self.to_assocjson(state.cache_items), fd)
            else:
                path = os.path.join(self.save_feats_dir, '%s.%d.%s.pkl' %
                                    (tag, env.rank, seq_name))
                torch.save(state.cache_items, path)
            state.cache_items = []

    def finalize(self, state):
        self.save_data(state, finalize=True)
        return NotImplemented

    def to_assocjson(self, data):
        data = to_json(data)
        ret = {'initialized': True, 'data': data, 'class': self.__class__.__name__}
        return ret

    def next_id(self, state):
        state.id_cnt += 1
        if env.distributed:
            return state.id_cnt * env.world_size + env.rank
        return state.id_cnt

    def get_gt(self, image_id, vimage_id):
        import os
        from senseTk.common import TrackSet
        frame_id = int(os.path.basename(image_id).split('.')[0])
        seq_dir = os.path.dirname(os.path.dirname(image_id))
        gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
        seq = os.path.basename(seq_dir)
        vseq, vframe = MultiFrameDataset.parse_seq_info(vimage_id, '{root}/{seq}/{fr}.{ext}')
        if self.use_gt and os.path.exists(gt_file):
            if not hasattr(self, 'seq_name') or self.seq_name != vseq:
                self.seq_name = vseq
                self.gt = TrackSet(gt_file)
            return vseq, vframe, self.gt[frame_id]
        else:
            return vseq, vframe, None

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
        if embeds is not None:
            embeds = F.normalize(embeds, dim=1)
        else:
            embeds = bboxes.new_zeros((bboxes.shape[0], 2))
        scale_h, scale_w = info[2]
        pad_h, pad_w = info[6], info[7]
        bboxes[:, [0, 2]] -= pad_w
        bboxes[:, [0, 2]] /= scale_w
        bboxes[:, [1, 3]] -= pad_h
        bboxes[:, [1, 3]] /= scale_h
        return raw_bboxes, bboxes, embeds

    def predict(self, data):
        ious = data['mat']['ious']
        dists = data['mat']['dist']
        sims = data['mat']['sims']
        skipped_frs = data['left']['skipped']
        gt_sims = data['mat'].get('gt_sims', None)
        scores = ious.new_zeros(ious.shape)
        mask = (ious > 0.5) & (dists < 0.2)
        scores[mask] = ious[mask]
        if isinstance(skipped_frs, list):
            skipped_frs = torch.from_numpy(np.array(skipped_frs)).to(ious.device)
        skipped_frs = skipped_frs.unsqueeze(-1).repeat(1, ious.size(1))
        mask = (sims > self.high_sim_thr) & ((dists / skipped_frs) < 0.15)
        scores[mask & (scores < 0.5)] = 0.5 + sims[mask & (scores < 0.5)] / 10
        # ious[sims < self.low_sim_thr] -= 0.3
        # ious = (gt_sims & gt_masks).float()
        if False and gt_sims is not None:
            scores = gt_sims.float()
        return scores

    def decay(self, score, left):
        skipped = left['skipped']
        if skipped > 1:
            return score / 10.
        return score

    def prepare_data(self, real_frame, dets, dfeats, bboxes, embeds, tracklets, o_gids=None, gids=None):
        skipped_frs = [(self.fr - trk.last) for trk in tracklets]
        uids = [trk.uid for trk in tracklets]
        # if self.use_gt:
        #     ooids = [trk.gid for trk in tracklets]
        data = {
            'frame': self.fr, 'real_frame': real_frame,
            'right': {
                'bboxes': bboxes,
            },
            'left': {
                'bboxes': dets,
                'skipped': skipped_frs,
                'uids': uids,
            }
        }
        if self.save_feats_type != 'assocjson':
            data['right']['embeds'] = embeds
            data['left']['embeds'] = dfeats
        ious = bbox_overlaps(dets[:, :4], bboxes[:, :4])
        ndist = bbox_dist(dets[:, :4], bboxes[:, :4])
        sims = torch.matmul(dfeats, embeds.T)
        data['mat'] = {
            'ious': ious,
            'dist': ndist,
            'sims': sims,
        }
        if o_gids is not None and gids is not None:
            gt_sims = o_gids.reshape(-1, 1) == gids.reshape(1, -1)
            gt_masks = (o_gids >= 0).reshape(-1, 1) | (gids >= 0).reshape(1, -1)
            data['right']['gids'] = gids
            data['left']['gids'] = o_gids
            data['mat']['gt_sims'] = gt_sims
            data['mat']['gt_masks'] = gt_masks
            # mask = o_gids == 29
            # if env.rank == 3:
            #     for u, v, w in zip(o_gids, ooids, uids):
            #         builtin_print(u, v, w)
            #     # builtin_print(ooids)
            #     builtin_print(ious[mask])
        return data

    def assign(self, scores, data):
        top_k = min(scores.size(1), 5)
        mx, r_inds = scores.topk(top_k, dim=1)
        l_inds = torch.arange(scores.size(0)).to(scores.device).reshape(-1, 1).repeat(1, top_k)
        inds = torch.stack([l_inds, r_inds]).permute(1, 2, 0).reshape(-1, 2)
        mx = mx.reshape(-1)
        sinds = sorted(list(range(mx.size(0))),
                       key=lambda x: self.decay(mx[x], {k: data['left'][k][int(inds[x][0])] for k in data['left']}), reverse=True)

        valid_r = {}
        valid_l = {}
        ma = []
        l = set(range(scores.size(0)))
        r = set(range(scores.size(1)))
        for i in range(mx.size(0)):
            i = sinds[i]
            li, ri = map(int, inds[i])
            if mx[i] > self.sim_thr and valid_l.get(li, -1) < 0 and valid_r.get(ri, -1) < 0:
                ma.append((li, ri, float(mx[i])))
                valid_l[li] = ri
                valid_r[ri] = li
                l.remove(li)
                r.remove(ri)
        return ma, list(l), list(r), valid_l, valid_r

    def forward(self, state, inputs):
        state.fr += 1
        self.fr = state.fr
        bboxes, real_bboxes, embeds = self.preprocess(
            inputs['dt_bboxes'], inputs.get('id_embeds', None), inputs['image_info'])
        seq, real_frame, gt = self.get_gt(inputs['image_id'], inputs['vimage_id'])
        self.device = bboxes.device
        keep = bboxes[:, 4] > self.keep_thr
        bboxes = bboxes[keep]
        embeds = embeds[keep]
        real_bboxes = real_bboxes[keep]
        if gt is not None:
            gtbboxes = self.collect(gt, device=self.device)
            gt_ious = bbox_overlaps(real_bboxes[:, :4], gtbboxes[:, :4])
            g, gi = gt_ious.max(dim=1)
            gids = gtbboxes[:, 4][gi].int()
            gids[g < 0.5] = -1
            raw_old_gids = [int(t.gid) for t in state.tracklets]
            o_gids = torch.from_numpy(np.array(raw_old_gids)).to(gids.device)
            old_gids = set(raw_old_gids)
            new_gids = {*[int(gid) for gid in gids]}
        if self.pred_det:
            dets = [trk.pred_det(self.fr) for trk in state.tracklets]
        else:
            dets = [trk.det for trk in state.tracklets]
        if self.avg_feat:
            dfeats = [trk.avg_feat for trk in state.tracklets]
        else:
            dfeats = [trk.feat for trk in state.tracklets]

        valid = {}
        valid_l = {}
        if len(dets) > 0 and len(bboxes) > 0:
            dets = torch.stack(dets)
            dfeats = torch.stack(dfeats)
            # # print(embeds.shape, bboxes.shape)
            if not self.use_gt:
                data = self.prepare_data(real_frame, dets, dfeats, bboxes, embeds, state.tracklets)
            else:
                data = self.prepare_data(real_frame, dets, dfeats, bboxes, embeds,
                                         state.tracklets, o_gids=o_gids, gids=gids)

            scores = self.predict(data)
            ma, l, r, valid_l, valid_r = self.assign(scores, data)
            valid.update(valid_r)
            if self.save_feats:
                state.cache_items.append(to_numpy(data))
                self.save_data(state)
            for i, matched, matched_score in ma:
                prev_fr = state.tracklets[i].last
                state.tracklets[i].update(bboxes[matched], embeds[matched], fr=self.fr)
                if gt is not None:
                    if self.verbose and state.tracklets[i].gid != gids[matched]:
                        print(seq, self.fr, 'miss_met', self.fr - prev_fr, (int(state.tracklets[i].uid), int(state.tracklets[i].gid), int(gids[matched])), matched_score,
                              ious[i, matched], ndist[i, matched], sims[i, matched])
                    if state.tracklets[i].uid == 147 and env.rank == 3 and False:
                        builtin_print(gids[matched])
                    state.tracklets[i].gid = gids[matched]
            for i in l:
                if gt is not None:
                    if state.tracklets[i].uid == 147 and env.rank == 3 and False:
                        builtin_print(data['mat']['ious'][i])
                        builtin_print(scores[i])
                    for mm, k in enumerate(gids):
                        if k == state.tracklets[i].gid:
                            break
                    if self.verbose and self.fr - state.tracklets[i].last > 5 and int(state.tracklets[i].gid) in new_gids and int(state.tracklets[i].gid) >= 0:
                        print(seq, self.fr, 'unmet', self.fr - state.tracklets[i].last, (int(state.tracklets[i].gid), valid.get(mm, '---')),
                              ious[i, matched], ndist[i, matched], sims[i, matched])
        ids = []
        thr = self.output_thr.setdefault(seq, self.output_thr['default'])
        for j in range(bboxes.size(0)):
            track_conf = 0.
            if j not in valid:
                if bboxes[j, 4] > thr:
                    nid = self.next_id(state)
                    t = Tracklet(bboxes[j], embeds[j], nid, self.fr, **self.tracklet_cfg)
                    if gt is not None:
                        t.gid = gids[j]
                        if self.verbose and int(gids[j]) in old_gids and int(gids[j]) >= 0:
                            matched = -1
                            for i in range(len(raw_old_gids)):
                                to = state.tracklets[i]
                                if raw_old_gids[i] == t.gid and i not in valid_l:
                                    matched = i
                            if matched >= 0:
                                if env.rank == 3:
                                    builtin_print(seq, self.fr, 'wrongly ascended: ', self.fr - state.tracklets[matched].last,
                                                  int(state.tracklets[matched].uid), raw_old_gids[matched],
                                                  int(gids[j]), data['mat']['ious'][matched, j], data['mat']['sims'][matched, j])
                    state.tracklets.append(t)
                    track_conf = 1.
                else:
                    nid = self.next_id(state)
                    track_conf = 0
            else:
                nid = state.tracklets[valid[j]].uid
                track_conf = 1.
            ids.append((track_conf, nid))
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
        assert xgb_flag, 'xgboost import failed'

    def predict(self, data):
        ious = data['mat']['ious']
        dists = data['mat']['dist']
        sims = data['mat']['sims']
        intvals = data['left']['skipped']
        gt_sims = data['mat'].get('gt_sims', None)
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

    def decay(self, score, left):
        return score


@MODULE_ZOO_REGISTRY.register('pure_a')
class PureAppearanceTracker(MotionAppearanceOnlineTracker):

    def __init__(self, sim_thr=0.8, interval=30, **kwargs):
        super().__init__(keep_thr=0.1, output_thr=0.1, sim_thr=sim_thr, interval=interval, **kwargs)

    def predict(self, data):
        return data['mat']['sims']

    def decay(self, score, left):
        skipped = left['skipped']
        if skipped > 1:
            return score - 1
        return score


@MODULE_ZOO_REGISTRY.register('pure_m')
class PureMotionTracker(MotionAppearanceOnlineTracker):

    def predict(self, data):
        return data['mat']['ious']
        return ious

    def decay(self, score, left):
        skipped = left['skipped']
        if skipped > 1:
            return score / 10.
        return score
