import numpy as np
from senseTk.functions import LAP_Matching
from senseTk.common import Det, TrackSet
from scipy.optimize import linear_sum_assignment
import copy
from collections import deque
import math
import torch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    if isinstance(bboxes1, np.ndarray):
        bboxes1 = torch.from_numpy(bboxes1)
    if isinstance(bboxes2, np.ndarray):
        bboxes2 = torch.from_numpy(bboxes2)

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3]
                                                   - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3]
                                                       - bboxes2[:, 1])
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3]
                                                   - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3]
                                                       - bboxes2[:, 1])
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


class Tracklet:
    def __init__(self, det: Det, feat) -> None:
        self.det = det
        self.feat = feat
        self.locs = deque(maxlen=9)

    @property
    def xyxy(self):
        ret = [self.det.x1, self.det.y1, self.det.x2, self.det.y2]
        return np.array(ret, dtype=np.float)

    @property
    def track_score(self):
        return 1. / (1. + self.cum_score)

    def clone(self):
        return copy.deepcopy(self)

    @property
    def motion_score(self):
        lmotions = []
        pre = -1
        pre_pos = None
        for ptr, t in enumerate(self.actions):
            if t[3] is not None:
                x1, y1, x2, y2, cf = t[2]
                if pre >= 0:
                    dx1, dy1, dx2, dy2 = x1 - \
                        pre_pos[0], y1 - pre_pos[1], x2 - \
                        pre_pos[2], y2 - pre_pos[3]
                    dx1, dy1, dx2, dy2 = map(
                        lambda x: x / (ptr - pre), (dx1, dy1, dx2, dy2))
                    lmotions.append((dx1, dy1, dx2, dy2))
                pre = ptr
                pre_pos = x1, y1, x2, y2
        if len(lmotions) > 0:
            mscore = np.array(lmotions).std(axis=1).mean()
        else:
            mscore = 0.
        return float(1 - mscore / 10) * 0.5


class BaseTracker:
    def __init__(self, use_gt=False, loc_pred=False, **kwargs) -> None:
        self._use_gt = use_gt
        self._loc_pred = loc_pred
        self.avg_deltas = deque(maxlen=25)
        self.init()

    def init(self):
        self.tracklets = []
        self.fr = 0
        self.outputs = TrackSet()
        self.idc = 0

    def get_shape_sim(self, x_row, y_row):
        x1 = np.maximum(x_row[:, 0].reshape(-1, 1), y_row[:, 0].reshape(1, -1))
        x2 = np.minimum(x_row[:, 2].reshape(-1, 1), y_row[:, 2].reshape(1, -1))
        y1 = np.maximum(x_row[:, 1].reshape(-1, 1), y_row[:, 1].reshape(1, -1))
        y2 = np.minimum(x_row[:, 3].reshape(-1, 1), y_row[:, 3].reshape(1, -1))
        iw = np.maximum(x2 - x1, 0)
        ih = np.maximum(y2 - y1, 0)
        intersec = iw * ih
        xa = np.maximum(x_row[:, 2] - x_row[:, 0], 0) * np.maximum(
            x_row[:, 3] - x_row[:, 1], 0)
        ya = np.maximum(y_row[:, 2] - y_row[:, 0], 0) * np.maximum(
            y_row[:, 3] - y_row[:, 1], 0)
        a = xa.reshape(-1, 1) + ya.reshape(1, -1)
        return intersec / (a - intersec)

    def get_sim(self, framedata):
        y_d = framedata['dets']
        # y_f = framedata['feats']
        dets_collect = [self.predict_loc(t) for t in self.tracklets]
        # feats_collect = [t.feat for t in self.tracklets]
        x_d = np.stack(dets_collect)
        # x_f = np.stack(feats_collect)
        shape_sim = self.get_shape_sim(x_d, y_d)
        # feats_sim = self.get_feat_sim(x_f, y_f)
        # sim = shape_sim * feats_sim
        sim = shape_sim
        sim[sim < 0.3] = 0
        return sim

    def get_gt_sim(self, framedata):
        assert self._current_gt is not None
        gt_dets = {d.uid: d for d in self._current_gt}
        sims = []
        for t in self.tracklets:
            sims.append([])
            for i in range(len(framedata['dets'])):
                if t.gt_id not in gt_dets:
                    sims[-1].append(0)
                else:
                    gdet = gt_dets[t.gt_id]
                    x1, y1, x2, y2 = map(float, framedata['dets'][i])
                    ddet = Det(x1, y1, x2 - x1, y2 - y1)
                    iou = ddet.iou(gdet)
                    if iou > 0.5:
                        sims[-1].append(iou
                                        + (1 - t.det.uid / self.idc) / 100)
                    else:
                        sims[-1].append(0)
        return np.array(sims, dtype=np.float32)

    def check_reliable(self, tracklet: Tracklet):
        if len(tracklet.locs) == tracklet.locs.maxlen:
            cx = tracklet.locs[0][0] + tracklet.locs[-1][0]
            cy = tracklet.locs[0][1] + tracklet.locs[-1][1]
            cx /= 2
            cy /= 2
            dx = cx - tracklet.locs[tracklet.locs.maxlen // 2][0]
            dy = cy - tracklet.locs[tracklet.locs.maxlen // 2][1]
            bias = (dx**2 + dy**2)**0.5 / tracklet.det.w
            if bias < 0.2:
                return True
        return False

    def predict_loc(self, tracklet: Tracklet):
        if self.check_reliable(tracklet):
            vx = (tracklet.locs[-1][0] - tracklet.locs[0]
                  [0]) / (tracklet.locs.maxlen - 1)
            vy = (tracklet.locs[-1][1] - tracklet.locs[0]
                  [1]) / (tracklet.locs.maxlen - 1)
            dt = min(self.fr - tracklet.last_update, 5)
            cx = tracklet.det.cx + dt * vx
            cy = tracklet.det.cy + dt * vy
            w = tracklet.det.w
            h = tracklet.det.h
            ret = [cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2.]
            return np.array(ret, dtype=np.float)
        else:
            return tracklet.xyxy

    def dump(self, path):
        with open(path, 'w') as fd:
            self.outputs.dump(fd)

    @property
    def new_id(self):
        self.idc += 1
        return self.idc

    def update_binary(self, i, j, framedata):
        self.tracklets[i].feat = framedata['feats'][j] if 'feats' in framedata else None
        x1, y1, x2, y2 = map(float, framedata['dets'][j])
        w = x2 - x1
        h = y2 - y1
        d = Det(x1,
                y1,
                w,
                h,
                cls=1,
                confidence=float(framedata['confs'][j]),
                uid=self.tracklets[i].det.uid,
                fr=self.fr,
                status=1)
        self.cur_delta[0] += (d.cx - self.tracklets[i].det.cx) / \
            (self.fr - self.tracklets[i].last_update)
        self.cur_delta[1] += (d.cy - self.tracklets[i].det.cy) / \
            (self.fr - self.tracklets[i].last_update)
        self.tracklets[i].det = d
        self.tracklets[i].last_update = self.fr
        self.tracklets[i].locs.append(((x1 + x2) / 2., (y1 + y2) / 2.))

    def update_unary(self, i):
        self.tracklets[i].det.conf *= 0.9
        self.tracklets[i].det.status = 0
        # t.locs.append(None)

    def append_new(self, j, framedata):
        x1, y1, x2, y2 = map(float, framedata['dets'][j])
        w = x2 - x1
        h = y2 - y1
        d = Det(x1,
                y1,
                w,
                h,
                cls=1,
                confidence=float(framedata['confs'][j]),
                uid=self.new_id,
                fr=self.fr,
                status=1)
        t = Tracklet(d, framedata['feats'][j] if 'feats' in framedata else None)
        if self._use_gt:
            assert self._current_gt is not None
            gt_id = -1
            mxiou = -1
            for gdet in self._current_gt:
                iou = t.det.iou(gdet)
                if iou > 0.5 and iou > mxiou:
                    mxiou = iou
                    gt_id = gdet.uid
            t.gt_id = gt_id
        t.last_update = self.fr
        t.locs.append(((x1 + x2) / 2., (y1 + y2) / 2.))
        self.tracklets.append(t)

    def dismiss(self):
        new_tracklets = []
        for t in self.tracklets:
            if self.fr - t.last_update < 50:
                new_tracklets.append(t)
        self.tracklets = new_tracklets

    def output(self):
        stable = len(self.avg_deltas) > 3
        if stable:
            deltas = np.array(list(self.avg_deltas)[-4:])
            stds = deltas[:, 0].std() + deltas[:, 1].std()
            # print(deltas, stds)
            # input()
            stable = stds < 1.5
        deltas = self.avg_deltas
        for t in self.tracklets:
            if t.det.status == 1:
                self.outputs.append_data(t.det.copy())
            elif self._loc_pred and (stable and t.det.status == 0 and self.check_reliable(t) and self.fr - t.last_update < 5):
                p = self.predict_loc(t)
                d = t.det.copy()
                d.fr = self.fr
                # print(d.cx, d.cy, '=>', float(
                #     (p[0] + p[2]) / 2.), float((p[1] + p[3]) / 2.))
                d.cx = float((p[0] + p[2]) / 2.)
                d.cy = float((p[1] + p[3]) / 2.)
                self.outputs.append_data(d)

    def __call__(self, framedata, gt=None):
        self._current_gt = gt
        self.fr += 1
        matched = []
        if isinstance(framedata['dets'], (list, tuple)):
            framedata['dets'] = framedata['dets'][0]
        if isinstance(framedata['confs'], (list, tuple)):
            framedata['confs'] = framedata['confs'][0]
        if 'feats' in framedata and isinstance(framedata['feats'], (list, tuple)):
            framedata['feats'] = framedata['feats'][0]
        if 'origin_size' in framedata:
            framedata['dets'][:, 0::2] *= framedata['origin_size'][0]
            framedata['dets'][:, 1::2] *= framedata['origin_size'][1]
        lmiss = set(range(len(self.tracklets)))
        rmiss = set(range(framedata['dets'].shape[0]))
        if lmiss and rmiss:
            if self._use_gt:
                sim = self.get_gt_sim(framedata)
            else:
                sim = self.get_sim(framedata)
            l, r = linear_sum_assignment(sim, maximize=True)
            # print(sim)
            for i, j in zip(l, r):
                if sim[i, j] > 0.01:
                    matched.append((int(i), int(j)))
                    lmiss.remove(int(i))
                    rmiss.remove(int(j))
        self.cur_delta = [0, 0]
        for i, j in matched:
            self.update_binary(i, j, framedata)
        if matched:
            self.cur_delta[0] /= len(matched)
            self.cur_delta[1] /= len(matched)
            self.avg_deltas.append(self.cur_delta)
        for i in lmiss:
            self.update_unary(i)
        for j in rmiss:
            self.append_new(j, framedata)
        self.dismiss()
        self.output()
        self._current_gt = None


class MotionAppearanceTracker(BaseTracker):
    def get_shape_sim(self, x_row, y_row):
        def get_wh(row):
            return row[:, 2] - row[:, 0], row[:, 3] - row[:, 1]

        def get_xy(row):
            return (row[:, 2] + row[:, 0]) / 2., (row[:, 3] + row[:, 1]) / 2.

        xw, xh = get_wh(x_row)
        yw, yh = get_wh(y_row)
        xx, xy = get_xy(x_row)
        yx, yy = get_xy(y_row)
        xr1 = xh / (xw + 1e-9)
        yr1 = yh / (yw + 1e-9)
        dr1 = np.abs(xr1.reshape(-1, 1) - yr1.reshape(1, -1))

        dr2 = np.sqrt((xx.reshape(-1, 1) - yx.reshape(1, -1))**2
                      + (xy.reshape(-1, 1) - yy.reshape(1, -1))**2) / (
                          xw.reshape(-1, 1) + 1e-9)
        # print(dr2.shape, xw.shape, dr1.shape, xw.shape, xh.shape)
        return (1 - np.minimum(np.arctan(dr1 / 2.), 1)) * (
            1 - np.minimum(np.arctan(dr2 / 4.), 1))

    def get_feat_sim(self, x_row, y_row):
        nx = np.sqrt(np.sum((x_row * x_row), axis=1))
        x_row = x_row / nx.reshape(-1, 1)
        ny = np.sqrt(np.sum((y_row * y_row), axis=1))
        y_row = y_row / ny.reshape(-1, 1)
        aff = np.matmul(x_row, y_row.T)
        return aff

    def get_sim(self, framedata):
        y_d = framedata['dets']
        y_f = framedata['feats']
        dets_collect = [self.predict_loc(t) for t in self.tracklets]
        feats_collect = [t.feat for t in self.tracklets]
        x_d = np.stack(dets_collect)
        x_f = np.stack(feats_collect)
        shape_sim = self.get_shape_sim(x_d, y_d)
        feats_sim = self.get_feat_sim(x_f, y_f)
        sim = shape_sim * feats_sim
        sim[sim < 0.03] = 0
        return sim


class BaseMHTracker:

    _max_slots = 256
    _max_extend = 4
    _max_len = 32
    _sink_thr = 0.7

    def __init__(self, use_gt=False, ** kwargs) -> None:
        self._use_gt = use_gt
        self.init()

    def init(self):
        self.tracklets = []
        self.fr = 0
        self.outputs = TrackSet()
        self.idc = 0

    def dump(self, path):
        with open(path, 'w') as fd:
            self.outputs.dump(fd)

    @staticmethod
    def sigmoid_inv(s_):
        return math.log(s_ + 1e-12) - math.log(1 - s_ + 1e-12)

    @property
    def new_id(self):
        self.idc += 1
        return self.idc

    def gen_tracklet(self, framedata, x, action_score):
        x1, y1, x2, y2 = map(float, framedata['dets'][x])
        conf = float(framedata['confs'][x])
        feat = framedata['feats'][x]
        d = Det(x1, y1, x2 - x1, y2 - y1, confidence=conf,
                uid=self.new_id, fr=self.fr, status=1)
        t = Tracklet(d, feat)
        t.actions = deque(maxlen=self._max_len)
        t.gts = deque(maxlen=self._max_len)
        t.gts_keys = {}
        action_score = self.sigmoid_inv(action_score)
        t.actions.append((x, action_score, (x1, y1, x2, y2, conf), feat))
        t.cum_score = math.exp(- action_score)
        gid = -1
        if self._gtid_for_dets is not None:
            gid = self._gtid_for_dets[x]
        t.gts.append(gid)
        t.gts_keys[gid] = 1
        t.pre_score = action_score
        return t

    def run_tracklet(self, tracklet: Tracklet, score, action_score):
        t = tracklet.clone()
        action_score = self.sigmoid_inv(action_score)
        if len(t.actions) == t.actions.maxlen:
            t.cum_score -= math.exp(-t.actions[0][1])
        t.cum_score += math.exp(- action_score)
        t.actions.append((None, action_score, None, None))
        t.det.status = 0
        t.det.conf *= 0.1
        gid = -2
        t.gts.append(gid)
        t.gts_keys[-2] = t.gts_keys.get(-2, 0) + 1
        return t

    def merge_tracklet(self, tracklet, framedata, x, score, action_score):
        t = tracklet.clone()
        x1, y1, x2, y2 = map(float, framedata['dets'][x])
        conf = float(framedata['confs'][x])
        feat = framedata['feats'][x]
        action_score = self.sigmoid_inv(action_score)
        if len(t.actions) == t.actions.maxlen:
            t.cum_score -= math.exp(-t.actions[0][1])
        t.cum_score += math.exp(- action_score)
        t.actions.append((x, action_score, (x1, y1, x2, y2, conf), feat))
        t.feat = feat
        t.det.status = 1
        t.det.fr = self.fr
        t.det.x1 = x1
        t.det.x2 = x2
        t.det.y1 = y1
        t.det.y2 = y2
        # t.det.conf = 0.8 * max(t.det.conf, conf) + 0.2 * conf
        t.det.conf = conf
        gid = -1
        if self._gtid_for_dets is not None:
            gid = self._gtid_for_dets[x]
        t.gts.append(gid)
        t.gts_keys[gid] = t.gts_keys.get(gid, 0) + 1
        t.pre_score = score
        return t

    def add_to_tracklet(self, u, v, score, action_score, framedata):
        if u < 0:
            return self.gen_tracklet(framedata, v, action_score)
        elif v < 0:
            return self.run_tracklet(self.tracklets[u], score, action_score)
        else:
            return self.merge_tracklet(self.tracklets[u], framedata, v, score, action_score)

    def dismiss(self):
        new_tracklets = []
        for t in self.tracklets:
            if t.det.conf >= 0.1:
                new_tracklets.append(t)
        self.tracklets = new_tracklets

    def output(self):
        for t in self.tracklets:
            if t.det.conf > 0.4:
                self.outputs.append_data(t.det.copy())

    def get_shape_sim(self, x_row, y_row):
        def get_wh(row):
            return row[:, 2] - row[:, 0], row[:, 3] - row[:, 1]

        def get_xy(row):
            return (row[:, 2] + row[:, 0]) / 2., (row[:, 3] + row[:, 1]) / 2.

        xw, xh = get_wh(x_row)
        yw, yh = get_wh(y_row)
        xx, xy = get_xy(x_row)
        yx, yy = get_xy(y_row)
        xr1 = xh / (xw + 1e-9)
        yr1 = yh / (yw + 1e-9)
        dr1 = np.abs(xr1.reshape(-1, 1) - yr1.reshape(1, -1))

        dr2 = np.sqrt((xx.reshape(-1, 1) - yx.reshape(1, -1))**2
                      + (xy.reshape(-1, 1) - yy.reshape(1, -1))**2) / (
                          xw.reshape(-1, 1) + 1e-9)
        # print(dr2.shape, xw.shape, dr1.shape, xw.shape, xh.shape)
        return (1 - np.minimum(np.arctan(dr1 / 2.), 1)) * (
            1 - np.minimum(np.arctan(dr2 / 4.), 1))

    def get_feat_sim(self, x_row, y_row):
        nx = np.sqrt(np.sum((x_row * x_row), axis=1))
        x_row = x_row / nx.reshape(-1, 1)
        ny = np.sqrt(np.sum((y_row * y_row), axis=1))
        y_row = y_row / ny.reshape(-1, 1)
        aff = np.matmul(x_row, y_row.T)
        return aff

    def get_sim000(self, framedata):
        y_d = framedata['dets']
        y_f = framedata['feats']
        dets_collect = [t.xyxy for t in self.tracklets]
        feats_collect = [t.feat for t in self.tracklets]
        if len(y_d) > 0 and len(dets_collect) > 0:
            x_d = np.stack(dets_collect)
            x_f = np.stack(feats_collect)
            shape_sim = self.get_shape_sim(x_d, y_d)
            feats_sim = self.get_feat_sim(x_f, y_f)
            sim = shape_sim * feats_sim
        else:
            sim = np.zeros((len(dets_collect), len(y_d)))
        sim_with_virtual = np.full(
            (sim.shape[0], sim.shape[1] + 1), self._sink_thr)
        sim_with_virtual[:, :-1] = sim
        sim_with_virtual -= self._sink_thr
        sim = np.exp(sim_with_virtual * 8)
        # print(sim_with_virtual[:5])
        # print(sim_with_virtual[-5:])
        # print(sim[-10:])
        if sim.shape[0] > 0:
            mxs = sim.max(axis=0)
            init_sc = 1 / (np.exp(mxs) + 1)
            init_sc = init_sc.reshape(1, sim.shape[1])
        else:
            init_sc = np.ones((1, sim.shape[1]))
        sim = sim / np.sum(sim, axis=1).reshape(-1, 1)
        sim = np.concatenate([sim, init_sc], axis=0)
        # sim[-1, :] *= sim.shape[1]
        # sim[:, -1] *= sim.shape[0]
        # sim[sim < 0.03] = 0
        # print(sim)
        # input()
        return sim

    def get_sim(self, framedata):
        neg_penalty = 5
        length_penalty = 0.3
        n = len(self.tracklets)
        m = len(framedata['dets'])
        rfeats = torch.from_numpy(framedata['feats'])
        rconfs = torch.from_numpy(framedata['confs'])
        all_lfeats = []
        all_lmasks = []
        for i in range(n):
            lfeats = [t[3]
                      for t in self.tracklets[i].actions if t[3] is not None]
            lmasks = [(1)
                      for t in self.tracklets[i].actions if t[3] is not None]
            while len(lfeats) < self._max_len:
                lfeats.insert(0, lfeats[-1])
                lmasks.insert(0, 0)
            lfeats = np.stack(lfeats)
            lmasks = np.array(lmasks)
            all_lfeats.append(lfeats)
            all_lmasks.append(lmasks)
        init_sc = torch.zeros(1, m)
        if n > 0:
            all_lfeats = torch.from_numpy(np.stack(all_lfeats))
            all_lmasks = torch.from_numpy(np.stack(all_lmasks))
            all_lfeats = all_lfeats.reshape(n * self._max_len, -1)
            sims = all_lfeats.matmul(rfeats.permute(
                1, 0)).reshape(n, self._max_len, m)
            sims_raw = sims
            sims = (sims - self._sink_thr) / (1 - self._sink_thr)
            sims[sims < 0] *= neg_penalty
            sims = (all_lmasks.unsqueeze(-1) * sims).sum(dim=1)
            # if len(sims) > 21:
            #     print(sims[5])
            #     print(sims[9])
            #     print(sims[14])
            #     print(sims[17])
            #     print(sims[18])
            #     print(sims[20])
            #     print(sims_raw[20, :, 6])
            #     # print(sims[19])
            #     # print(sims[21])
            #     input()
            # sims = sims * (rconfs.sqrt()).reshape(1, m)
            sims = torch.cat([sims, init_sc], dim=0)
        else:
            sims = init_sc
        # sims += rconfs.reshape(1, m)
        sims += 1
        # miss_sc = torch.from_numpy(
        #     np.array([t.pre_score for t in self.tracklets] + [0])).reshape(-1, 1)
        if n > 0:
            all_lfeats = all_lfeats.reshape(n, self._max_len, -1)
            miss_sc = torch.bmm(
                all_lfeats, all_lfeats[:, -1].reshape(n, -1, 1)).squeeze(-1)
            miss_sc = (miss_sc - self._sink_thr) / (1 - self._sink_thr)
            miss_sc[miss_sc < 0] *= neg_penalty
            # if len(sims) > 20:
            #     print('missss')
            #     print(miss_sc[20])
            # print(miss_sc.shape, all_lmasks.shape)
            miss_sc = (miss_sc * all_lmasks).sum(dim=1) - \
                all_lmasks.sum(dim=1) * length_penalty
            #     all_lmasks[:, 0] - 0.3
            # miss_sc = miss_sc.sum(dim=1) - 0.3
            # print(miss_sc.shape)
        else:
            miss_sc = torch.zeros((0,))
            # print(miss_sc.shape)
        miss_sc = torch.cat([miss_sc, torch.zeros((1,))]).unsqueeze(-1)
        sims = torch.cat([sims, miss_sc], dim=1)
        # if len(sims) > 21:
        #     print(sims[5])
        #     print(sims[9])
        #     print(sims[14])
        #     print(sims[17])
        #     print(sims[18])
        #     print(sims[20])
        #     # print(sims[19])
        #     # print(sims[21])
        #     print(sims[-1])
        #     input()
        return sims

    def has_intersection(self, x, y):
        if x[-1] >= 0 and x[-1] == y[-1]:
            return True
        x = x[0]
        y = y[0]
        if x >= 0 and y >= 0:
            if x == y or self.tracklets[x].det.uid == self.tracklets[y].det.uid:
                return True
            if self.tracklets[y].actions[-1][0] is not None and self.tracklets[y].actions[-1][0] == self.tracklets[x].actions[-1][0]:
                return True
            com = min(len(self.tracklets[y].actions),
                      len(self.tracklets[x].actions))
            if self.tracklets[y].actions[-com][0] is not None and self.tracklets[y].actions[-com][0] == self.tracklets[x].actions[-com][0]:
                return True
            # # self.tracklets[x].actions:
            # for fr in range(self.fr - 10, self.fr + 1):
            #     if fr in self.tracklets[y].actions and fr in self.tracklets[x].actions:
            #         if self.tracklets[x].actions[fr][0] is not None and self.tracklets[x].actions[fr][0] == self.tracklets[y].actions[fr][0]:
            #             return True
        return False

    def do_prunning(self, hyps):
        pruned = []
        for one in hyps:
            flag = True
            if one.actions[0][0] is None and one.actions[-1][0] is None:
                flag = False
                for k in range(len(one.actions)):
                    if one.actions[k][0] is not None:
                        flag = True
                        break
            if flag:
                pruned.append(one)
        return pruned

    def inspect(self, u, v, framedata):
        scores = []
        for i in self.tracklets[u].actions:
            if i[3] is not None:
                score = np.sum(i[3] * framedata['feats'][v])
                scores.append(score)
        return scores

    def __call__(self, framedata, gt=None):
        import time
        a0 = time.time()
        self.fr += 1
        print(self.fr, end='\r')
        if isinstance(framedata['dets'], (list, tuple)):
            framedata['dets'] = framedata['dets'][0]
        if isinstance(framedata['confs'], (list, tuple)):
            framedata['confs'] = framedata['confs'][0]
        if isinstance(framedata['feats'], (list, tuple)):
            framedata['feats'] = framedata['feats'][0]
        if 'origin_size' in framedata:
            framedata['dets'][:, 0::2] *= framedata['origin_size'][0]
            framedata['dets'][:, 1::2] *= framedata['origin_size'][1]
        a1 = time.time()
        if gt is not None:
            gtdets = np.array([(gd.x1, gd.y1, gd.x2, gd.y2) for gd in gt])
            gtids = np.array([gd.uid for gd in gt])
            ious = bbox_overlaps(framedata['dets'], gtdets)
            self._gtid_for_dets = np.full((framedata['dets'].shape[0]), -1)
            s, i = ious.max(dim=1)
            mask = (s > 0.5).numpy()
            self._gtid_for_dets[mask] = gtids[i.numpy()[mask]]
            # print(*enumerate(self._gtid_for_dets))
            # for i, t in enumerate(self.tracklets):
            #     print(i, t.gts, t.det.uid)
            # if not hasattr(self, 'gt_queues'):
            #     self.gt_queues = {}
            # marks = set()
            # for gd in gt:
            #     if gd.uid not in self.gt_queues:
            #         self.gt_queues[gd.uid] = deque(maxlen=self._max_len)
            #     self.gt_queues[gd.uid].append(1)
            #     marks.add(gd.uid)
            # for gid in self.gt_queues:
            #     if gid not in marks:
            #         self.gt_queues[gid].append(0)
            # print(self._gtid_for_dets)
            # input()
        else:
            self._gtid_for_dets = None
        a2 = time.time()
        sim = self.get_sim(framedata)
        a3 = time.time()
        # print(sim[:3])
        # input()
        new_hyps = []
        for i in range(-1, sim.shape[0] - 1):
            tmp = []
            mscore = 1. if i < 0 else self.tracklets[i].motion_score
            for j in range(-1, sim.shape[1] - 1):
                if i < 0 and j < 0:
                    continue
                score = float(sim[i, j])
                if i >= 0:
                    score = score + mscore
                # if i == 26 and j == 23:
                #     print(u, v, score, valid[i])
                # if j >= 0:
                #     score *= framedata['confs'][j]
                action_score = max(float(score), 0) / (self._max_len + 1) / 2
                # if i >= 0:
                #     score *= self.tracklets[i].track_score
                # if j >= 0:
                #     print(
                #         list(self.tracklets[i].gts), self._gtid_for_dets[j], score, action_score, self.tracklets[i].track_score)
                if self._use_gt and False:
                    rid = -2
                    score = 0
                    if j >= 0:
                        rid = self._gtid_for_dets[j]
                        score = 1
                    if i >= 0:
                        for k, v in self.tracklets[i].gts_keys.items():
                            if rid == -2 and k >= 0:
                                rid = k
                            if k == rid and rid >= 0:
                                score += v
                            if k != rid and k != -2:
                                score -= v
                tmp.append((i, j, score, action_score))
            if i != -1:
                tmp = sorted(tmp, key=lambda x: x[2], reverse=True)
                tmp = tmp[:self._max_extend]
            new_hyps += tmp
            # input()
            # print(self.tracklets[i].det.uid if i >=
            #       0 else i, j, score, action_score)
        a4 = time.time()
        new_hyps = sorted(new_hyps, key=lambda x: x[2], reverse=True)
        # new_hyps = self.do_prunning(new_hyps)
        a5 = time.time()
        valid = [True] * len(new_hyps)
        new_tracklets = []
        # print('---'*10)
        for i, (u, v, score, action_score) in enumerate(new_hyps):
            if valid[i]:
                new_tracklets.append(
                    self.add_to_tracklet(u, v, score, action_score, framedata))
                # print(new_tracklets[-1].det.uid, v, action_score)
                for j in range(i + 1, len(new_hyps)):
                    if valid[j] and self.has_intersection((u, v), new_hyps[j][:2]):
                        valid[j] = False
        a6 = time.time()
        old_tracklets = self.tracklets
        self.tracklets = new_tracklets
        self.output()
        self.tracklets = old_tracklets
        split_ = len(new_tracklets)
        rec_row = {}
        rec_col = {}
        for i, flag in enumerate(valid):
            if not flag and len(new_tracklets) < self._max_slots:
                u, v, sc, ac = new_hyps[i]
                if sc < 0:
                    continue
                flag = True
                if u >= 0:
                    uid = self.tracklets[u].det.uid
                    if uid not in rec_row:
                        rec_row[uid] = 0
                    if rec_row[uid] + 1 > self._max_extend:
                        flag = False
                if v >= 0:
                    if v not in rec_col:
                        rec_col[v] = 0
                    if rec_col[v] + 1 > self._max_extend:
                        flag = False
                if flag:
                    if u >= 0:
                        rec_row[uid] += 1
                    if v >= 0:
                        rec_col[v] += 1
                    new_tracklets.append(
                        self.add_to_tracklet(*(new_hyps[i]), framedata))
        if len(new_tracklets) > self._max_slots:
            new_tracklets = new_tracklets[:self._max_slots]
        new_tracklets = self.do_prunning(new_tracklets)
        self.tracklets = new_tracklets
        a7 = time.time()
        # for aii in range(1, 8):
        #     print('%d~%d: ' % (aii-1, aii), locals()
        #           ['a%d' % aii] - locals()['a%d' % (aii-1)], end=' ')
        # print()
        # print('time: %.3f' % (a7 - a0))
        # self.dismiss()
        # for _, one in enumerate(self.tracklets):
        #     print(list(one.gts), one.det.uid, one.det.conf, one.motion_score,
        #           (self._max_len + 1) / (1 + math.exp(-one.actions[-1][1])))
        #     if _ + 1 == split_:
        #         print('-'*30)
        #         break
        # print('end---')
        # input()
        # print('---'*10)
        # print('---'*10)
