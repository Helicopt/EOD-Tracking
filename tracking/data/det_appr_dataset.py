import enum
import torch
import math
from torch.utils.data import Dataset, DataLoader
from senseTk.common import TrackSet, Det
import random
from functools import lru_cache
import numpy as np
import pickle
import yaml
import os

seed = 2077
random.seed(seed)
np.random.seed(seed)


@lru_cache(maxsize=None)
def Cc(n, m):
    assert 0 <= m <= n
    if n - m < m:
        m = n - m
    if m == 0:
        return 1
    return Cc(n, m - 1) * (n - m + 1) / m


def BCurve(points, m):
    n = len(points)
    if m <= 1:
        return points
    ret = []
    for i in range(m):
        t = i / (m - 1)
        x, y = 0, 0
        for j in range(n):
            px, py = points[j]
            c = Cc(n - 1, j) * (1 - t)**(n - 1 - j) * t**j
            x += c * px
            y += c * py
        ret.append((x, y))
    return ret


def randsign():
    return 1 if random.randint(0, 1) == 0 else -1


class Calibrator:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.f = 2
        self.scale = 750
        self.w, self.h = 1920, 1080
        self.xo = x - self.w / self.scale / 2
        self.zo = z
        self.fo = np.array([x, y, z], dtype=np.float32)

    def __call__(self, points):
        rp = np.array(points, dtype=np.float32)

        d = self.fo - rp
        t = -self.f / d[:, 1]
        ip = self.fo + t.reshape(-1, 1) * d
        ip = ip[:, 0::2]
        ip[:, 0] -= self.xo
        ip[:, 1] = ip[:, 1] - self.zo
        ip *= self.scale

        ret = []
        for x, y in ip:
            ret.append((float(x), float(y)))

        return ret


def normalize(x):
    eps = 1e-12
    return x / (np.sqrt((x * x).sum(axis=1)) + eps).reshape(-1, 1)


class PseudoTracks(Dataset):
    def __init__(self, length=200, n=15, min_len=100) -> None:
        self.n = n
        self.seqlen = length
        ts = TrackSet()
        bound_w = 500
        bound_h = 1200
        avg_h = 162
        avg_w = 54
        sigma = 0.1
        self.min_len = min_len
        calibrator = Calibrator(x=bound_w / 2, y=-200, z=avg_h * 2)
        for i in range(n):
            L = random.randint(self.min_len, self.seqlen)
            start = random.randint(1, length - L + 1)
            end = random.randint(start + L - 1, length)
            sx, sy = random.randint(1, bound_w), random.randint(1, bound_h)
            tx, ty = random.randint(1, bound_w), random.randint(1, bound_h)
            d = ((sx - tx)**2 + (tx - ty)**2)**0.5
            mx, my = (sx + tx) / 2, (sy + ty) / 2
            mx += random.betavariate(2, 5) * d * randsign()
            my += random.betavariate(2, 5) * d * randsign()

            ps = BCurve([(sx, sy), (mx, my), (tx, ty)], end - start + 1)

            w = avg_w * (1 + random.gauss(0, sigma))
            h = avg_h * (1 + random.gauss(0, sigma))

            lt = []
            for x, y in ps:
                lt.append((x - w / 2, y, h))
            rt = []
            for x, y in ps:
                rt.append((x + w / 2, y, h))
            lb = []
            for x, y in ps:
                lb.append((x - w / 2, y, 0))
            rb = []
            for x, y in ps:
                rb.append((x + w / 2, y, 0))

            # print(lt[0], rt[0], lb[0], rb[0])

            lt = calibrator(lt)
            rt = calibrator(rt)
            lb = calibrator(lb)
            rb = calibrator(rb)

            # print(lt[0], rt[0], lb[0], rb[0])

            x1_1 = list(map(lambda x: x[0], lt))
            x1_2 = list(map(lambda x: x[0], lb))
            x1 = list(map(lambda a: sum(a) / 2, zip(x1_1, x1_2)))
            x2_1 = list(map(lambda x: x[0], rt))
            x2_2 = list(map(lambda x: x[0], rb))
            x2 = list(map(lambda a: sum(a) / 2, zip(x2_1, x2_2)))
            y1_1 = list(map(lambda x: x[1], lt))
            y1_2 = list(map(lambda x: x[1], rt))
            y1 = list(map(lambda a: sum(a) / 2, zip(y1_1, y1_2)))
            y2_1 = list(map(lambda x: x[1], lb))
            y2_2 = list(map(lambda x: x[1], rb))
            y2 = list(map(lambda a: sum(a) / 2, zip(y2_1, y2_2)))

            for fr, one in enumerate(zip(x2, y1, x1, y2)):
                xx, yy, u, v = one
                ww = u - xx
                hh = v - yy
                d = Det(xx, yy, ww, hh, uid=i + 1, fr=fr + start)
                if d.trim((calibrator.w, calibrator.h)).iou(d) > 0.2:
                    d.status = 1
                else:
                    d.status = 0
                d.conf = 1
                ts.append_data(d)
        self.data = ts
        self.gt = ts
        self._feat_dim = 508
        self._trans_mat = np.random.randn(self._feat_dim,
                                          self._feat_dim).astype(np.float32)
        with open('/home/toka/data/psd-1/gt.txt', 'w') as fd:
            self.data.dump(fd, formatter='fr.i,id.i,x1,y1,w,h,st.i,1,st.i')

    def __len__(self) -> int:
        return self.seqlen

    def __getitem__(self, index: int):
        index += 1
        ds = self.data[index]
        features_ = np.zeros((len(ds), self._feat_dim), dtype=np.float32)
        for i, d in enumerate(ds):
            features_[i][d.uid - 1] = 1
        features_ = np.matmul(features_, self._trans_mat)

        features = []
        for i, d in enumerate(ds):
            sc = []
            for j, o in enumerate(ds):
                sc.append(d.iou(o))
            sc = np.array(sc, dtype=np.float32).reshape(1, -1)
            feat = np.matmul(sc, features_) + \
                np.random.randn(1, self._feat_dim).astype(np.float32) * 0.1
            # feat = features_[i].reshape(1, -1)
            features.append(feat)

        dets = []
        feats = []
        ids = []
        for i, d in enumerate(ds):
            flag = False
            for j, dd in enumerate(ds):
                if dd.iou(d) > 0.6 and dd.y2 > d.y2 + 5:
                    flag = True
            if d.status == 0 or flag or d.conf < 0.3:
                continue
            # if False:
            #     continue
            else:
                x1, y1, w, h = d.toList()
                x2 = x1 + w
                y2 = y1 + h
                _std = 2e-2
                x1 += w * random.gauss(0, _std)
                y1 += h * random.gauss(0, _std)
                x2 += w * random.gauss(0, _std)
                y2 += h * random.gauss(0, _std)
                dets.append((x1, y1, x2, y2))
                feats.append(features[i])
                ids.append(d.uid)
        alldets = []
        allids = []
        for gd in self.gt[index]:
            if gd.status != 1:
                continue
            x1, y1, w, h = gd.toList()
            x2 = x1 + w
            y2 = y1 + h
            a_det = (x1, y1, x2, y2)
            a_id = gd.uid
            alldets.append(a_det)
            allids.append(a_id)
        tot = self._feat_dim + 4
        alldets = np.array(alldets,
                           np.float32) / 1000  # / 1000 * (math.sqrt(4 / tot))
        targets = {
            'gt_ids': np.array(ids, dtype=np.int),
            'all_dets': alldets if alldets.shape[0] > 0
            else alldets.reshape(0, 4),
            'all_ids': np.array(allids, dtype=np.int),
        }
        if len(dets) > 0:
            ret = {
                'n': len(dets),
                'dets': np.array(dets, dtype=np.float32) / 1000,
                'feats': np.concatenate(feats, axis=0),
                'targets': targets,
            }
            ret['new_slots'] = np.concatenate([
                normalize(ret['feats']) * (math.sqrt(self._feat_dim / tot)),
                ret['dets'] / 1000 * (math.sqrt(4 / tot)),
            ],
                axis=1)
        else:
            ret = {
                'n': 0,
                'dets': np.zeros((0, 4), dtype=np.float32),
                'feats': np.zeros((0, self._feat_dim), dtype=np.float32),
                'targets': targets,
            }
            ret['new_slots'] = np.zeros((0, self._feat_dim + 4),
                                        dtype=np.float32)
        # print(ret['new_slots'][0], ret['new_slots'][2])
        # print(ret['targets']['gt_ids'][0], ret['targets']['gt_ids'][2])
        # print(np.sum(ret['new_slots'][0] * ret['new_slots'][0]))
        return ret


class PseudoTracks2(PseudoTracks):
    def __init__(
        self,
        gtfile='/mnt/lustre/share/fengweitao/MOT16/gts/train/MOT16-02',
        **kwargs,
    ) -> None:
        ts = TrackSet(gtfile, filter=lambda x: x.label in [1, 2, 7, 8])
        self.n = ts.id_count()
        self.seqlen = ts.max_fr - ts.min_fr + 1
        self.data = ts
        self.gt = ts
        self._feat_dim = 508
        self._trans_mat = np.random.randn(self._feat_dim,
                                          self._feat_dim).astype(np.float32)


class PseudoTracks3(PseudoTracks):
    def __init__(
        self,
        dtfile='/mnt/lustre/share/fengweitao/MOT17/train/MOT17-02-SDP/det/det.txt',
        gtfile='/mnt/lustre/share/fengweitao/MOT16/gts/train/MOT16-02',
        **kwargs,
    ) -> None:
        ts_ = TrackSet(dtfile)
        gs = TrackSet(gtfile)
        ts = TrackSet()
        for fr in ts_.frameRange():
            for i, d in enumerate(ts_[fr]):
                mxiou = 0.
                mxid = -1
                for j, g in enumerate(gs[fr]):
                    if g.iou(d) > mxiou:
                        mxiou = g.iou(d)
                        mxid = g.uid
                if mxid < 0:
                    d.uid = gs.id_count() + i + 1
                else:
                    d.uid = mxid
                ts.append_data(d)
        self.n = ts.id_count()
        self.seqlen = ts.max_fr - ts.min_fr + 1
        self.data = ts
        self.gt = gs
        self._feat_dim = 508
        self._trans_mat = np.random.randn(self._feat_dim,
                                          self._feat_dim).astype(np.float32)


if __name__ == '__main__':
    ps = PseudoTracks(n=200)
    print(ps[0])
    dl = DataLoader(ps, batch_size=1, shuffle=False, num_workers=1)
    for d in dl:
        print(d)
        break


class RealTracks:

    __max_per_frame = 128

    def __init__(self, p4checkpoints, seqinfos, gts=None, clip_len=30, test=False) -> None:
        if test:
            clip_len = 1
        self._data_path = p4checkpoints
        self._data = []
        self._inds = []
        self._clip_len = clip_len
        self._seq_names = []
        self._gt_path = gts
        self._test_mode = test
        with open(seqinfos) as fd:
            self._infos = yaml.safe_load(fd)
        for n, ckpt in enumerate(self._data_path):
            seqname = os.path.splitext(os.path.basename(ckpt))[0]
            self._seq_names.append(seqname)
            with open(ckpt, 'rb') as fd:
                d = pickle.load(fd)
                self._data.append(d)
                assert self._infos[seqname]['L'] == len(d)
                for i in range(0, len(d) - clip_len + 1):
                    self._inds.append((n, i, clip_len))
        self._gts = None
        if gts is not None:
            assert len(gts) == len(self._data_path)
            self._gts = []
            for i, gtfile in enumerate(gts):
                all_gts = []
                gs = TrackSet(gtfile, filter=lambda x: x.status == 1)
                for fr in range(1, self._infos[self._seq_names[i]]['L'] + 1):
                    dets = []
                    ids = []
                    vis = []
                    for d in gs[fr]:
                        d: Det = d
                        dets.append((d.x1, d.y1, d.x2, d.y2))
                        ids.append(d.uid)
                        vis.append(d.conf)
                    all_gts.append((dets, ids, vis))
                assert len(all_gts) == len(self._data[i])
                self._gts.append(all_gts)

        print('Loaded %d ckpts, %d clips using clip_len=%d' %
              (len(self._data), len(self._inds), self._clip_len))

    def normalize_shape(self, data):
        shape = list(data.shape)
        assert shape[0] <= self.__max_per_frame
        shape[0] = self.__max_per_frame
        a = np.zeros(shape, dtype=data.dtype)
        a[:data.shape[0]] = data
        return a

    def normalize_inds(self, shape):
        a = np.zeros((self.__max_per_frame, ), dtype=np.uint8)
        a[:shape[0]] = 1
        return a

    def _preprocess(self, dets):
        return np.ones((dets.shape[0], ), dtype=np.bool8)

    def __len__(self):
        return len(self._inds)

    def __getitem__(self, index):
        ret = {}
        all_dets = []
        all_feats = []
        all_conf = []
        all_masks = []
        n, i, l = self._inds[index]
        W = self._infos[self._seq_names[n]]['W']
        H = self._infos[self._seq_names[n]]['H']
        ret['origin_size'] = (W, H)
        for one in range(i, i + l):
            dets, feats = self._data[n][one]
            inds = self._preprocess(dets)
            dets = dets[inds]
            feats = feats[inds]
            conf = dets[:, -1].copy()
            dets = dets[:, :4].copy()
            feats = feats.copy()
            dets[:, ::2] = dets[:, ::2] / W
            dets[:, 1::2] = dets[:, 1::2] / H
            if self._test_mode:
                all_dets.append(dets)
                all_feats.append(feats)
                all_conf.append(conf)
            else:
                all_dets.append(self.normalize_shape(dets))
                all_feats.append(self.normalize_shape(feats))
                all_conf.append(self.normalize_shape(conf))
                all_masks.append(self.normalize_inds(dets.shape))
        if self._test_mode:
            ret['dets'] = all_dets
            ret['feats'] = all_feats
            ret['confs'] = all_conf
        else:
            ret['dets'] = np.stack(all_dets)
            ret['feats'] = np.stack(all_feats)
            ret['confs'] = np.stack(all_conf)
            ret['inds'] = np.stack(all_masks)
        if self._gts is not None:
            gt_dets = []
            gt_ids = []
            gt_masks = []
            gt_vis = []
            for one in range(i, i + l):
                dets, ids, vis = self._gts[n][one]
                dets = np.array(dets, dtype=np.float32)
                ids = np.array(ids, dtype=np.int32)
                vis = np.array(vis, dtype=np.float32)
                dets[:, ::2] = dets[:, ::2] / W
                dets[:, 1::2] = dets[:, 1::2] / H
                if self._test_mode:
                    gt_dets.append(dets)
                    gt_ids.append(ids)
                    gt_vis.append(vis)
                else:
                    gt_dets.append(self.normalize_shape(dets))
                    gt_ids.append(self.normalize_shape(ids))
                    gt_vis.append(self.normalize_shape(vis))
                    gt_masks.append(self.normalize_inds(dets.shape))
            if self._test_mode:
                ret['gt_dets'] = gt_dets
                ret['gt_ids'] = gt_ids
                ret['gt_vis'] = gt_vis
            else:
                ret['gt_dets'] = np.stack(gt_dets)
                ret['gt_ids'] = np.stack(gt_ids)
                ret['gt_vis'] = np.stack(gt_vis)
                ret['gt_inds'] = np.stack(gt_masks)
        return ret


class SparseRTracks(RealTracks):

    conf_thr = 0.05

    def _preprocess(self, dets):
        inds = dets[:, 4] > self.conf_thr
        return inds

    # def __getitem__(self, index):
    #     ret = super().__getitem__(index)
    #     if self._test_mode:
    #         for i, one in enumerate(ret['confs']):
    #             valid_mask = one > self.conf_thr
    #             for k in ['confs', 'dets', 'feats']:
    #                 ret[k][i] = ret[k][i][valid_mask]
    #     else:
    #         valid_mask = ret['confs'] > self.conf_thr
    #         ret['inds'] = ret['inds'] * valid_mask
    #     return ret
