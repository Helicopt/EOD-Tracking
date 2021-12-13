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
import json

seed = 2077
random.seed(seed)
np.random.seed(seed)

__all__ = ['SubRealDetsDataset', 'RealDets'] + ['RealDets0%d0' % i for i in range(1, 7)]


class SubRealDetsDataset(Dataset):

    __max_per_frame = 512
    conf_thr = 0.55

    def __init__(self, data, infos, gt_data=None, clip_len=30, test=False):
        super().__init__()
        if test:
            clip_len = 1
        self._data = data
        self._infos = infos
        self._gts = gt_data
        self._inds = []
        self._clip_len = clip_len
        self._test_mode = test
        for n, rows in enumerate(self._data):
            for i in range(0, len(rows) - clip_len + 1):
                self._inds.append((n, i, clip_len))

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
        inds = dets[:, 4] > self.conf_thr
        return inds

    def __len__(self):
        return len(self._inds)

    def __getitem__(self, index):
        ret = {}
        all_dets = []
        all_feats = []
        all_conf = []
        all_masks = []
        n, i, l = self._inds[index]
        W = self._infos[n]['W']
        H = self._infos[n]['H']
        ret['origin_size'] = (W, H)
        for one in range(i, i + l):
            # dets, feats = self._data[n][one]
            dets = self._data[n][one]
            inds = self._preprocess(dets)
            dets = dets[inds]
            # feats = feats[inds]
            conf = dets[:, -1].copy()
            dets = dets[:, :4].copy()
            # feats = feats.copy()
            dets[:, ::2] = dets[:, ::2] / W
            dets[:, 1::2] = dets[:, 1::2] / H
            if self._test_mode:
                all_dets.append(dets)
                # all_feats.append(feats)
                all_conf.append(conf)
            else:
                all_dets.append(self.normalize_shape(dets))
                # all_feats.append(self.normalize_shape(feats))
                all_conf.append(self.normalize_shape(conf))
                all_masks.append(self.normalize_inds(dets.shape))
        if self._test_mode:
            ret['dets'] = all_dets
            # ret['feats'] = all_feats
            ret['confs'] = all_conf
        else:
            ret['dets'] = np.stack(all_dets)
            # ret['feats'] = np.stack(all_feats)
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


class RealDets(SubRealDetsDataset):

    def __init__(self, p4checkpoints, gts=None, clip_len=30, test=False) -> None:
        self._data_path = p4checkpoints
        self._data = []
        self._seq_names = {}
        self._seq_infos = {}
        self._seq_ptrs = {}
        self._gt_path = gts
        self._infos = []
        with open(p4checkpoints) as fd:
            rows = fd.readlines()
            for row in rows:
                row = json.loads(row)
                image_id = row['image_id']
                bbox = row['bbox']
                score = row['score']
                H = int(row['height'])
                W = int(row['width'])
                fr = int(os.path.splitext(os.path.basename(image_id))[0])
                seq_name = os.path.basename(os.path.dirname(os.path.dirname(image_id)))
                if seq_name not in self._seq_names:
                    self._seq_names[seq_name] = {}
                    self._seq_infos[seq_name] = [H, W]
                if fr not in self._seq_names[seq_name]:
                    self._seq_names[seq_name][fr] = []
                self._seq_names[seq_name][fr].append(bbox + [score])
        for n, seq_name in enumerate(sorted(self._seq_names.keys())):
            seq_data_frames = sorted(self._seq_names[seq_name].keys())
            rows = []
            for fr in seq_data_frames:
                row = self._seq_names[seq_name][fr]
                row = np.array(row)
                rows.append(row)
            self._data.append(rows)
            H, W = self._seq_infos[seq_name]
            self._infos.append({'H': H, 'W': W})
            self._seq_ptrs[seq_name] = n
        self._gts = None
        # if gts is not None:
        #     assert len(gts) == len(self._data_path)
        #     self._gts = []
        #     for i, gtfile in enumerate(gts):
        #         all_gts = []
        #         gs = TrackSet(gtfile, filter=lambda x: x.status == 1)
        #         for fr in range(1, self._infos[self._seq_names[i]]['L'] + 1):
        #             dets = []
        #             ids = []
        #             vis = []
        #             for d in gs[fr]:
        #                 d: Det = d
        #                 dets.append((d.x1, d.y1, d.x2, d.y2))
        #                 ids.append(d.uid)
        #                 vis.append(d.conf)
        #             all_gts.append((dets, ids, vis))
        #         assert len(all_gts) == len(self._data[i])
        #         self._gts.append(all_gts)

        super().__init__(self._data, self._infos, gt_data=self._gts, clip_len=clip_len, test=test)
        print('Loaded %d ckpts, %d clips using clip_len=%d' %
              (len(self._data), len(self._inds), self._clip_len))

    def get(self, seq_name):
        assert seq_name in self._seq_ptrs
        subdata = SubRealDetsDataset(
            [self._data[self._seq_ptrs[seq_name]]],
            [self._infos[self._seq_ptrs[seq_name]]],
            gt_data=None, clip_len=self._clip_len, test=self._test_mode)
        subdata.conf_thr = self.conf_thr
        return subdata

    @property
    def all_sequences(self):
        return list(self._seq_ptrs.keys())


class RealDets060(RealDets):

    conf_thr = 0.60


class RealDets050(RealDets):

    conf_thr = 0.50


class RealDets040(RealDets):

    conf_thr = 0.40


class RealDets030(RealDets):

    conf_thr = 0.30


class RealDets020(RealDets):

    conf_thr = 0.20


class RealDets010(RealDets):

    conf_thr = 0.20
