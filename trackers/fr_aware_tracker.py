import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from typing import List

from .byte_tracker_misc.utils.kalman_filter import KalmanFilter
from .byte_tracker_misc.utils import matching
from .byte_tracker_misc.basetrack import BaseTrack
from .no_tracking import NoTracking
from .byte_tracker_misc.basetrack import ByteTrackState
from .byte_tracker_misc.utils.tracklet import STrack
from .simple_tracker import joint_stracks, sub_stracks, remove_duplicate_stracks
from ..utils.matching import bbox_overlaps, bbox_dist
from ..utils.debug import logger_print, info_debug, env
import time
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from scipy.interpolate import interp1d
import xgboost as xgb

import cv2


def sim(af, bf):
    return float(torch.sum(af * bf))


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return [to_numpy(d) for d in data]
    elif isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    return data


class info_collector(object):
    def __init__(self, collect_type=None, data_dir='./data/new_assoc', seq_limit=100):
        self._confirmed_cnt = 0
        self._confirmed_correct = 0
        self._unconfirmed_cnt = 0
        self._miss_tot = 0
        self._miss_false = 0
        self._collections = {}
        self._print_enable = False
        self._dump_enable = True
        self._count_enable = False
        self._collect_type = collect_type
        self._data_dir = data_dir
        self._seq_limit = seq_limit
        self._cm_seq_count = {}

    def match(self, a, b):
        if not self._count_enable:
            return
        if a and b:
            self._confirmed_cnt += 1
            if a & b:
                self._confirmed_correct += 1
        else:
            self._unconfirmed_cnt += 1

    def miss(self, a, b):
        if not self._count_enable:
            return
        if a and b:
            self._miss_tot += 1
            if a & b:
                self._miss_false += 1

    def print_info(self):
        if not self._print_enable:
            return
        logger_print('confirmed: %.3f[%d/%d], unconfirmed %.3f[%d/%d]' %
                     (self._confirmed_correct
                      / max(self._confirmed_cnt, 1),
                      self._confirmed_correct,
                      self._confirmed_cnt,
                      self._unconfirmed_cnt / max(self._confirmed_cnt + self._unconfirmed_cnt, 1),
                      self._unconfirmed_cnt,
                      self._confirmed_cnt + self._unconfirmed_cnt))
        logger_print('missed: %.3f[%d/%d]' %
                     (self._miss_false
                      / max(self._miss_tot, 1),
                      self._miss_false,
                      self._miss_tot))

    @staticmethod
    def _collect(tracks: List[STrack], frame):
        ret = {}
        for track in tracks:
            assert track.frame_id == frame
            box = track.tlbr
            embed = track._embed
            det_score = track.score
            uid = int(track.track_id)
            gid = list(track.gid)
            ret[uid] = box, embed, det_score, gid
        return ret

    def collect(self, seq, frame, tracks):
        if not self._dump_enable or self._collect_type != 'feature':
            return True
        if seq not in self._collections:
            self._collections[seq] = {}
        data = self._collect(tracks, frame)
        for uid in data:
            if uid not in self._collections[seq]:
                self._collections[seq][uid] = {}
            self._collections[seq][uid][frame] = data[uid]
        return True

    @staticmethod
    def _collect_cm(mcm, acm, ecm, lvl, cfg, lids, rids, frame=-1):
        gmat = [
            [
                (int(bool(ls & rs)) if ls or rs else -1) for rs in rids
            ]
            for ls in lids
        ]
        gmat = np.array(gmat)
        if not gmat.shape[0]:
            return {}
        wrongs_m1 = (gmat > 0) & (mcm > 0.5)
        wrongs_a1 = (gmat > 0) & (acm < 0.8)
        wrongs_m2 = (gmat == 0) & (mcm < 0.5)
        wrongs_a2 = (gmat == 0) & (acm > 0.8)
        # print(mcm[wrongs_m1], gmat[wrongs_m1], 'mo')
        # print(acm[wrongs_a1], gmat[wrongs_a1], 'ap')
        # print(mcm[wrongs_m2], gmat[wrongs_m2], 'mo')
        # print(acm[wrongs_a2], gmat[wrongs_a2], 'ap')
        ret = {'acm': acm.copy(), 'mcm': mcm.copy(), 'ecm': ecm, 'lvl': lvl, 'skip': cfg['skip'], 'gmat': gmat}
        return ret

    def collect_cm(self, seq, frame, motion_cm, appearance_cm, edists, level, seq_config, lids=None, rids=None, extra=None):
        if not self._dump_enable or self._collect_type != 'cm':
            return True
        if seq.startswith('S-'):
            splits = seq.split('-')
            skip = splits[1]
            rseq = '-'.join(splits[3:])
            rseq = 'S-' + skip + '-' + rseq
        else:
            rseq = seq
        seq = seq + '_' + str(level)
        rseq = rseq + '_' + str(level)
        if seq not in self._collections:
            self._collections[seq] = {}
        if self._cm_seq_count.get(rseq, 0) < self._seq_limit:
            data = self._collect_cm(motion_cm, appearance_cm, edists, level, seq_config, lids, rids, frame=frame)
            if data:
                if extra:
                    data['extra'] = to_numpy(extra)
                self._collections[seq][frame] = data
                self._cm_seq_count[rseq] = self._cm_seq_count.get(rseq, 0) + 1
            return True
        else:
            return False

    def dump(self):
        if not self._dump_enable or self._collect_type is None:
            return
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        for seq in self._collections:
            path = f'{self._collect_type}.{seq}.pth'
            path_seq = os.path.join(self._data_dir, path)
            torch.save(self._collections[seq], path_seq)
        self._collections = {}


def get_single_nn_model(model_cfg):
    import importlib
    import sys
    model_path = model_cfg['model_def']
    mode_file = os.path.basename(os.path.splitext(model_path)[0])
    sys.path.insert(0, os.path.dirname(model_path))
    mod = importlib.import_module(f'{mode_file}')
    model = getattr(mod, model_cfg['model_name'])(**model_cfg['model_args'])
    state_dict = torch.load(model_cfg['model_ckpt'], map_location='cpu')
    model.load_state_dict(state_dict['model'])
    logger_print(f'loaded model from {model_cfg["model_ckpt"]}, epoch {state_dict["epoch"]}')
    logger_print(f'eval metrics: {state_dict["metrics"]}')
    return model


def show_distances(dmat, index, lboxes, rboxes, limg, rimg, mode='left', dst='./dist.jpg'):
    from matplotlib import pyplot as plt
    if mode == 'right':
        tmp = lboxes
        lboxes = rboxes
        rboxes = tmp
        tmp = limg
        limg = rimg
        rimg = tmp
        dmat = dmat.T
    wsep = 10
    wbox = 100
    hbox = 300
    hsep = 100
    margin = 10
    width = len(rboxes) * (wbox + wsep) + margin * 2
    height = hsep + hbox * 2 + margin * 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8) + 255
    x1, y1, x2, y2 = map(int, lboxes[index].tlbr)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, rimg.shape[1]), min(y2, rimg.shape[0])
    canvas[margin:margin + hbox, margin:margin + wbox] = cv2.resize(limg[y1:y2, x1:x2], (wbox, hbox))
    for i in range(len(rboxes)):
        x1, y1, x2, y2 = map(int, rboxes[i].tlbr)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, rimg.shape[1]), min(y2, rimg.shape[0])
        canvas[margin + hsep + hbox:margin + hsep + hbox * 2,
               margin + i * (wbox + wsep):margin + i * (wbox + wsep) + wbox] = cv2.resize(rimg[y1:y2, x1:x2], (wbox, hbox))
        cv2.putText(canvas, '{:.2f}'.format(dmat[index, i]), (margin + i * (wbox + wsep), margin + hsep + hbox - margin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    cv2.imwrite(dst, canvas)


@MODULE_ZOO_REGISTRY.register('fr_aware_tracker')
class FRATracker(NoTracking):
    def __init__(self, cfg, old_kalman=False, frame_rate=30, collect=None, data_dir='./data', assoc_with_extra=False,
                 xgb_model=None, xgb_gatings=[None, None], nn_model=None, nn_gatings=[None, None], use_gt=False, pretracking=True):
        super(FRATracker, self).__init__()
        # self.tracked_stracks = []  # type: list[STrack]
        # self.lost_stracks = []  # type: list[STrack]
        # self.removed_stracks = []  # type: list[STrack]

        # self.frame_id = 0

        self.cfg = cfg
        # self.det_thresh = args.track_thresh
        self.track_thresh = self.cfg.get('track_thresh', 0.6)
        self.track_buffer = self.cfg.get('track_buffer', 30)
        self.conf_thresh = self.cfg.get('conf_thresh', 0.1)
        self.match_thresh = self.cfg.get('match_thresh', 0.9)
        self.sim_thresh = self.cfg.get('sim_thresh', 0.8)

        self.det_thresh = self.track_thresh + 0.1
        self.base_frame_rate = frame_rate
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.collect_type = collect
        self.collector = info_collector(collect_type=collect, data_dir=data_dir)
        self.gating_0 = 1 - 0.5
        self.gating_1 = 1 - 0.7
        self.pretracking = pretracking
        self.assoc_with_extra = assoc_with_extra
        if nn_model is not None:
            self.nn_model = get_single_nn_model(nn_model)
            self.unsorted = nn_model.get('unsorted', False)
            self.gating_0 = nn_gatings[0]
            self.gating_1 = nn_gatings[1]
        elif xgb_model is not None:
            # xgb_model = 'eod/tasks/trk/tools/xgb_lfr_25.model'
            self.xgb_model = xgb.Booster({'nthread': 8})
            self.xgb_model.load_model(xgb_model)
            self.gating_0 = xgb_gatings[0]
            self.gating_1 = xgb_gatings[1]

        self.device = None
        self.use_gt = use_gt
        self.old_kalman = old_kalman

    def initialize(self, state):
        super().initialize(state)

        state.tracked_stracks = []  # type: list[STrack]
        state.lost_stracks = []  # type: list[STrack]
        state.removed_stracks = []  # type: list[STrack]
        state.done_flag = False
        state.frame_id = 0
        state.cache = {}
        state.prev_gids = None

    def finalize(self, state):
        tag = 'data'
        self.collector.dump()
        return NotImplemented

    def get_gt(self, image_id, vimage_id):
        from senseTk.common import TrackSet
        frame_id = int(os.path.basename(image_id).split('.')[0])
        seq_dir = os.path.dirname(os.path.dirname(image_id))
        gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
        # seq = os.path.basename(seq_dir)
        seq = os.path.basename(os.path.dirname(vimage_id))
        if not seq.startswith('S-') and not seq.startswith('D_'):
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

    def assign_gt(self, real_bboxes, gt):
        if gt is not None:
            gtbboxes = self.collect(gt, device=self.device)
            gt_ious = bbox_overlaps(real_bboxes[:, :4], gtbboxes[:, :4])
            # g, gi = gt_ious.max(dim=1)
            # gids = gtbboxes[:, 4][gi].int()
            # gids[g < 0.5] = -1
            # raw_old_gids = [int(t.gid) for t in state.tracklets]
            # o_gids = torch.from_numpy(np.array(raw_old_gids)).to(gids.device)
            # old_gids = set(raw_old_gids)
            # new_gids = {*[int(gid) for gid in gids]}
            gids = []
            for i in range(gt_ious.size(0)):
                row = gt_ious[i]
                inds = (row > 0.5).nonzero()
                gid_row = gtbboxes[:, 4][inds]
                gids.append(set(list(map(int, gid_row))))
            return gids
        return None

    def preprocess(self, bboxes, embeds, info):
        raw_bboxes = bboxes
        bboxes = bboxes.clone()
        if embeds is not None:
            # lvls = embeds[:, -1:]
            embeds = F.normalize(embeds, dim=1)
            # embeds = torch.cat([embeds, lvls], dim=1)
        else:
            embeds = bboxes.new_zeros((bboxes.shape[0], 2))
        scale_h, scale_w = info[2]
        pad_h, pad_w = info[6], info[7]
        bboxes[:, [0, 2]] -= pad_w
        bboxes[:, [0, 2]] /= scale_w
        bboxes[:, [1, 3]] -= pad_h
        bboxes[:, [1, 3]] /= scale_h
        return raw_bboxes, bboxes, embeds

    def get_seq_config(self, seq):
        if seq.startswith('S-'):
            splits = seq.split('-')
            _, skip, = splits[:2]
            major, seqno, = splits[-2:]
            # print(skip, major, seqno)
            skip = int(skip)
        else:
            skip = 16
            major = seq
            seqno = 'X'
        ret = {'thr_first': 0.8, 'thr_second': 0.9, 'skip': skip, 'seq': '-'.join([str(skip), major, seqno])}
        if skip <= 2:
            pass
        elif skip <= 8:
            ret.update({
                'thr_first': 0.7,
                'thr_seoncd': 0.8,
            })
        elif skip <= 25:
            ret.update({
                'thr_first': 0.6,
                'thr_seoncd': 0.7,
            })
        else:
            ret.update({
                'thr_first': 0.5,
                'thr_seoncd': 0.6,
            })
        return ret

    def predict(self, mcm, acm, ecm, cfg, lvl=0):
        gmat = mcm
        intvls = np.array([cfg['skip'], lvl]).reshape(
            1, 1, 2).repeat(gmat.shape[0], 0).repeat(gmat.shape[1], 1)
        X = np.concatenate([intvls.astype(np.float32), acm.astype(np.float32).reshape(*gmat.shape, 1),
                            mcm.astype(np.float32).reshape(*gmat.shape, 1), ecm.astype(np.float32).reshape(*gmat.shape, 1), ], axis=2)
        N, M, C = X.shape
        X = X.reshape(N * M, C)
        if X.nbytes == 0:
            Y = X.reshape(N, M)
        else:
            Y = self.xgb_model.predict(xgb.DMatrix(X))
            Y = 1 - Y.reshape(N, M)
        # print(Y)
        return Y

    def nn_predict(self, mcm, acm, ecm, cfg, lvl=0, extra=None):
        gmat = mcm
        if extra is not None:
            min_diff = extra['ctrl_emb'].cpu().numpy().astype(np.float32)
            if not self.unsorted:
                min_diff = np.sort(min_diff)
            min_diff = interp1d(np.arange(min_diff.shape[0]), min_diff)(
                np.arange(0, min_diff.shape[0] - 1, (min_diff.shape[0] - 1) / 300))[:300]
            min_diff = torch.from_numpy(min_diff).to(self.device).float()
        else:
            min_diff = torch.zeros((300, )).to(self.device).float()
        intvls = np.array([cfg['skip'], lvl]).reshape(
            1, 1, 2).repeat(gmat.shape[0], 0).repeat(gmat.shape[1], 1)
        X = np.concatenate([intvls.astype(np.float32), acm.astype(np.float32).reshape(*gmat.shape, 1),
                            mcm.astype(np.float32).reshape(*gmat.shape, 1), ecm.astype(np.float32).reshape(*gmat.shape, 1), ], axis=2)
        N, M, C = X.shape
        X = X.reshape(N * M, C)
        if X.nbytes == 0:
            Y = X.reshape(N, M)
        else:
            self.nn_model.eval()
            self.nn_model = self.nn_model.to(self.device)
            X = torch.from_numpy(X).to(self.device)
            Y = self.nn_model({'features': X, 'ctrl': min_diff.reshape(1, -1)})['pred']
            Y = 1 - Y.reshape(N, M)
            Y = Y.cpu().numpy()
        # print(Y)
        return Y

    def nn_predict2(self, cfg, extra=None, lvl=0):
        if extra is None:
            return np.ones((1, 0))
        acm, ecm, mcm = extra['original_cos_sim']
        gmat = mcm
        if extra is not None:
            min_diff = extra['ctrl_emb'].cpu().numpy().astype(np.float32)
            if not self.unsorted:
                min_diff = np.sort(min_diff)
            min_diff = interp1d(np.arange(min_diff.shape[0]), min_diff)(
                np.arange(0, min_diff.shape[0] - 1, (min_diff.shape[0] - 1) / 300))[:300]
            min_diff = torch.from_numpy(min_diff).to(self.device).float()
        else:
            min_diff = torch.zeros((300, )).to(self.device).float()
        intvls = torch.from_numpy(np.array([cfg['skip'], lvl]).reshape(
            1, 1, 2).repeat(gmat.shape[0], 0).repeat(gmat.shape[1], 1)).to(self.device)

        X = torch.cat([intvls.float(), acm.float().reshape(*gmat.shape, 1),
                       mcm.float().reshape(*gmat.shape, 1), ecm.float().reshape(*gmat.shape, 1), ], dim=2)
        N, M, C = X.shape
        X = X.reshape(N * M, C)
        if X.numel() == 0:
            Y = np.zeros((N, M))
        else:
            self.nn_model.eval()
            self.nn_model = self.nn_model.to(self.device)
            # X = torch.from_numpy(X).to(self.device)
            Y = self.nn_model({'features': X, 'ctrl': min_diff.reshape(1, -1)})['pred']
            Y = 1 - Y.reshape(N, M)
            Y = Y.cpu().numpy()
        # print(Y)
        return Y

    def forward(self, state, inputs):
        if state.done_flag:
            return inputs
        self.save_state_cache(state, inputs)
        state.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        # logger_print(inputs['framerate'], inputs['vimage_id'])
        # info_debug(inputs)
        output_results, real_output_results, embeds = self.preprocess(
            inputs['dt_bboxes'], inputs.get('id_embeds', None), inputs['image_info'])
        seq, gt = self.get_gt(inputs['image_id'], inputs['vimage_id'])
        seq_config = self.get_seq_config(seq)
        if 'skip' in seq_config:
            max_time_lost = int(self.base_frame_rate / seq_config['skip'])
        else:
            max_time_lost = self.max_time_lost
        self.device = output_results.device
        gids = self.assign_gt(real_output_results, gt)
        if 'extra' in inputs and inputs['extra'] is not None:
            inputs['extra']['gids'] = (state.prev_gids, gids)
        state.prev_gids = gids
        total_scores = real_output_results[:, 4]
        total_bboxes = real_output_results[:, :4]
        total_cls = real_output_results[:, 5]
        oringal_indices = torch.arange(total_scores.shape[0], device=self.device)

        remain_inds = total_scores > self.track_thresh
        remain_inds_i = remain_inds.nonzero()

        # still
        remain_inds = remain_inds.cpu().numpy()
        scores = total_scores.cpu().numpy()
        bboxes = total_bboxes.cpu().numpy()
        total_cls = total_cls.cpu().numpy()

        inds_low = scores > self.conf_thresh
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        inds_second_i, = inds_second.nonzero()
        dets_second = bboxes[inds_second]
        embeds_second = embeds[inds_second]
        dets = bboxes[remain_inds]
        embeds_first = embeds[remain_inds]
        clss = total_cls[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_second = total_cls[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, embed=emb) for
                          (tlbr, s, c, emb) in zip(dets, scores_keep, clss, embeds_first)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in state.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        assert len(unconfirmed) == 0
        ''' Step 2: First association, with high score detection boxes'''
        split_ind = len(tracked_stracks)
        strack_pool = joint_stracks(tracked_stracks, state.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        lost_ones = strack_pool[split_ind:]
        strack_pool = strack_pool[:split_ind]
        if 'skip' not in seq_config or seq_config['skip'] < 0.0 * self.base_frame_rate or self.old_kalman:
            force_flag = False
        else:
            force_flag = True
        dists = matching.iou_distance(strack_pool, detections)
        edists = matching.e_distance(strack_pool, detections)
        sims = matching.cos_sim(strack_pool, detections)
        motion_cm = dists
        appearance_cm = sims
        if gids is not None:
            lvl0_flag = self.collector.collect_cm(seq, state.frame_id, motion_cm, appearance_cm, edists, 0, seq_config, lids=[
                t.gid for t in strack_pool], rids=[gids[int(i)] for i in remain_inds_i], extra=inputs.get('extra', None))
        thr_first = seq_config['thr_first']
        trk_inds = [trk.trk_ind for trk in strack_pool]
        # has_assoc = inputs.get('affinities', None) is not None and not self.pretracking
        has_assoc = self.assoc_with_extra
        if has_assoc:
            # logger_print('affinities', inputs['affinities'].shape)
            # logger_print(remain_inds_i.squeeze(-1))
            # logger_print(trk_inds)
            dists2 = self.nn_predict2(seq_config, inputs['extra'] if 'extra' in inputs else None, lvl=0)
            inputs['affinities'] = dists2
            if dists2.nbytes == 0:
                if dists.shape[1] == 0:
                    dists2 = dists2.reshape(0, 0)
                else:
                    dists2 = dists2.reshape(dists.shape[1], -1)
            dists = inputs['affinities'][remain_inds_i.squeeze(-1).cpu().numpy()][:, trk_inds].T
            # logger_print(dists.min(axis=1))
            self.gating_0 = 0.9
        elif hasattr(self, 'nn_model'):
            dists = self.nn_predict(motion_cm, appearance_cm, edists, seq_config, lvl=0,
                                    extra=inputs['extra'] if 'extra' in inputs else None)
        elif hasattr(self, 'xgb_model'):
            dists = self.predict(motion_cm, appearance_cm, edists, seq_config, lvl=0)
        else:
            dists = dists * thr_first + (1 - sims) * 2 * (1 - thr_first)
        # logger_print(dists.shape, dists)
        # if not self.cfg.get('mot20', False):
        #     dists = matching.fuse_score(dists, detections)
        # print(dists)
        # print(sims)
        # if state.frame_id > 3:
        #     exit(0)
        # rimg = cv2.imread(inputs['image_id'])
        # if hasattr(state, 'limg'):
        #     show_distances(dists, 0, strack_pool, detections, state.limg, rimg)
        # state.limg = rimg
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=1 - self.gating_0)
        if gids is not None:
            self.analyse(dists, strack_pool, detections, gids, sims, edists, matches, state.frame_id)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if gids is not None:
                # logger_print(track.gid, gids[int(remain_inds_i[idet])])
                self.collector.match(track.gid, gids[int(remain_inds_i[idet])])
                if not (track.gid & gids[int(remain_inds_i[idet])]) and track.gid:
                    print('gid not match', track.gid, gids[int(remain_inds_i[idet])],
                          dists[itracked, idet], sims[itracked, idet], edists[itracked, idet], seq, state.frame_id, seq_config.get('skip', -1))
                track.gid = gids[int(remain_inds_i[idet])]
            # if track.track_id == 39:
            #     s = sim(track._embed[:-1], embeds_first[idet][:-1])
            #     # logger_print(s, idet, state.frame_id, rk=3)
            #     if s < 0.6:
            #         logger_print(state.frame_id, track.tlbr, det.tlbr, rk=3)
            #         logger_print(track._embed[-1], embeds_first[idet][-1], rk=3)
            #         logger_print(track._embed, embeds_first[idet], rk=3)
            track._embed = embeds_first[idet]
            if track.state == ByteTrackState.Tracked:
                track.update(detections[idet], state.frame_id, force=force_flag)
                track.trk_ind = int(remain_inds_i[idet])
                activated_starcks.append(track)
            else:
                track.re_activate(det, state.frame_id, new_id=False)
                track.trk_ind = int(remain_inds_i[idet])
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, embed=emb) for
                                 (tlbr, s, c, emb) in zip(dets_second, scores_second, cls_second, embeds_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == ByteTrackState.Tracked]
        trk_inds = [trk.trk_ind for trk in r_tracked_stracks]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        edists = matching.e_distance(r_tracked_stracks, detections_second)
        sims = matching.cos_sim(r_tracked_stracks, detections_second)
        motion_cm = dists
        appearance_cm = sims
        if gids is not None:
            lvl1_flag = self.collector.collect_cm(seq, state.frame_id, motion_cm, appearance_cm, edists, 1, seq_config, lids=[
                t.gid for t in r_tracked_stracks], rids=[gids[int(i)] for i in inds_second_i], extra=inputs.get('extra', None))
            if not lvl0_flag and not lvl1_flag:
                state.done_flag = True
        thr_second = seq_config['thr_second']
        if has_assoc:
            # logger_print('affinities', inputs['affinities'].shape)
            # logger_print(remain_inds_i)
            # logger_print(trk_inds)
            dists = inputs['affinities'][inds_second_i][:, trk_inds].T
            self.gating_1 = 0.9
        elif hasattr(self, 'nn_model'):
            dists = self.nn_predict(motion_cm, appearance_cm, edists, seq_config, lvl=1,
                                    extra=inputs['extra'] if 'extra' in inputs else None)
        elif hasattr(self, 'xgb_model'):
            dists = self.predict(motion_cm, appearance_cm, edists, seq_config, lvl=1)
        else:
            dists = dists * thr_second + (1 - sims) * 2 * (1 - thr_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=1 - self.gating_1)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if gids is not None:
                # logger_print(track.gid, gids[int(inds_second_i[idet])])
                self.collector.match(track.gid, gids[int(inds_second_i[idet])])
                track.gid = gids[int(inds_second_i[idet])]
            track._embed = embeds_second[idet]
            if track.state == ByteTrackState.Tracked:
                track.update(det, state.frame_id, force=force_flag)
                track.trk_ind = inds_second_i[idet]
                activated_starcks.append(track)
            else:
                track.re_activate(det, state.frame_id, new_id=False)
                track.trk_ind = inds_second_i[idet]
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if gids is not None:
                for u_det in u_detection:
                    self.collector.miss(track.gid, gids[int(remain_inds_i[u_det])])
            if not track.state == ByteTrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        assert not unconfirmed
        if gids is not None:
            gids = [gids[int(remain_inds_i[i])] for i in u_detection]
        rrinds = [int(remain_inds_i[i]) for i in u_detection]
        embeds_first = [embeds_first[i] for i in u_detection]
        motion_cm = matching.iou_distance(lost_ones, detections)
        edists = matching.e_distance(lost_ones, detections)
        sims = matching.cos_sim(lost_ones, detections)
        if hasattr(self, 'xgb_model'):
            dists = self.predict(motion_cm, sims, edists, seq_config, lvl=0)
        else:
            dists = (1 - sims + edists) / 2.
        # logger_print(dists)
        # if not self.cfg.get('mot20', False):
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.2)
        for itracked, idet in matches:
            lost_ones[itracked].re_activate(detections[idet], state.frame_id)
            lost_ones[itracked].trk_ind = rrinds[idet]
            lost_ones[itracked]._embed = embeds_first[idet]
            refind_stracks.append(lost_ones[itracked])
            if gids is not None:
                track = lost_ones[itracked]
                if not (track.gid & gids[int(idet)]) and track.gid:
                    print('lost gid not match', track.gid, gids[int(idet)],
                          dists[itracked, idet], sims[itracked, idet], edists[itracked, idet], seq, state.frame_id, seq_config.get('skip', -1))
            if gids is not None:
                lost_ones[itracked].gid = gids[int(idet)]
        u_unconfirmed = []
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, state.frame_id)
            track.trk_ind = rrinds[inew]
            # if track.track_id == 39:
            #     logger_print(inew, track.tlbr, rk=3)
            if gids is not None:
                track.gid = gids[int(inew)]
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in state.lost_stracks:
            if state.frame_id - track.end_frame > max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        state.tracked_stracks = [t for t in state.tracked_stracks if t.state == ByteTrackState.Tracked]
        state.tracked_stracks = joint_stracks(state.tracked_stracks, activated_starcks)
        state.tracked_stracks = joint_stracks(state.tracked_stracks, refind_stracks)
        state.lost_stracks = sub_stracks(state.lost_stracks, state.tracked_stracks)
        state.lost_stracks.extend(lost_stracks)
        state.lost_stracks = sub_stracks(state.lost_stracks, state.removed_stracks)
        state.removed_stracks.extend(removed_stracks)
        # state.tracked_stracks, state.lost_stracks = remove_duplicate_stracks(state.tracked_stracks, state.lost_stracks)
        # get scores of lost tracks
        # output_stracks = [track for track in state.tracked_stracks if track.is_activated]
        output_stracks = [track for track in state.tracked_stracks if track.state == ByteTrackState.Tracked]

        # transfer format ['image_info', 'dt_bboxes', 'image_id']
        output_targets = np.array([track.dt_bboxes for track in output_stracks])
        # output_dets = torch.cat([bboxes, ids], dim=1)
        if output_targets.size != 0:
            output_targets = output_targets[output_targets[:, -1] > 0]
            dt_boxes = torch.from_numpy(output_targets)
            if False:
                self.visualize_result(dt_boxes, seq, state.frame_id, inputs['image_id'])
            raw_dt_boxes = self.postprocess(dt_boxes, inputs['image_info'])
        else:
            raw_dt_boxes = torch.from_numpy(output_targets)
        inputs['dt_bboxes'] = raw_dt_boxes
        # logger_print(raw_dt_boxes.shape)
        # self.collector.print_info()
        self.collector.collect(seq, state.frame_id, output_stracks)
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

    def save_state_cache(self, state, data):
        state.cache['dt_bboxes'] = data['dt_bboxes']
        state.cache['id_embeds'] = data['id_embeds']
        state.cache['original'] = data['original']

    def load_state_cache(self, state):
        if not hasattr(state, 'cache'):
            return {
                'dt_bboxes': None,
                'id_embeds': None,
                'original': None,
            }
        return {
            'dt_bboxes': state.cache.get('dt_bboxes', None),
            'id_embeds': state.cache.get('id_embeds', None),
            'original': state.cache.get('original', None),
        }

    def grab_cache(self):
        if not self.states:
            return {}
        data = {}
        for i in sorted(self.states.keys()):
            state = self.states[i]
            cache = self.load_state_cache(state)
            for k in cache:
                if k not in data:
                    data[k] = []
                data[k].append(cache[k])
        original = data.pop('original')
        ret = {
            'data': [data],
            'original': original,
        }
        return ret

    def visualize_result(self, boxes, seq, frame_id, image_id):
        import cv2
        import random
        self.colors = [80, 160, 255]
        img = cv2.imread(image_id)
        output_dir = './imgcache2'
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, (box[0], box[1], box[2], box[3]))
            color = random.sample(self.colors, 3)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img, str(int(box[7])), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        cv2.imwrite(os.path.join(output_dir, '{}_{}.jpg'.format(seq, frame_id)), img)

    def analyse(self, dists, strack_pool, detections, gids, sims, edists, matches, frame=None):
        match_rec = [-1] * len(strack_pool)
        occupation = [-1] * len(detections)
        for i, j in matches:
            match_rec[i] = j
            occupation[j] = i
        for i in range(len(dists)):
            correct_ind = []
            for j in range(len(detections)):
                if gids[j] & strack_pool[i].gid:
                    correct_ind.append(j)
            if match_rec[i] >= 0 and match_rec[i] in correct_ind or match_rec[i] == -1 and len(correct_ind) <= 0:
                continue
            elif strack_pool[i].gid:
                print('frame index:', frame)
                if match_rec[i] == -1:
                    print('very weird: %d(%s) not matched' % (i, strack_pool[i].gid))
                else:
                    print('very weird: %d(%s) %d(%s)' % (i, strack_pool[i].gid, match_rec[i], gids[match_rec[i]]))
                    print('dists:', dists[i][match_rec[i]], sims[i][match_rec[i]], edists[i][match_rec[i]])
                for j in correct_ind:
                    print(j, gids[j], 'correct dists:', dists[i][j], sims[i][j],
                          edists[i][j], occupation[j], strack_pool[occupation[j]].gid if occupation[j] >= 0 else 'None')