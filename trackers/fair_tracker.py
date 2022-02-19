import numpy as np
# from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
# import cv2
import torch.nn.functional as F

# from .fair_tracker_misc.utils import *
from ..utils.debug import logger_print, logger
from .no_tracking import NoTracking
from ..utils.matching import bbox_overlaps, bbox_dist
from .byte_tracker_misc.utils.kalman_filter import KalmanFilter
from .byte_tracker_misc.utils import matching
from .byte_tracker_misc.basetrack import BaseTrack, ByteTrackState

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

import time

__all__ = ['JDETracker']

global time_sync_rec
time_sync_rec = {}


def time_sync(tag, log=True):
    return
    global time_sync_rec
    torch.cuda.synchronize()
    stamp = time.time()
    if tag in time_sync_rec:
        count = stamp - time_sync_rec[tag]
        print('[%s] %.4f' % (tag, count))
    else:
        count = None
    time_sync_rec[tag] = stamp
    return count


def time_clean():
    global time_sync_rec
    time_sync_rec = {}


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, label, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        self.track_conf = 1.0
        self.label = label

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != ByteTrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != ByteTrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = ByteTrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = ByteTrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = ByteTrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def dt_bboxes(self):
        return np.concatenate((self.tlbr, np.array([self.score, self.label, self.track_conf, self.track_id])), axis=0)

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


@MODULE_ZOO_REGISTRY.register('fairtracker')
class JDETracker(NoTracking):
    def __init__(self, cfg, frame_rate=30):
        super().__init__()
        self.cfg = cfg

        self.det_thresh = cfg.get('conf_thres', 0.6)
        self.buffer_size = cfg.get('track_buffer', 30)
        self.max_time_lost = self.buffer_size
        # self.max_per_image = cfg.K
        # self.mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        # self.std = np.array(cfg.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()
        # self.inputs_embs = []

        self.device = None
        self.use_gt = False

    def initialize(self, state):
        super().initialize(state)

        state.tracked_stracks = []  # type: list[STrack]
        state.lost_stracks = []  # type: list[STrack]
        state.removed_stracks = []  # type: list[STrack]

        state.frame_id = 0

    def finalize(self, state):
        tag = 'data'
        # self.collector.dump()
        return NotImplemented

    def get_gt(self, image_id, vimage_id):
        from senseTk.common import TrackSet
        frame_id = int(os.path.basename(image_id).split('.')[0])
        seq_dir = os.path.dirname(os.path.dirname(image_id))
        gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
        # seq = os.path.basename(seq_dir)
        seq = os.path.basename(os.path.dirname(vimage_id))
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
        splits = seq.split('-')
        _, skip, = splits[:2]
        major, seqno, = splits[-2:]
        # print(skip, major, seqno)
        skip = int(skip)
        ret = {'thr_first': 0.8, 'thr_second': 0.9, 'skip': skip}
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

    def forward(self, state, inputs):
        state.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        output_results, real_output_results, embeds = self.preprocess(
            inputs['dt_bboxes'], inputs.get('id_embeds', None), inputs['image_info'])
        seq, gt = self.get_gt(inputs['image_id'], inputs['vimage_id'])
        seq_config = self.get_seq_config(seq)
        self.device = output_results.device
        gids = self.assign_gt(real_output_results, gt)
        total_scores = real_output_results[:, 4]
        total_bboxes = real_output_results[:, :5]
        total_cls = real_output_results[:, 5]

        remain_inds = total_scores > self.det_thresh
        dets = total_bboxes[remain_inds].cpu().numpy()
        cls_la = total_cls[remain_inds].cpu().numpy()
        id_feature = embeds[remain_inds].cpu().numpy()
        # self.inputs_embs.append((dets, id_feature))
        # print(id_feature.shape)
        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], la, f, 30) for
                          (tlbrs, la, f) in zip(dets[:, :5], cls_la, id_feature)]
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

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, state.lost_stracks)
        # Predict the current location with KF
        # for strack in strack_pool:
        # strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == ByteTrackState.Tracked:
                track.update(detections[idet], state.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, state.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == ByteTrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == ByteTrackState.Tracked:
                track.update(det, state.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, state.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == ByteTrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], state.frame_id)
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
            track.activate(self.kalman_filter, state.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in state.lost_stracks:
            if state.frame_id - track.end_frame > self.max_time_lost:
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
        state.tracked_stracks, state.lost_stracks = remove_duplicate_stracks(state.tracked_stracks, state.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in state.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(state.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        # transfer format ['image_info', 'dt_bboxes', 'image_id']
        output_targets = np.array([track.dt_bboxes for track in output_stracks])
        # output_dets = torch.cat([bboxes, ids], dim=1)
        if output_targets.size != 0:
            output_targets = output_targets[output_targets[:, -1] > 0]
            dt_boxes = torch.from_numpy(output_targets)
            raw_dt_boxes = self.postprocess(dt_boxes, inputs['image_info'])
        else:
            raw_dt_boxes = torch.from_numpy(output_targets)
        inputs['dt_bboxes'] = raw_dt_boxes
        # self.collector.print_info()
        # self.collector.collect(seq, state.frame_id, output_stracks)
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
