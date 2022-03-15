from __future__ import division

# Standard Library
import json
import math
import os
import re

# Import from third library
import cv2
import numpy as np
import torch
import copy
from easydict import EasyDict
import pickle as pk

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import DATASET_REGISTRY
from eod.tasks.det.data.datasets.custom_dataset import CustomDataset
from eod.data.datasets.transforms import build_transformer

from eod.data.data_utils import get_image_size

from torch.nn.modules.utils import _pair

from ..utils.read_helper import read_lines
from ..utils.debug import info_debug

__all__ = ['MultiFrameDataset']


@DATASET_REGISTRY.register('mot')
class MultiFrameDataset(CustomDataset):

    suffix = 'mf_re2'

    @property
    def next_id(self):
        self.id_cnt += 1
        return self.id_cnt

    def __init__(self,
                 meta_file,
                 image_reader,
                 transformer,
                 num_classes,
                 noaug_ratio=0,
                 transformer_noaug=None,
                 num_ids=None,
                 num_expected=6,
                 frame_involved=12,
                 add_self=False,
                 random_select=True,
                 repeat_to=None,
                 orientation=0,
                 orient_div=8,
                 orient_thr=0.2,
                 test_mode=False,
                 online=False,
                 evaluator=None,
                 label_mapping=None,
                 cache=None,
                 extra_cache=None,
                 clip_box=True,
                 ignore_vis_under=0.0,
                 filter=None,
                 multiframerates=None,
                 ):
        self.id_cnt = 0
        self.num_ids = num_ids
        self.test_mode = test_mode
        self.is_train = not test_mode
        self.num_expected = num_expected
        self.frame_involved = frame_involved
        self.add_self = add_self
        self.random_select = random_select
        self.repeat_num = repeat_to
        self.use_orientation = orientation
        self.orient_div = orient_div
        self.orient_thr = orient_thr
        self.online = online
        self.ignore_vis_under = ignore_vis_under
        self.multiframerates = multiframerates
        self.filter = filter
        super(MultiFrameDataset, self).__init__(
            meta_file, image_reader, transformer, num_classes,
            evaluator=evaluator, label_mapping=label_mapping, cache=cache, clip_box=clip_box)
        if extra_cache is not None:
            for ecache in extra_cache:
                with open(ecache, 'rb') as fd:
                    ecache_data = pk.load(fd)
                    self.cache_image.update(ecache_data)
        if transformer_noaug is not None:
            for trans in transformer_noaug:
                if 'kwargs' in trans and trans['kwargs'].get('extra_input', False):
                    trans['kwargs']['dataset'] = self
            self.transformer_noaug = build_transformer(transformer_noaug)
        else:
            self.transformer_noaug = lambda x: x
        self.noaug_ratio = noaug_ratio

    @staticmethod
    def parse_seq_info(filename, formatter):
        if formatter == '{root}/{seq}/img1/{fr}.{ext}':
            frame_id = int(os.path.splitext(os.path.basename(filename))[0])
            seq_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))
        elif formatter == '{root}/{seq}/{fr}.{ext}':
            frame_id = int(os.path.splitext(os.path.basename(filename))[0])
            seq_name = os.path.basename(os.path.dirname(filename))
        else:
            seq_name = os.path.splitext(os.path.basename(filename))[0]
            frame_id = 0
        return seq_name, frame_id

    def get_orient(self, ins_a, ins_b):
        ax1, ay1, ax2, ay2 = ins_a['bbox']
        bx1, by1, bx2, by2 = ins_b['bbox']
        dx = ((bx1 + bx2) - (ax1 + ax2)) / 2.
        dy = ((by1 + by2) - (ay1 + ay2)) / 2.
        scale = ((ax2 - ax1) * (ay2 - ay1)) ** 0.5
        dx /= scale
        dy /= scale
        s = (dx**2 + dy**2)**0.5
        if s > self.orient_thr:
            ang = math.acos(dx / s)
            if dy < 0:
                ang = math.pi * 2 - ang
            sector = math.pi * 2 / self.orient_div
            ocls = round(ang / sector)
            oreg = (ang - ocls * sector) / sector
            ocls = ocls % self.orient_div
            return ocls, oreg
        else:
            return -1, 0

    def _normal_init(self):
        self.sequences = {}
        self.id_mapping = {}
        self.seq_metas = []
        if not isinstance(self.meta_file, list):
            self.meta_file = [self.meta_file]
        for idx, meta_file in enumerate(self.meta_file):
            for data in read_lines(meta_file):
                filename = data['filename']
                seq_name, frame_id = self.parse_seq_info(filename, data['formatter'])
                if 'virtual_filename' not in data:
                    vfilename = data['filename']
                    vseq_name, vframe_id = seq_name, frame_id
                else:
                    vfilename = data['virtual_filename']
                    vseq_name, vframe_id = self.parse_seq_info(vfilename, '{root}/{seq}/{fr}.{ext}')
                if self.filter is not None and not re.match(self.filter, vseq_name):
                    continue
                if vseq_name not in self.sequences:
                    self.sequences[vseq_name] = {}
                if seq_name not in self.id_mapping:
                    self.id_mapping[seq_name] = {}
                for instance in data.get('instances', []):
                    track_id = instance.get('track_id', -1)
                    vis = instance.get('vis_rate', 1.0)
                    if vis < self.ignore_vis_under:
                        track_id = -1
                    if track_id >= 0 and track_id not in self.id_mapping[seq_name]:
                        self.id_mapping[seq_name][track_id] = self.next_id
                    instance['track_id'] = self.id_mapping[seq_name].get(track_id, 0)
                self.sequences[vseq_name][vframe_id] = len(self.metas)
                if self.label_mapping is not None:
                    data = self.set_label_mapping(data, self.label_mapping[idx], 0)
                    data['image_source'] = idx
                self.metas.append(data)
                self.seq_metas.append((vseq_name, vframe_id))
                if 'image_height' not in data or 'image_width' not in data:
                    logger.warning('image size is not provided, '
                                   'set aspect grouping to 1.')
                    self.aspect_ratios.append(1.0)
                else:
                    self.aspect_ratios.append(data['image_height'] / data['image_width'])
        self.seq_controls = {}
        for seq in self.sequences:
            frs = list(self.sequences[seq].keys())
            be = min(frs)
            en = max(frs)
            self.seq_controls[seq] = {
                'begin': be,
                'end': en,
            }
            if self.use_orientation:
                for fr in range(be, en + 1):
                    frame_data = self.metas[self.sequences[seq][fr]]
                    for instance in frame_data['instances']:
                        instance['orient_cls'] = -1
                        instance['orient_reg'] = 0.
                    if fr + self.use_orientation <= en:
                        nxt_frame_data = self.metas[self.sequences[seq][fr + self.use_orientation]]
                        def sort_key_cur(x): return frame_data['instances'][x]['track_id']
                        def sort_key_nxt(x): return nxt_frame_data['instances'][x]['track_id']
                        n = len(frame_data['instances'])
                        m = len(nxt_frame_data['instances'])
                        cur_ids = sorted(list(range(n)), key=sort_key_cur)
                        nxt_ids = sorted(list(range(m)), key=sort_key_nxt)
                        i = 0
                        j = 0
                        while i < n and j < m:
                            ins_i = frame_data['instances'][cur_ids[i]]
                            ii = ins_i['track_id']
                            ins_j = nxt_frame_data['instances'][nxt_ids[j]]
                            jj = ins_j['track_id']
                            if ii <= 0 or ii < jj:
                                i += 1
                            elif jj <= 0 or jj < ii:
                                j += 1
                            else:
                                ocls, oreg = self.get_orient(ins_i, ins_j)
                                instance['orient_cls'] = ocls
                                instance['orient_reg'] = oreg
                                i += 1
                                j += 1
        if not self.test_mode:
            assert self.num_ids is None or self.id_cnt < self.num_ids, 'num_ids in config is less than actual id_cnt loaded'
        logger.info('---- %d items in total, %d IDs' % (len(self.metas), self.id_cnt))

    def get_input(self, idx):
        """parse annotation into input dict
        """
        data = self.metas[idx]
        return self._get_input(data, idx)

    def get_input_by_seq_frame(self, seq_name, frame, idx):
        """parse annotation into input dict
        """
        data = self.metas[self.sequences[seq_name][frame]]
        return self._get_input(data, idx)

    def _get_input(self, data, idx):
        data = copy.deepcopy(data)
        img_id = filename = data['filename']
        vimg_id = data.get('virtual_filename', data['filename'])
        gt_bboxes = []
        ig_bboxes = []
        for instance in data.get('instances', []):
            if instance['is_ignored']:
                ig_bboxes.append(instance['bbox'])
            else:
                data_ins = instance['bbox'] + [instance['label'], instance['track_id']]
                if self.use_orientation:
                    data_ins.append(instance['orient_cls'])
                    data_ins.append(instance['orient_reg'])
                gt_bboxes.append(data_ins)

        if len(ig_bboxes) == 0:
            ig_bboxes = self._fake_zero_data(1, 4)
        if len(gt_bboxes) == 0:
            gt_bboxes = self._fake_zero_data(1, 6)

        gt_bboxes = torch.as_tensor(gt_bboxes, dtype=torch.float32)
        ig_bboxes = torch.as_tensor(ig_bboxes, dtype=torch.float32)
        cache = False
        try:
            if self.cache is not None:
                img = self.get_cache_image(data)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if self.image_reader.color_mode != 'BGR':
                    cvt_color = getattr(cv2, 'COLOR_BGR2{}'.format(self.image_reader.color_mode))
                    img = cv2.cvtColor(img, cvt_color)
                cache = True
            else:
                img = self.image_reader(filename, data.get('image_source', 0))
        except:  # noqa
            img = self.image_reader(filename, data.get('image_source', 0))
        input = EasyDict({
            'image': img,
            'gt_bboxes': gt_bboxes,
            'gt_ignores': ig_bboxes,
            'flipped': False,
            'filename': filename,
            'image_id': img_id,
            'vimage_id': vimg_id,
            'dataset_idx': idx,
            'neg_target': data.get('neg_target', 0),
            'cache': cache,
        })
        return input

    def prepare_input(self, input, noaug_flag=False):
        image_h, image_w = get_image_size(input.image)

        if noaug_flag:
            transform = self.transformer_noaug
        else:
            transform = self.transformer

        input = transform(input)
        scale_factor = input.get('scale_factor', 1)

        new_image_h, new_image_w = get_image_size(input.image)
        pad_w, pad_h = input.get('dw', 0), input.get('dh', 0)
        input.image_info = [new_image_h, new_image_w, scale_factor,
                            image_h, image_w, input.flipped, pad_w, pad_h]
        return input

    @property
    def ref_num(self):
        return self.num_expected + (1 if self.add_self else 0)

    def __getitem__(self, idx):
        """
        Get a single image data: from dataset

        Arguments:
            - idx (:obj:`int`): index of image, 0 <= idx < len(self)

        Returns:
            - input (:obj:`dict`)

        Output example::

            {
                # (FloatTensor): [1, 3, h, w], RGB format
                'image': ..,
                # (list): [resized_h, resized_w, scale_factor, origin_h, origin_w]
                'image_info': ..,
                # (FloatTensor): [N, 5] (x1, y1, x2, y2, label)
                'gt_bboxes': ..,
                # (FloatTensor): [N, 4] (x1, y1, x2, y2)
                'ig_bboxes': ..,
                # (str): image name
                'filename': ..
            }
        """
        seq_name, frame_id = self.seq_metas[idx]
        if self.multiframerates is None:
            options = [fr for fr in self.sequences[seq_name] if abs(
                fr - frame_id) <= self.frame_involved and fr != frame_id]
            sampled_framerate = 1
        else:
            sampled_framerate = int(np.random.choice(self.multiframerates, 1))
            options = [fr for fr in self.sequences[seq_name]
                       if abs(fr - frame_id) <= self.frame_involved * sampled_framerate
                       and fr != frame_id and abs(fr - frame_id) % sampled_framerate == 0]
        if not self.random_select:
            old_state = np.random.get_state()
            new_state = np.random.MT19937(0).state
            np.random.set_state(new_state)
        chosen = np.random.choice(options, min(len(options), self.num_expected), replace=False)
        if len(chosen) < self.num_expected:
            options.append(frame_id)
            chosen2 = np.random.choice(options, self.num_expected - len(chosen))
            chosen = np.concatenate([chosen, chosen2])
        chosen = list(map(int, chosen))
        if re.match(r'^S-[\d]+-', seq_name):
            sampled_framerate *= int(seq_name.split('-')[1])

        # print(seq_name, frame_id, chosen)

        if len(options) < 2:
            noaug_flag = False
        else:
            if self.noaug_ratio < 1e-12:
                noaug_flag = False
            elif self.noaug_ratio > 1 - 1e-12:
                noaug_flag = True
            else:
                noaug_flag = np.random.rand() < self.noaug_ratio

        input_main = self.get_input_by_seq_frame(seq_name, frame_id, idx)
        input_main = self.prepare_input(input_main, noaug_flag=noaug_flag)
        input_ref = [self.get_input_by_seq_frame(seq_name, fr, idx) for fr in chosen]
        input_ref = [self.prepare_input(o, noaug_flag=noaug_flag) for o in input_ref]
        if self.add_self:
            input_ref.append(input_main)
        if not self.random_select:
            np.random.set_state(old_state)
        return {
            'main': input_main, 'ref': input_ref,
            'begin_flag': frame_id == self.seq_controls[seq_name]['begin'],
            'end_flag': frame_id == self.seq_controls[seq_name]['end'],
            'noaug_flag': noaug_flag,
            'framerate': sampled_framerate,
        }

    @property
    def all_sequences(self):
        seqs = sorted(self.sequences.keys())
        ret = {}
        for seq in seqs:
            frs = sorted(self.sequences[seq].keys())
            ret[seq] = [self.sequences[seq][fr] for fr in frs]
        return ret

    def dump(self, output):
        out_res = []
        image_info = output['image_info']
        if not output['dt_bboxes'].numel():  # not even a box
            return out_res
        bboxes = self.tensor2numpy(output['dt_bboxes'])
        image_ids = output['image_id']
        vimage_ids = output['vimage_id']
        for b_ix in range(len(image_info)):
            info = image_info[b_ix]
            height, width = map(int, info[3:5])
            img_id = image_ids[b_ix]
            vimg_id = vimage_ids[b_ix]
            scores = bboxes[:, 5]
            keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
            keep_ix = sorted(keep_ix, key=lambda ix: scores[ix], reverse=True)
            scale_h, scale_w = _pair(info[2])
            img_bboxes = bboxes[keep_ix].copy()
            # sub pad
            pad_w, pad_h = info[6], info[7]
            img_bboxes[:, [1, 3]] -= pad_w
            img_bboxes[:, [2, 4]] -= pad_h
            # clip
            if self.clip_box:
                np.clip(img_bboxes[:, [1, 3]], 0, info[1], out=img_bboxes[:, [1, 3]])
                np.clip(img_bboxes[:, [2, 4]], 0, info[0], out=img_bboxes[:, [2, 4]])
            img_bboxes[:, 1] /= scale_w
            img_bboxes[:, 2] /= scale_h
            img_bboxes[:, 3] /= scale_w
            img_bboxes[:, 4] /= scale_h

            for i in range(len(img_bboxes)):
                box_score, cls = img_bboxes[i][5:7]
                bbox = img_bboxes[i].copy()
                score = float(box_score)
                res = {
                    'height': height,
                    'width': width,
                    'image_id': img_id,
                    'vimage_id': vimg_id,
                    'bbox': bbox[1: 1 + 4].tolist(),
                    'score': score,
                    'label': int(cls)
                }
                if img_bboxes.shape[1] >= 8:
                    res['track_id'] = int(img_bboxes[i][8])
                out_res.append(res)
        return out_res
