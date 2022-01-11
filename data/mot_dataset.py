from __future__ import division

# Standard Library
import json
import math
import os

# Import from third library
import cv2
import numpy as np
import torch
import copy
from easydict import EasyDict

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
                 test_mode=False,
                 online=False,
                 evaluator=None,
                 label_mapping=None,
                 cache=None,
                 clip_box=True,
                 ignore_vis_under=0.0,
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
        self.online = online
        self.ignore_vis_under = ignore_vis_under
        super(MultiFrameDataset, self).__init__(
            meta_file, image_reader, transformer, num_classes,
            evaluator=evaluator, label_mapping=label_mapping, cache=cache, clip_box=clip_box)
        if transformer_noaug is not None:
            for trans in transformer_noaug:
                if 'kwargs' in trans and trans['kwargs'].get('extra_input', False):
                    trans['kwargs']['dataset'] = self
            self.transformer_noaug = build_transformer(transformer_noaug)
        else:
            self.transformer_noaug = lambda x: x
        self.noaug_ratio = noaug_ratio

    def parse_seq_info(self, filename, formatter):
        if formatter == '{root}/{seq}/img1/{fr}.{ext}':
            frame_id = int(os.path.splitext(os.path.basename(filename))[0])
            seq_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))
        else:
            seq_name = os.path.splitext(os.path.basename(filename))[0]
            frame_id = 0
        return seq_name, frame_id

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
                if seq_name not in self.sequences:
                    self.sequences[seq_name] = {}
                    self.id_mapping[seq_name] = {}
                for instance in data.get('instances', []):
                    track_id = instance.get('track_id', -1)
                    vis = instance.get('vis_rate', 1.0)
                    if vis < self.ignore_vis_under:
                        track_id = -1
                    if track_id >= 0 and track_id not in self.id_mapping[seq_name]:
                        self.id_mapping[seq_name][track_id] = self.next_id
                    instance['track_id'] = self.id_mapping[seq_name].get(track_id, 0)
                self.sequences[seq_name][frame_id] = len(self.metas)
                if self.label_mapping is not None:
                    data = self.set_label_mapping(data, self.label_mapping[idx], 0)
                    data['image_source'] = idx
                self.metas.append(data)
                self.seq_metas.append((seq_name, frame_id))
                if 'image_height' not in data or 'image_width' not in data:
                    logger.warning('image size is not provided, '
                                   'set aspect grouping to 1.')
                    self.aspect_ratios.append(1.0)
                else:
                    self.aspect_ratios.append(data['image_height'] / data['image_width'])
        self.seq_controls = {}
        for seq in self.sequences:
            frs = list(self.sequences[seq].keys())
            self.seq_controls[seq] = {
                'begin': min(frs),
                'end': max(frs),
            }
        # load annotations (and proposals)
        # self.img_infos = self.load_annotations(self.ann_file)
        # self.img_infos = self.make_items(self.img_infos)
        # filter images with no annotation during training
        # if self.is_train:
        #     valid_inds = self._filter_imgs()
        #     self.img_infos = [self.img_infos[i] for i in valid_inds]
        # original_len = len(self.img_infos)
        # if self.repeat_num is not None:
        #     assert isinstance(self.repeat_num, int)
        #     times = self.repeat_num // len(self.img_infos)
        #     rest = self.repeat_num % len(self.img_infos)
        #     repeat_ = []
        #     for i in range(times):
        #         repeat_ += self.img_infos
        #     if rest > 0:
        #         inds = self._sample_k_frames(len(self.img_infos), rest)
        #         for i in inds:
        #             repeat_.append(self.img_infos[i])
        #     self.img_infos = repeat_
        # set group flag for the sampler
        # if self.is_train:
        #     self._set_group_flag()
        # processing pipeline
        # self.pipeline = Compose(pipeline)
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
        gt_bboxes = []
        ig_bboxes = []
        for instance in data.get('instances', []):
            if instance['is_ignored']:
                ig_bboxes.append(instance['bbox'])
            else:
                gt_bboxes.append(instance['bbox'] + [instance['label'], instance['track_id']])

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
        options = [fr for fr in self.sequences[seq_name] if abs(
            fr - frame_id) <= self.frame_involved and fr != frame_id]
        if not self.random_select:
            old_state = np.random.get_state()
            new_state = np.random.MT19937(0).state
            np.random.set_state(new_state)
        chosen = np.random.choice(options, min(len(options), self.num_expected), replace=False)
        if len(chosen) < self.num_expected:
            options.append(frame_id)
            chosen2 = np.random.choice(options, self.num_expected - len(chosen))
            chosen = np.concatenate([chosen, chosen2])
        chosen = list(chosen)
        # print(seq_name, frame_id, chosen)

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
        for b_ix in range(len(image_info)):
            info = image_info[b_ix]
            height, width = map(int, info[3:5])
            img_id = image_ids[b_ix]
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
                    'bbox': bbox[1: 1 + 4].tolist(),
                    'score': score,
                    'label': int(cls)
                }
                if img_bboxes.shape[1] >= 8:
                    res['track_id'] = int(img_bboxes[i][8])
                out_res.append(res)
        return out_res

    # def make_items(self, img_infos):
    #     ret = []
    #     for vi in img_infos:
    #         if isinstance(vi, list):
    #             if self.k_frames is None:
    #                 k_frames = len(vi)
    #             else:
    #                 k_frames = self.k_frames
    #             sample_indexes = self._sample_k_frames(len(vi), k_frames)
    #             for i in sample_indexes:
    #                 img_info = vi[i]
    #                 first_frame = 0
    #                 last_frame = img_info['length'] - 1

    #                 def trim(x): return min(max(x, first_frame), last_frame)
    #                 # local
    #                 if self.random_select:
    #                     offsets = np.random.choice(
    #                         2 * self.frame_involved + 1, self.num_expected, replace=False) - self.frame_involved
    #                 else:
    #                     offsets = np.array(self._sample_k_frames(2 * self.frame_involved + 1, self.num_expected)) \
    #                         - self.frame_involved
    #                 local_infos = [vi[trim(i + i_)] for i_ in offsets]
    #                 if self.add_self:
    #                     local_infos.insert(0, vi[i])

    #                 # mem
    #                 if self.random_select:
    #                     offsets = np.random.choice(
    #                         2 * self.frame_involved + 1, self.num_expected, replace=False) - self.frame_involved
    #                 else:
    #                     offsets = np.array(self._sample_k_frames(2 * self.frame_involved + 1, self.num_expected)) \
    #                         - self.frame_involved
    #                 mem_infos = [
    #                     vi[trim(max(i - self.frame_involved * 2 - 1, 0) + i_)] for i_ in offsets]

    #                 # global
    #                 if self.random_select:
    #                     offsets = np.random.choice(
    #                         last_frame + 1, self.num_expected, replace=False)
    #                 else:
    #                     offsets = (np.array(self._sample_k_frames(
    #                         last_frame + 1, self.num_expected)) + last_frame - i) % last_frame
    #                 global_infos = [vi[i_] for i_ in offsets]

    #                 def fn_map(x): return x['filename']
    #                 def key_map(k): return (lambda x: x['ann'][k])
    #                 len_l = len(local_infos)
    #                 len_m = len(mem_infos)
    #                 len_g = len(global_infos)
    #                 data = dict(
    #                     width=img_info['width'],
    #                     height=img_info['height'],
    #                     filename=[img_info['filename'], *map(fn_map, local_infos), *map(
    #                         fn_map, mem_infos), *map(fn_map, global_infos)],
    #                     ann={k: [img_info['ann'][k], *map(key_map(k), local_infos), *map(key_map(k), mem_infos), *map(key_map(k), global_infos)]
    #                          for k in img_info['ann']},
    #                     splits={'_c': 0,
    #                             '_l': [1 + i for i in range(len_l)],
    #                             '_m': [len_l + 1 + i for i in range(len_m)],
    #                             '_g': [len_l + len_m + 1 + i for i in range(len_g)],
    #                             },
    #                 )
    #                 #print('before', data['filename'], data['ann'])
    #                 # if self.random_flip_pair:
    #                 #     if random.randint(0,1):
    #                 #         data['filename'].reverse()
    #                 #         for k in data['ann']:
    #                 #             data['ann'][k].reverse()
    #                 #print('after', data['filename'], data['ann'])
    #                 ret.append(data)
    #         else:
    #             img_info = vi
    #             data = dict(width=img_info['width'], height=img_info['height'],
    #                         filename=[img_info['filename']]
    #                         * (self.num_expected * 3 + 1),
    #                         ann={
    #                             k: [img_info['ann'][k]] * (self.num_expected * 3 + 1) for k in img_info['ann']},
    #                         splits={'_c': 0,
    #                                 '_l': [1 + i for i in range(self.num_expected)],
    #                                 '_m': [self.num_expected + 1 + i for i in range(self.num_expected)],
    #                                 '_g': [self.num_expected * 2 + 1 + i for i in range(self.num_expected)],
    #                                 },
    #                         )
    #             ret.append(data)
    #     return ret

    # # @staticmethod
    # # def _additional_check(res, is_train=True):
    # #     ret = []
    # #     def check_(x):
    # #         if not is_train: return True
    # #         return x['ann']['bboxes'].shape[0] > 0
    # #     for one in res:
    # #         # h, w = one['width'], one['height']
    # #         # mx_ratio = max(h/w, w/h)
    # #         # if mx_ratio < 4:
    # #         #     ret.append(one)
    # #         if isinstance(one, list):
    # #             ret_one = []
    # #             for two in one:
    # #                 if check_(two):
    # #                     ret_one.append(two)
    # #             ret.append(ret_one)
    # #         elif isinstance(one, dict):
    # #             if check_(one):
    # #                 ret.append(one)
    # #     return ret

    # @staticmethod
    # def _sample_k_frames(n, k):
    #     if n < k:
    #         return [i for i in range(n)]
    #     d = n / float(k)
    #     eps = 1e-5
    #     pre = d / 2
    #     ret = []
    #     cnt = 0
    #     while cnt < k:
    #         ret.append(int(pre + eps))
    #         pre += d
    #         cnt += 1
    #     return ret

    # def _filter_imgs(self, min_size=32):
    #     """Filter images too small."""
    #     valid_inds = []
    #     for i, img_info in enumerate(self.img_infos):
    #         flag = 0

    #         def size_check(img_info): return min(
    #             img_info['width'], img_info['height']) >= min_size
    #         def gt_check(
    #             img_info): return self.test_mode or img_info['ann']['bboxes'][0].shape[0] > 0
    #         checks = [size_check, gt_check]
    #         for chk in checks:
    #             if chk(img_info):
    #                 flag += 1
    #             else:
    #                 break
    #         if flag == len(checks):
    #             valid_inds.append(i)
    #     return valid_inds

    # def _set_group_flag(self):
    #     """Set flag according to image aspect ratio.

    #     Images with aspect ratio greater than 1 will be set as group 1,
    #     otherwise group 0.
    #     """
    #     self.flag = np.zeros(len(self), dtype=np.uint8)
    #     for i in range(len(self)):
    #         img_info = self.img_infos[i]
    #         if img_info['width'] / img_info['height'] > 1:
    #             self.flag[i] = 1

    # def load_anno_vid(self, contents, **kwargs):

    #     self.CLASSES = tuple(VID_CLASS_TO_INDEX.keys())[1:]
    #     img_infos = []
    #     hnds = []
    #     pool = mp.Pool(max(1, self.cpus - 2))
    #     slices = (len(contents) + pool._processes - 1) // pool._processes
    #     for si in range(pool._processes):
    #         hnds.append(pool.apply_async(
    #             load_anno_vid_,
    #             (self, contents[si * slices: si * slices + slices]), kwargs))
    #     logger.info('%d ImageNetVID loading tasks created.' % len(hnds))
    #     for i, hnd in enumerate(hnds):
    #         img_vids = hnd.get()
    #         senseTk.functions.brief_pos_bar((i + 1.) / len(hnds))
    #         img_infos += img_vids
    #     return img_infos

    # def load_anno_mot(self, contents, **kwargs):
    #     self.CLASSES = ['human']
    #     img_infos = []
    #     iniparser = configparser.ConfigParser()
    #     for one in contents:
    #         img_vid = []
    #         seq = os.path.basename(one)
    #         seq_dir = os.path.join(self.data_root, one)
    #         gt = senseTk.common.TrackSet(
    #             os.path.join(seq_dir, 'gt', 'gt.txt'),
    #             formatter='fr.i id.i x1 y1 w h st.i la.i cf',
    #             filter=lambda x: x.status == 1 and x.conf > 0.1
    #         )
    #         gt_ = senseTk.common.TrackSet()
    #         for oid in gt.allId():
    #             u = gt(oid)
    #             nid_ = self.next_id()
    #             for f in u.frameRange():
    #                 if u[f]:
    #                     u[f][0].uid = nid_
    #                     gt_.append_data(u[f][0])
    #         gt = gt_
    #         for i in gt.frameRange():
    #             for j in gt[i]:
    #                 j.label = 1
    #         img_dir = os.path.join(self.data_root, self.img_prefix, seq)
    #         if not os.path.exists(img_dir):
    #             img_dir = os.path.join(seq_dir, 'img1')
    #         formatter = senseTk.functions.autoPattern(img_dir)
    #         meta_file = os.path.join(seq_dir, 'seqinfo.ini')
    #         meta = iniparser.read(meta_file)
    #         img_info = dict(width=iniparser.getint('Sequence', 'imWidth'),
    #                         height=iniparser.getint('Sequence', 'imHeight'),
    #                         fps=iniparser.getint('Sequence', 'frameRate'),
    #                         length=iniparser.getint('Sequence', 'seqLength'),
    #                         start_with_zero=False,
    #                         seq=one,
    #                         )
    #         for i in range(img_info['length']):
    #             i += 1
    #             filename = os.path.join(img_dir, (formatter % i))
    #             img_one = img_info.copy()
    #             img_one['filename'] = filename
    #             img_one['ann'] = generate_labels(gt[i])
    #             img_one['fr'] = i
    #             img_vid.append(img_one)
    #         img_infos.append(img_vid)
    #     return img_infos

    # def load_anno_hie(self, contents, **kwargs):
    #     self.CLASSES = ['human']
    #     img_infos = []
    #     iniparser = configparser.ConfigParser()
    #     for one in contents:
    #         img_vid = []
    #         seq = os.path.basename(one)
    #         img_dir = os.path.join(self.data_root, 'images', one)
    #         seq_len = len(os.listdir(img_dir))
    #         gt = senseTk.common.TrackSet(
    #             os.path.join(self.data_root, 'labels',
    #                          'train', 'track1', seq + '.txt'),
    #             formatter='fr.i id.i x1 y1 w h st.i -1 -1 -1',
    #             filter=lambda x: x.status == 1
    #         )
    #         gt_ = senseTk.common.TrackSet()
    #         for oid in gt.allId():
    #             u = gt(oid)
    #             nid_ = self.next_id()
    #             for f in u.frameRange():
    #                 if u[f]:
    #                     u[f][0].uid = nid_
    #                     gt_.append_data(u[f][0])
    #         gt = gt_
    #         for i in gt.frameRange():
    #             for j in gt[i]:
    #                 j.label = 1
    #                 j.fr += 1
    #         formatter = senseTk.functions.autoPattern(img_dir)
    #         test_img = formatter % 1
    #         im = mmcv.imread(os.path.join(img_dir, test_img))
    #         imHeight, imWidth, C = im.shape
    #         img_info = dict(width=imWidth,
    #                         height=imHeight,
    #                         length=seq_len,
    #                         start_with_zero=False,
    #                         seq=seq,
    #                         fps=None)
    #         for i in range(img_info['length']):
    #             i += 1
    #             filename = os.path.join(img_dir, (formatter % i))
    #             img_one = img_info.copy()
    #             img_one['filename'] = filename
    #             img_one['ann'] = generate_labels(gt[i])
    #             img_one['fr'] = i
    #             img_vid.append(img_one)
    #         img_infos.append(img_vid)
    #     return img_infos

    # def load_anno_det(self, contents, **kwargs):

    #     self.CLASSES = tuple(VID_CLASS_TO_INDEX.keys())[1:]
    #     img_infos = []
    #     hnds = []
    #     pool = mp.Pool(max(1, self.cpus - 2))
    #     slices = (len(contents) + pool._processes - 1) // pool._processes
    #     for si in range(pool._processes):
    #         hnds.append(pool.apply_async(
    #             load_anno_det_,
    #             (self, contents[si * slices: si * slices + slices]), kwargs))
    #     logger.info('%d ImageNetDET loading tasks created.' % len(hnds))
    #     for i, ret in enumerate(hnds):
    #         img_infos += ret.get()
    #         senseTk.functions.brief_pos_bar((i + 1.) / len(hnds))
    #     return img_infos

    # def load_anno_crowd(self, contents, ground_truth, **kwargs):
    #     self.CLASSES = ['human']
    #     img_infos = []
    #     gt_path = os.path.join(self.data_root, ground_truth)
    #     gt = {}
    #     with open(gt_path) as fd:
    #         lines = fd.readlines()
    #         for line in lines:
    #             u = json.loads(line)
    #             gt[u['ID']] = u
    #     for one in contents:
    #         filename = os.path.join(
    #             self.data_root, self.img_prefix, one + '.jpg')
    #         imWidth = 720
    #         imHeight = 480
    #         gt_i = []
    #         for g in gt[one]['gtboxes']:
    #             ignore = g['extra'].pop('ignore', 0)
    #             if g['tag'] == 'person' and not ignore:
    #                 x1, y1, w, h = g['fbox']
    #                 uid = self.next_id()
    #                 gt_i.append(senseTk.common.Det(
    #                     x1, y1, w, h, cls=1, uid=uid))
    #         img_info = dict(width=imWidth, height=imHeight)
    #         img_info['filename'] = filename
    #         img_info['ann'] = generate_labels(gt_i)
    #         if len(img_info['ann']['bboxes']):
    #             img_infos.append(img_info)
    #     return img_infos

    # def load_anno_caltech(self, contents, **kwargs):
    #     self.CLASSES = ['human']
    #     img_infos = []
    #     for one in contents:
    #         img_vid = []
    #         seq = os.path.basename(one)
    #         img_dir = os.path.join(self.data_root, one + '.seq.imgs')
    #         seq_len = len(os.listdir(img_dir))
    #         gt = senseTk.common.TrackSet()
    #         vbb = os.path.join(self.data_root, 'annotations', one + '.vbb')
    #         a = scipyio.loadmat(vbb)
    #         a = a['A']
    #         labels = a['objLbl'][0][0][0]
    #         objl = a['objLists'][0][0][0]
    #         n = int(a['nFrame'][0][0])
    #         assert n == seq_len
    #         for i in range(n):
    #             if len(objl[i]) <= 0:
    #                 continue
    #             objList = objl[i][0]
    #             for one in objList:
    #                 occl = float(one['occl'])
    #                 if occl > 0.7:
    #                     continue
    #                 uid = int(one['id'])
    #                 label = str(labels[uid - 1][0])
    #                 if label.upper() not in ['PERSON']:
    #                     continue
    #                 pos = one['pos'][0]
    #                 x, y, w, h = map(float, pos)
    #                 if w * h < 450:
    #                     continue
    #                 d = Det(x, y, w, h)
    #                 d.fr = i + 1
    #                 d.label = 1
    #                 d.uid = uid
    #                 # d.st = st
    #                 gt.append_data(d)
    #         gt_ = senseTk.common.TrackSet()
    #         for oid in gt.allId():
    #             u = gt(oid)
    #             nid_ = self.next_id()
    #             for f in u.frameRange():
    #                 if u[f]:
    #                     u[f][0].uid = nid_
    #                     gt_.append_data(u[f][0])
    #         gt = gt_
    #         formatter = senseTk.functions.autoPattern(img_dir)
    #         test_img = formatter % 1
    #         im = mmcv.imread(os.path.join(img_dir, test_img))
    #         imHeight, imWidth, C = im.shape
    #         img_info = dict(width=imWidth,
    #                         height=imHeight,
    #                         length=seq_len,
    #                         start_with_zero=False,
    #                         seq=seq,
    #                         fps=None)
    #         for i in range(img_info['length']):
    #             i += 1
    #             filename = os.path.join(img_dir, (formatter % i))
    #             img_one = img_info.copy()
    #             img_one['filename'] = filename
    #             img_one['ann'] = generate_labels(gt[i])
    #             img_one['fr'] = i
    #             img_vid.append(img_one)
    #         img_infos.append(img_vid)
    #     return img_infos

    # def load_annotations(self, ann_file):
    #     with open(ann_file) as fd:
    #         contents = yaml.safe_load(fd)
    #     typ = contents.pop('type')
    #     settings = contents.get('settings', {})
    #     settings['is_train'] = self.is_train
    #     self.k_frames = settings.get('k_frames', None)
    #     if os.path.exists(ann_file + '.%s.pkl' % self.suffix):
    #         ann_file += '.%s.pkl' % self.suffix
    #     ext = os.path.splitext(ann_file)[-1]
    #     if ext == '.pkl':
    #         with open(ann_file, 'rb') as fd:
    #             # ret = self._additional_check(pickle.load(fd), is_train=self.is_train)
    #             ret = pickle.load(fd)
    #             logger.info('%s: previous cache found' % (ann_file))
    #             logger.info('%s: %d images/videos in total' %
    #                         (ann_file, len(ret)))
    #         return ret
    #     self.cpus = mp.cpu_count()
    #     # self.cpus = 1
    #     if self.data_root is None:
    #         self.data_root = ''
    #     func_map = {
    #         'ImageNetDET': 'det',
    #         'ImageNetVID': 'vid',
    #         'MOT': 'mot',
    #         'HumanInEvents': 'hie',
    #         'CrowdHuman': 'crowd',
    #         'CalTech': 'caltech',
    #     }
    #     contents = contents['content']
    #     result = getattr(self, 'load_anno_%s' %
    #                      func_map[typ])(contents, **settings)
    #     with open(ann_file + '.%s.pkl' % self.suffix, 'wb') as fd:
    #         pickle.dump(result, fd)
    #     logger.info('%s: %d images in total' % (ann_file, len(result)))
    #     return result

    # def pre_pipeline(self, results):
    #     results['img_prefix'] = self.img_prefix
    #     results['seg_prefix'] = self.seg_prefix
    #     results['bbox_fields'] = []
    #     results['mask_fields'] = []

    # def prepare_train_img(self, idx):
    #     img_info = self.img_infos[idx]
    #     ann_info = self.get_ann_info(idx)
    #     results = dict(img_info=img_info, ann_info=ann_info)
    #     # if self.proposals is not None:
    #     #     results['proposals'] = self.proposals[idx]
    #     self.pre_pipeline(results)
    #     results = self.pipeline(results)
    #     if results is None:
    #         print('Null index %d' % idx)
    #     return results

    # def prepare_test_img(self, idx):
    #     img_info = self.img_infos[idx]
    #     results = dict(img_info=img_info)
    #     # if self.proposals is not None:
    #     #     results['proposals'] = self.proposals[idx]
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)
