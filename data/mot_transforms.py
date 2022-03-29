from __future__ import division

# Standard Library
import copy
import math
from collections.abc import Sequence
import torchvision.transforms as transforms
import numbers
import sys
from PIL import Image

# Import from third library
import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation
)
from eod.utils.general.registry_factory import AUGMENTATION_REGISTRY, BATCHING_REGISTRY
from eod.utils.general.global_flag import ALIGNED_FLAG
from eod.data.datasets.transforms import (
    Augmentation,
    has_gt_bboxes,
    has_gt_ignores,
    has_gt_keyps,
    has_gt_masks,
    has_gt_semantic_seg,
    check_fake_gt
)
from eod.tasks.det.data.datasets.det_transforms import Flip
from eod.data.data_utils import (
    coin_tossing,
    get_image_size,
    is_numpy_image,
    is_pil_image
)

# TODO: Check GPU usage after move this setting down from upper line
cv2.ocl.setUseOpenCL(False)


__all__ = [
    'ImageWindow',
    'MOTFlip',
]


def tensor2numpy(data):
    if check_fake_gt(data.gt_bboxes) and check_fake_gt(data.gt_ignores):
        return np.zeros((0, 5))
    if data.gt_bboxes is None:
        gts = np.zeros((0, 5))
    else:
        gts = data.gt_bboxes.cpu().numpy()
    if data.gt_ignores is not None:
        igs = data.gt_ignores.cpu().numpy()
    else:
        igs = np.zeros((0, 4))
    if igs.shape[0] != 0:
        ig_bboxes = np.hstack([igs, -np.ones(igs.shape[0])[:, np.newaxis]])
        new_gts = np.concatenate((gts, ig_bboxes))
    else:
        new_gts = gts
    return new_gts


def numpy2tensor(boxes_t):
    ig_bboxes_t = boxes_t[boxes_t[:, 4] == -1][:, :4]
    gt_bboxes_t = boxes_t[boxes_t[:, 4] != -1]
    gt_bboxes_t = torch.as_tensor(gt_bboxes_t, dtype=torch.float32)
    ig_bboxes_t = torch.as_tensor(ig_bboxes_t, dtype=torch.float32)
    if len(ig_bboxes_t) == 0:
        ig_bboxes_t = torch.zeros((1, 4))
    if len(gt_bboxes_t) == 0:
        gt_bboxes_t = torch.zeros((1, 5))
    return gt_bboxes_t, ig_bboxes_t


def np_bbox_iof_overlaps(b1, b2):
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    return inter_area / np.maximum(area1[:, np.newaxis], 1)


def boxes2polygons(boxes, sample=4):
    num_boxes = len(boxes)
    boxes = np.array(boxes).reshape(num_boxes, -1)
    a = (boxes[:, 2] - boxes[:, 0] + 1) / 2
    b = (boxes[:, 3] - boxes[:, 1] + 1) / 2
    x = (boxes[:, 2] + boxes[:, 0]) / 2
    y = (boxes[:, 3] + boxes[:, 1]) / 2

    angle = 2 * math.pi / sample
    polygons = np.zeros([num_boxes, sample, 2], dtype=np.float32)
    for i in range(sample):
        polygons[:, i, 0] = a * math.cos(i * angle) + x
        polygons[:, i, 1] = b * math.sin(i * angle) + y
    return polygons


@AUGMENTATION_REGISTRY.register('window')
class ImageWindow(Augmentation):
    """ random crop image base gt and crop region (iof)
    """

    def __init__(self, means=127.5, scale=600, crop_prob=0.5):
        self.means = means
        self.img_scale = scale
        self.crop_prob = crop_prob

    def _nobbox_crop(self, width, height, image):
        h = w = min(width, height)
        if width == w:
            left = 0
        else:
            left = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((left, t, left + w, t + h))
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        return image_t

    def _random_crop(self, width, height, image, new_gts):
        for _ in range(100):
            if not coin_tossing(self.crop_prob):
                scale = 1
            else:
                scale = random.uniform(0.5, 1.)
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            if width == w:
                left = 0
            else:
                left = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((left, t, left + w, t + h))

            value = np_bbox_iof_overlaps(new_gts, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (new_gts[:, :2] + new_gts[:, 2:4]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = new_gts[mask].copy()

            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + ALIGNED_FLAG.offset) / w * self.img_scale
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + ALIGNED_FLAG.offset) / h * self.img_scale
            mask_b = np.minimum(b_w_t, b_h_t) >= 6.0

            if boxes_t.shape[0] == 0 or mask_b.shape[0] == 0:
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], roi[2:])
            boxes_t[:, 2:4] -= roi[:2]
            return image_t, boxes_t
        long_side = max(width, height)
        if len(image.shape) == 3:
            image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
        else:
            image_t = np.empty((long_side, long_side), dtype=image.dtype)
        image_t[:, :] = self.means
        image_t[0:0 + height, 0:0 + width] = image
        return image_t, new_gts

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"

        output = copy.copy(data)
        image = output.image
        new_gts = tensor2numpy(output)
        height, width = image.shape[:2]
        if check_fake_gt(new_gts):
            output.image = self._nobbox_crop(width, height, image)
            return output
        crop_image, boxes_t = self._random_crop(width, height, image, new_gts)
        gt_bboxes, ig_bboxes = numpy2tensor(boxes_t)
        output.gt_bboxes = gt_bboxes
        output.gt_ignores = ig_bboxes
        output.image = crop_image
        return output


@AUGMENTATION_REGISTRY.register('mot_flip')
class MOTFlip(Flip):

    def __init__(self, flip_p, num_orient_class=0):
        super(MOTFlip, self).__init__(flip_p)
        self.orient_div = num_orient_class

    def flip_boxes(self, boxes, width):
        x1 = boxes[:, 0].clone().detach()
        x2 = boxes[:, 2].clone().detach()
        boxes[:, 0] = width - ALIGNED_FLAG.offset - x2
        boxes[:, 2] = width - ALIGNED_FLAG.offset - x1
        if self.orient_div > 0 and boxes.shape[1] > 6:
            new_cls = boxes[:, 6].clone().detach()
            upper = new_cls > self.orient_div // 2
            lower = new_cls <= self.orient_div // 2
            new_cls[lower] = self.orient_div // 2 - new_cls[lower]
            new_cls[upper] = self.orient_div + self.orient_div // 2 - new_cls[upper]
            new_cls[new_cls == self.orient_div] = 0
            boxes[:, 6] = new_cls
            new_reg = boxes[:, 7].clone().detach()
            boxes[:, 7] = - new_reg
        return boxes
