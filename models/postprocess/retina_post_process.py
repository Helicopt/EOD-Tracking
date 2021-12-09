import copy

import torch
import torch.nn as nn
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.model import accuracy as A  # noqa F401
from eod.tasks.det.models.utils.anchor_generator import build_anchor_generator
from eod.models.losses import build_loss
from eod.tasks.det.models.losses.entropy_loss import apply_class_activation
from eod.tasks.det.models.utils.bbox_helper import offset2bbox, bbox_iou_overlaps
from eod.utils.env.dist_helper import allreduce, env
from eod.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from eod.tasks.det.models.postprocess.roi_predictor import build_roi_predictor
from ...utils.debug import info_debug


__all__ = ['BasewIDPostProcess', 'IOUwIDPostProcess']


@MODULE_ZOO_REGISTRY.register('retina_post_w_id')
class BasewIDPostProcess(nn.Module):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, num_classes, num_ids, inplanes, cfg):
        super(BasewIDPostProcess, self).__init__()
        self.prefix = self.__class__.__name__

        self.num_classes = num_classes
        self.num_ids = num_ids
        self.inplanes = inplanes
        assert self.num_classes > 1 and self.num_ids > 1

        self.anchor_generator = build_anchor_generator(cfg['anchor_generator'])
        self.supervisor = build_roi_supervisor(cfg['roi_supervisor'])
        # rpn needs predict when training
        if 'train' in cfg:
            train_cfg = copy.deepcopy(cfg)
            train_cfg.update(train_cfg['train'])
            self.train_predictor = build_roi_predictor(train_cfg['roi_predictor'])
        else:
            self.train_predictor = None

        test_cfg = copy.deepcopy(cfg)
        test_cfg.update(test_cfg.get('test', {}))
        self.test_predictor = build_roi_predictor(test_cfg['roi_predictor'])

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.id_loss = build_loss(cfg['id_loss'])
        self.cls_use_ghm = 'ghm' in self.cls_loss.name
        self.loc_use_ghm = 'ghm' in self.loc_loss.name
        self.cls_use_qfl = 'quality_focal_loss' in self.cls_loss.name

    @property
    def predictor(self):
        return self.test_predictor

    @property
    def num_anchors(self):
        return self.anchor_generator.num_anchors

    @property
    def class_activation(self):
        return self.cls_loss.activation_type

    @property
    def with_background_channel(self):
        return self.class_activation == 'softmax'

    def apply_activation(self, mlvl_preds, remove_background_channel_if_any):
        """apply activation on permuted class prediction
        Arguments:
            - mlvl_pred (list (levels) of tuple (predictions) of Tensor): first element must
            be permuted class prediction with last dimension be class dimension
        """
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            if self.with_background_channel and remove_background_channel_if_any:
                cls_pred = cls_pred[..., 1:]
            mlvl_activated_preds.append((cls_pred, *preds[1:]))
        return mlvl_activated_preds

    def permute_preds(self, mlvl_preds):
        """Permute preds from [B, A*C, H, W] to [B, H*W*A, C] """
        mlvl_permuted_preds, mlvl_shapes = [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            b, _, h, w = preds[0].shape
            k = self.num_anchors * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes

    def prepare_preds(self, input, save_anchors=True):
        strides = input['strides']
        image_info = input['image_info']
        mlvl_raw_preds = input['preds']
        device = mlvl_raw_preds[0][0].device

        # [B, hi*wi*A, :]
        mlvl_preds, mlvl_shapes = self.permute_preds(mlvl_raw_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]

        # [hi*wi*A, 4], for C4 there is only one layer, for FPN generate anchors for each layer
        mlvl_anchors = self.anchor_generator.get_anchors(mlvl_shapes, device=device)
        if save_anchors:
            self.mlvl_anchors = mlvl_anchors
        return mlvl_preds, mlvl_anchors, mlvl_shapes

    def forward(self, input):
        if 'main' in input and 'ref' in input:
            input = input['main']
        image_info = input['image_info']

        mlvl_preds, mlvl_anchors, mlvl_shapes = self.prepare_preds(input)

        output = {}

        if self.training:
            targets = self.supervisor.get_targets(mlvl_anchors, input, mlvl_preds)
            losses = self.get_loss(targets, mlvl_preds, mlvl_shapes)
            output.update(losses)

            if self.train_predictor is not None:
                mlvl_preds = self.apply_activation(mlvl_preds, remove_background_channel_if_any=True)
                results = self.train_predictor.predict(mlvl_anchors, mlvl_preds, image_info)
                output.update(results)
        else:
            id_feats = [lvl_feats[2] for lvl_feats in input['roi_features']]
            k = self.num_anchors
            id_feats = [id_features.permute(0, 2, 3, 1).reshape(id_features.size(
                0), id_features.size(2) * id_features.size(3), 1, -1).repeat(1, 1, k, 1).reshape(id_features.size(
                    0), id_features.size(2) * id_features.size(3) * k, -1) for id_features in id_feats]
            mlvl_preds = self.apply_activation(mlvl_preds, remove_background_channel_if_any=True)
            results = self.test_predictor.predict(mlvl_anchors, mlvl_preds, id_feats, image_info)
            output.update(results)

        return output

    def get_loss(self, targets, mlvl_preds, mlvl_shapes=None):
        mlvl_shapes = [shape[:-1] for shape in mlvl_shapes]
        cls_target, loc_target, id_target, cls_mask, loc_mask, valid_id_mask = targets
        mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        id_pred = torch.cat(mlvl_id_pred, dim=1)
        del mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred

        pos_normalizer = max(1, torch.sum(loc_mask.float()).item())
        id_pos_normalizer = max(1, torch.sum(valid_id_mask.float()).item())

        valid_id_pred, valid_id_target = self._mask_tensor([id_pred, id_target.unsqueeze(-1)], valid_id_mask)
        valid_id_target = valid_id_target.flatten()

        # cls loss
        if self.cls_use_ghm:
            cls_loss = self.cls_loss(cls_pred, cls_target, mlvl_shapes=mlvl_shapes)
            acc = self._get_acc(cls_pred, cls_target)
            id_loss = self.id_loss(valid_id_pred, valid_id_target, mlvl_shapes=mlvl_shapes)
            acc_id = self._get_acc(valid_id_pred, valid_id_target)
        elif self.cls_use_qfl:
            B, K, _ = cls_pred.size()
            scores = cls_pred.new_zeros((B, K))
            pos_loc_target, pos_loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)
            if pos_loc_pred.numel():
                pos_decode_loc_target, pos_decode_loc_pred = self.decode_loc(pos_loc_target, pos_loc_pred.detach(), loc_mask)  # noqa
                scores[loc_mask] = bbox_iou_overlaps(pos_decode_loc_pred, pos_decode_loc_target, aligned=True)
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer, scores=scores)
            acc = self._get_acc(cls_pred, cls_target)
            valid_scores, = self._mask_tensor([scores.unsqueeze(-1)], valid_id_mask)
            valid_scores = valid_scores.squeeze(-1)
            id_loss = self.id_loss(valid_id_pred, valid_id_target,
                                   normalizer_override=pos_normalizer, scores=valid_scores)
            acc_id = self._get_acc(valid_id_pred, valid_id_target)
        else:
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer)
            acc = self._get_acc(cls_pred, cls_target)
            id_loss = self.id_loss(valid_id_pred, valid_id_target, normalizer_override=id_pos_normalizer)
            acc_id = self._get_acc(valid_id_pred, valid_id_target)

        # loc loss
        if self.loc_use_ghm:
            loc_loss = self.loc_loss(loc_pred, loc_target, loc_mask=loc_mask, mlvl_shapes=mlvl_shapes)
        else:
            loc_target, loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)
            if loc_mask.sum() == 0:
                loc_loss = loc_pred.sum()
            else:
                loc_loss_key_fields = getattr(self.loc_loss, "key_fields", set())
                loc_loss_kwargs = {}
                if "anchor" in loc_loss_key_fields:
                    loc_loss_kwargs["anchor"] = self.get_sample_anchor(loc_mask)
                loc_loss = self.loc_loss(loc_pred, loc_target, normalizer_override=pos_normalizer, **loc_loss_kwargs)

        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.id_loss': id_loss,
            self.prefix + '.accuracy': acc,
            self.prefix + '.accuracy_id': acc_id
        }

    def _get_acc(self, cls_pred, cls_target):
        acc = A.accuracy_v2(cls_pred, cls_target, self.class_activation)
        return acc

    def _mask_tensor(self, tensors, mask):
        """
        Arguments:
            - tensor: [[M, N], K]
            - mask: [[M, N]]
        """
        mask = mask.reshape(-1)
        masked_tensors = []
        for tensor in tensors:
            n_dim = tensor.shape[-1]
            masked_tensor = tensor.reshape(-1, n_dim)[mask]
            masked_tensors.append(masked_tensor)
        return masked_tensors

    def get_sample_anchor(self, loc_mask):
        all_anchors = torch.cat(self.mlvl_anchors, dim=0)
        batch_anchor = all_anchors.view(1, -1, 4).expand((loc_mask.shape[0], loc_mask.shape[1], 4))
        sample_anchor = batch_anchor[loc_mask].contiguous()
        return sample_anchor

    def decode_loc(self, loc_target, loc_pred, loc_mask):
        sample_anchor = self.get_sample_anchor(loc_mask)
        decode_loc_pred = offset2bbox(sample_anchor, loc_pred)
        decode_loc_target = offset2bbox(sample_anchor, loc_target)
        return decode_loc_target, decode_loc_pred


@MODULE_ZOO_REGISTRY.register('retina_post_iou_w_id')
class IOUwIDPostProcess(BasewIDPostProcess):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, num_classes, num_ids, inplanes, cfg):
        super(IOUwIDPostProcess, self).__init__(num_classes, num_ids, inplanes, cfg)
        self.iou_branch_loss = build_loss(cfg['iou_branch_loss'])

    def apply_activation(self, mlvl_preds, remove_background_channel_if_any=True):
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            iou_pred = apply_class_activation(preds[3], 'sigmoid')
            mlvl_activated_preds.append((cls_pred * iou_pred, *preds[1:]))
        return mlvl_activated_preds

    def get_loss(self, targets, mlvl_preds, mlvl_shapes=None):
        mlvl_shapes = [shape[:-1] for shape in mlvl_shapes]
        # info_debug([mlvl_preds, targets])
        cls_target, loc_target, id_target, iou_target, cls_mask, loc_mask, valid_id_mask = targets
        mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred, mlvl_iou_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        id_pred = torch.cat(mlvl_id_pred, dim=1)
        iou_pred = torch.cat(mlvl_iou_pred, dim=1)

        del mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred, mlvl_iou_pred

        pos_normalizer = max(1, torch.sum(loc_mask.float()).item())
        id_pos_normalizer = max(1, torch.sum(valid_id_mask.float()).item())

        valid_id_pred, valid_id_target = self._mask_tensor([id_pred, id_target.unsqueeze(-1)], valid_id_mask)
        valid_id_target = valid_id_target.flatten()
        # cls loss
        if self.cls_use_ghm:
            cls_loss = self.cls_loss(cls_pred, cls_target, mlvl_shapes=mlvl_shapes)
            acc = self._get_acc(cls_pred, cls_target)
            id_loss = self.id_loss(valid_id_pred, valid_id_target, mlvl_shapes=mlvl_shapes)
            acc_id = self._get_acc(valid_id_pred, valid_id_target)
        elif self.cls_use_qfl:
            B, K, _ = cls_pred.size()
            scores = cls_pred.new_zeros((B, K))
            pos_loc_target, pos_loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)
            if pos_loc_pred.numel():
                pos_decode_loc_target, pos_decode_loc_pred = self.decode_loc(pos_loc_target, pos_loc_pred.detach(), loc_mask)  # noqa
                temp = bbox_iou_overlaps(pos_decode_loc_pred, pos_decode_loc_target, aligned=True)
                scores[loc_mask] = temp.to(scores.dtype)
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer, scores=scores)
            valid_scores, = self._mask_tensor([scores.unsqueeze(-1)], valid_id_mask)
            valid_scores = valid_scores.squeeze(-1)
            id_loss = self.id_loss(valid_id_pred, valid_id_target,
                                   normalizer_override=id_pos_normalizer, scores=valid_scores)
            acc = self._get_acc(cls_pred, cls_target)
            acc_id = self._get_acc(valid_id_pred, valid_id_target)
        else:
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer)
            acc = self._get_acc(cls_pred, cls_target)
            id_loss = self.id_loss(valid_id_pred, valid_id_target, normalizer_override=id_pos_normalizer)
            acc_id = self._get_acc(valid_id_pred, valid_id_target)

        # loc loss
        if self.loc_use_ghm:
            loc_loss = self.loc_loss(loc_pred, loc_target, loc_mask=loc_mask, mlvl_shapes=mlvl_shapes)
        else:
            loc_target, loc_pred, iou_pred = self._mask_tensor([loc_target, loc_pred, iou_pred], loc_mask)
            if loc_mask.sum() == 0:
                loc_loss = loc_pred.sum()
                iou_loss = iou_pred.sum()
            else:
                loc_loss_key_fields = getattr(self.loc_loss, "key_fields", set())
                loc_loss_kwargs = {}
                if "anchor" in loc_loss_key_fields:
                    loc_loss_kwargs["anchor"] = self.get_sample_anchor(loc_mask)
                if "weights" in loc_loss_key_fields:
                    loc_loss_kwargs["weights"] = iou_target
                weights_normalizer = self.get_weights_normalizer(iou_target)
                loc_loss = self.loc_loss(loc_pred, loc_target, normalizer_override=weights_normalizer, **loc_loss_kwargs)  # noqa
                iou_loss = self.iou_branch_loss(iou_pred.view(-1), iou_target, normalizer_override=pos_normalizer)

        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.id_loss': id_loss,
            self.prefix + '.accuracy': acc,
            self.prefix + '.accuracy_id': acc_id,
            self.prefix + '.iou_loss': iou_loss
        }

    def get_weights_normalizer(self, centerness_target):
        sum_centerness_targets = centerness_target.sum()
        allreduce(sum_centerness_targets)
        num_gpus = env.world_size
        ave_centerness_targets = max(sum_centerness_targets.item(), 1) / float(num_gpus)
        return ave_centerness_targets
