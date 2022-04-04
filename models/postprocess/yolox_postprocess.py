import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from eod.utils.model import accuracy as A  # noqa F401
from eod.tasks.det.models.utils.anchor_generator import build_anchor_generator
from eod.models.losses import build_loss
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.tasks.det.models.utils.bbox_helper import bbox_iou_overlaps as bbox_overlaps

from eod.utils.env.dist_helper import allreduce, env
from eod.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from eod.tasks.det.models.postprocess.roi_predictor import build_roi_predictor
from ...utils.debug import logger_print


__all__ = ['YoloxwIDPostProcess']


@MODULE_ZOO_REGISTRY.register('yolox_post_w_id')
class YoloxwIDPostProcess(nn.Module):
    def __init__(self,
                 num_classes,
                 num_ids,
                 inplanes,
                 cfg,
                 dismiss_aug=False,
                 norm_on_bbox=False,
                 balanced_loss_weight='none',
                 balance_scale=1.,
                 all_reduce_norm=True,
                 use_l1=False):
        super(YoloxwIDPostProcess, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_classes = num_classes
        self.num_ids = num_ids
        self.inplanes = inplanes
        self.tocaffe = False
        self.norm_on_bbox = norm_on_bbox
        test_cfg = copy.deepcopy(cfg)
        test_cfg.update(test_cfg.get('test', {}))
        train_cfg = copy.deepcopy(cfg)
        train_cfg.update(train_cfg.get('train', {}))
        self.supervisor = build_roi_supervisor(train_cfg['roi_supervisor'])
        self.predictor = build_roi_predictor(test_cfg['roi_predictor'])
        self.point_generator = build_anchor_generator(cfg['anchor_generator'])
        self.dense_points = self.point_generator.dense_points
        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.obj_loss = build_loss(cfg['obj_loss'])
        self.id_loss = build_loss(cfg['id_loss'])
        self.l1_loss = nn.L1Loss(reduction="none")
        self.use_l1 = use_l1
        self.dismiss_aug = dismiss_aug
        self.balanced_loss_weight = balanced_loss_weight
        self.balance_scale = balance_scale
        if self.balanced_loss_weight == 'none':
            self.lw_id = 1.
            self.lw_det = 1.
        elif self.balanced_loss_weight == 'auto':
            self.lw_id = nn.Parameter(-0.5 * torch.ones(1))
            self.lw_det = nn.Parameter(-0.6 * torch.ones(1))
        else:
            assert False, '%s not defined' % self.balanced_loss_weight

        # config
        self.cfg = copy.deepcopy(cfg)
        self.all_reduce_norm = all_reduce_norm

    @property
    def class_activation(self):
        return self.cls_loss.activation_type

    def apply_activation(self, mlvl_preds):
        """apply activation on permuted class prediction
        Arguments:
            - mlvl_pred (list (levels) of tuple (predictions) of Tensor): first element must
            be permuted class prediction with last dimension be class dimension
        """
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = preds[0].sigmoid()
            obj_pred = preds[3].sigmoid()
            cls_pred *= obj_pred
            mlvl_activated_preds.append((cls_pred, *preds[1:]))
        return mlvl_activated_preds

    def permute_preds(self, mlvl_preds):
        """Permute preds from [B, A*C, H, W] to [B, H*W*A, C] """
        mlvl_permuted_preds, mlvl_shapes = [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            b, _, h, w = preds[0].shape
            k = self.dense_points * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes

    def prepare_preds(self, input):
        strides = input['strides']
        mlvl_preds = input['preds']

        # [B, hi*wi*A, :]
        mlvl_preds, mlvl_shapes = self.permute_preds(mlvl_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]

        # [hi*wi*A, 2], for C4 there is only one layer, for FPN generate anchors for each layer
        mlvl_locations = self.point_generator.get_anchors(mlvl_shapes, device=mlvl_preds[0][0].device)

        mlvl_ori_loc_preds = None
        if self.use_l1:
            mlvl_ori_loc_preds = []
        for l_ix in range(len(mlvl_locations)):
            # add preds for l1 loss
            if self.use_l1:
                mlvl_ori_loc_preds.append(mlvl_preds[l_ix][1].clone())
            mlvl_preds[l_ix][1][..., :2] *= strides[l_ix]
            mlvl_preds[l_ix][1][..., :2] += mlvl_locations[l_ix]
            if self.norm_on_bbox:
                mlvl_preds[l_ix][1][..., 2:4] = F.relu(mlvl_preds[l_ix][1][..., 2:4])  # , inplace=True)
                mlvl_preds[l_ix][1][..., 2:4] *= strides[l_ix]
            else:
                mlvl_preds[l_ix][1][..., 2:4] = torch.exp(mlvl_preds[l_ix][1][..., 2:4]) * strides[l_ix]
        return mlvl_preds, mlvl_locations, mlvl_ori_loc_preds

    def forward(self, input):
        noaug_flag = input['noaug_flag']
        if 'main' in input and 'ref' in input:
            input = input['main']

        mlvl_preds, mlvl_locations, mlvl_ori_loc_preds = self.prepare_preds(input)

        if self.training:
            targets = self.supervisor.get_targets(mlvl_locations, input, mlvl_preds)
            if not self.use_l1:
                losses = self.get_loss(targets, mlvl_preds, noaug_flag=noaug_flag)
            else:
                losses = self.get_loss(targets, mlvl_preds, mlvl_ori_loc_preds, noaug_flag=noaug_flag)
            return losses
        else:
            id_feats = [lvl_feats[2] for lvl_feats in input['roi_features']]
            with torch.no_grad():
                mlvl_preds = self.apply_activation(mlvl_preds)
                results = self.predictor.predict(mlvl_preds, id_feats)
                return results

    def get_acc(self, cls_pred, cls_targets):
        max_value, cls_targets_acc = cls_targets.topk(1, 1, True, True)
        cls_targets_acc += 1
        cls_targets_acc[max_value == 0] = 0
        acc = A.accuracy_v2(cls_pred, cls_targets_acc, activation_type='qfl')
        return acc

    def get_loss(self, targets, mlvl_preds, mlvl_ori_loc_preds=None, noaug_flag=None):
        mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred, mlvl_obj_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        id_pred = torch.cat(mlvl_id_pred, dim=1)
        obj_pred = torch.cat(mlvl_obj_pred, dim=1)
        if self.use_l1:
            ori_loc_preds = torch.cat(mlvl_ori_loc_preds, dim=1)
            del mlvl_ori_loc_preds
        del mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred, mlvl_obj_pred

        cls_targets, reg_targets, id_targets, obj_targets, ori_reg_targets, fg_masks, num_fgs, valid_id_masks = targets
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        id_targets = torch.cat(id_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        if noaug_flag is not None and self.dismiss_aug:
            valid_id_masks = [(m & o) for m, o in zip(valid_id_masks, noaug_flag)]
        valid_id_masks = torch.cat(valid_id_masks, 0)
        if self.use_l1:
            ori_reg_targets = torch.cat(ori_reg_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.all_reduce_norm and dist.is_initialized():
            num_fgs = self.get_ave_normalizer(fg_masks)
            num_ids = self.get_ave_normalizer(valid_id_masks)
        else:
            num_fgs = fg_masks.sum().item()
            num_ids = valid_id_masks.sum().item()
        num_fgs = max(num_fgs, 1)
        num_ids = max(num_ids, 1)

        cls_pred = cls_pred.reshape(-1, self.num_classes - 1)
        cls_loss = self.cls_loss(cls_pred[fg_masks], cls_targets, normalizer_override=num_fgs)
        # print(cls_pred[fg_masks].shape, cls_targets.shape)
        id_pred = id_pred.reshape(-1, self.num_ids)
        id_loss = self.id_loss(id_pred[fg_masks][valid_id_masks],
                               id_targets[valid_id_masks][:, 1:], normalizer_override=num_ids)
        # print(id_pred[fg_masks].shape, id_targets.shape)
        acc = self.get_acc(cls_pred[fg_masks], cls_targets)
        acc_id = self.get_acc(id_pred[fg_masks][valid_id_masks], id_targets[valid_id_masks][:, 1:])

        loc_target = reg_targets.reshape(-1, 4)
        loc_pred = loc_pred.reshape(-1, 4)
        # loc_preds [xc, yc, w, h] -> [x1, y1, x2, y2]
        loc_pred_x1 = loc_pred[:, 0] - loc_pred[:, 2] / 2
        loc_pred_y1 = loc_pred[:, 1] - loc_pred[:, 3] / 2
        loc_pred_x2 = loc_pred[:, 0] + loc_pred[:, 2] / 2
        loc_pred_y2 = loc_pred[:, 1] + loc_pred[:, 3] / 2
        loc_pred = torch.stack([loc_pred_x1, loc_pred_y1, loc_pred_x2, loc_pred_y2], dim=-1)
        obj_pred = obj_pred.reshape(-1)
        if loc_pred[fg_masks].numel() > 0:
            loc_loss = self.loc_loss(loc_pred[fg_masks], loc_target, normalizer_override=num_fgs)
        else:
            loc_loss = loc_pred[fg_masks].sum()
        obj_loss = self.obj_loss(obj_pred, obj_targets, normalizer_override=num_fgs)
        # add l1 loss
        if self.use_l1:
            if ori_loc_preds.numel() > 0:
                l1_loss = (self.l1_loss(ori_loc_preds.view(-1, 4)[fg_masks], ori_reg_targets)).sum() / num_fgs
            else:
                l1_loss = ori_loc_preds.sum()
        else:
            l1_loss = torch.tensor(0.0).cuda()
        lw_id = self.lw_id
        lw_det = self.lw_det
        if self.balanced_loss_weight == 'auto':
            lw_id = torch.exp(-lw_id)
            lw_det = torch.exp(-lw_det)
            return {
                self.prefix + '.cls_loss': cls_loss * lw_det,
                self.prefix + '.loc_loss': loc_loss * lw_det,
                self.prefix + '.id_loss': id_loss * lw_id,
                self.prefix + '.obj_loss': obj_loss * lw_det,
                self.prefix + '.l1_loss': l1_loss * lw_det,
                self.prefix + '.lwnorm_loss': (self.lw_id + self.lw_det) * self.balance_scale,
                self.prefix + '.accuracy': acc,
                self.prefix + '.accuracy_id': acc_id,
            }
        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.id_loss': id_loss,
            self.prefix + '.obj_loss': obj_loss,
            self.prefix + '.l1_loss': l1_loss,
            self.prefix + '.accuracy': acc,
            self.prefix + '.accuracy_id': acc_id,
        }

    def get_ave_normalizer(self, _mask):
        ave_mask = torch.sum(_mask)
        if env.world_size > 1:
            allreduce(ave_mask)
        num_gpus = env.world_size
        ave_normalizer = max(ave_mask.item(), 1) / float(num_gpus)
        return ave_normalizer


@MODULE_ZOO_REGISTRY.register('yolox_post_w_assoc')
class YoloxwAssocProcess(YoloxwIDPostProcess):

    def __init__(self,
                 num_classes,
                 num_ids,
                 inplanes,
                 cfg,
                 iou_thr=0.7,
                 **kwargs):
        super().__init__(num_classes,
                         num_ids,
                         inplanes,
                         cfg, **kwargs)
        self.assoc_loss = build_loss(cfg['assoc_loss'])
        self.iou_thr = iou_thr

    @torch.no_grad()
    def match_detections(self, dets, gt_bboxes):
        gt_ids = gt_bboxes[:, 5]
        gt_bboxes = gt_bboxes[:, :4]
        ids = torch.zeros(dets.shape[0], dtype=torch.long, device=dets.device)
        if dets.numel() > 0 and gt_bboxes.numel() > 0:
            ious = bbox_overlaps(dets, gt_bboxes)
            mxs, inds = ious.max(dim=1)
            mask = mxs > self.iou_thr
            ids[mask] = gt_ids[inds[mask]].long()
        return ids

    def get_assoc_loss(self, input):
        n = len(input['affinities'])
        assoc_pred = torch.cat([aff.flatten() for aff in input['affinities']])
        masks = []
        targets = []
        with torch.no_grad():
            for i in range(n):
                a_dets = input['dt_bboxes'][i]
                a_gt_bboxes = input['gt_bboxes'][i]
                b_dets = input['refs']['data'][-1]['dt_bboxes'][i]
                b_gt_bboxes = input['refs']['original'][-1]['gt_bboxes'][i]
                a_ids = self.match_detections(a_dets, a_gt_bboxes)
                b_ids = self.match_detections(b_dets, b_gt_bboxes)
                target = (a_ids.reshape(-1, 1) == b_ids.reshape(1, -1)).float()
                mask = (a_ids > 0).reshape(-1, 1) | (b_ids > 0).reshape(1, -1)
                masks.append(mask)
                targets.append(target)
        masks = torch.cat([mask.flatten() for mask in masks])
        assoc_targets = torch.cat([target.flatten() for target in targets])
        pos_masks = masks & (assoc_targets > 0.99)
        num_all = masks.sum().item()
        num_all = max(num_all, 1)
        num_fgs = pos_masks.sum().item()
        num_fgs = max(num_fgs, 1)
        full_assoc_loss = self.assoc_loss(assoc_pred[masks], assoc_targets[masks], normalizer_override=num_all)
        pos_assoc_oss = self.assoc_loss(assoc_pred[pos_masks], assoc_targets[pos_masks], normalizer_override=num_fgs)
        losses = {
            self.prefix + '.assoc_loss': full_assoc_loss + pos_assoc_oss,
        }
        return losses

    def forward(self, input):
        noaug_flag = input['noaug_flag']
        if 'main' in input and 'ref' in input:
            input = input['main']

        mlvl_preds, mlvl_locations, mlvl_ori_loc_preds = self.prepare_preds(input)

        if self.training:
            targets = self.supervisor.get_targets(mlvl_locations, input, mlvl_preds)
            if not self.use_l1:
                losses = self.get_loss(targets, mlvl_preds, noaug_flag=noaug_flag)
            else:
                losses = self.get_loss(targets, mlvl_preds, mlvl_ori_loc_preds, noaug_flag=noaug_flag)
            losses_assoc = self.get_assoc_loss(input)
            losses.update(losses_assoc)
            return losses
        else:
            detections = [torch.cat([dets.new_full((dets.shape[0], 1), ix), dets], dim=1)
                          for ix, dets in enumerate(input['dt_bboxes'])]
            results = {
                'dt_bboxes': torch.cat(detections, dim=0),
                'id_embeds': torch.cat(input['id_embeds'], dim=0),
            }
            return results


@MODULE_ZOO_REGISTRY.register('yolox_post_w_id_n_orient')
class YoloxwIDnOrientPostProcess(YoloxwIDPostProcess):

    def __init__(self,
                 num_classes,
                 num_ids,
                 inplanes,
                 cfg,
                 num_orient_class, **kwargs):
        super().__init__(num_classes,
                         num_ids,
                         inplanes,
                         cfg, **kwargs)
        self.num_orient_class = num_orient_class
        self.orient_cls_loss = build_loss(cfg['orient_cls_loss'])
        self.orient_reg_loss = build_loss(cfg['orient_reg_loss'])

    def apply_activation(self, mlvl_preds):
        """apply activation on permuted class prediction
        Arguments:
            - mlvl_pred (list (levels) of tuple (predictions) of Tensor): first element must
            be permuted class prediction with last dimension be class dimension
        """
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = preds[0].sigmoid()
            obj_pred = preds[3].sigmoid()
            orient_pred = preds[4].sigmoid()
            cls_pred *= obj_pred
            mlvl_activated_preds.append((cls_pred, *preds[1:4], orient_pred, preds[5]))
        return mlvl_activated_preds

    def get_loss(self, targets, mlvl_preds, mlvl_ori_loc_preds=None, noaug_flag=None):
        mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred, mlvl_obj_pred, mlvl_orient_cls_pred, mlvl_orient_reg_pred = zip(
            *mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        id_pred = torch.cat(mlvl_id_pred, dim=1)
        obj_pred = torch.cat(mlvl_obj_pred, dim=1)
        ocls_pred = torch.cat(mlvl_orient_cls_pred, dim=1)
        oreg_pred = torch.cat(mlvl_orient_reg_pred, dim=1)
        if self.use_l1:
            ori_loc_preds = torch.cat(mlvl_ori_loc_preds, dim=1)
            del mlvl_ori_loc_preds
        del mlvl_cls_pred, mlvl_loc_pred, mlvl_id_pred, mlvl_obj_pred

        cls_targets, reg_targets, id_targets, obj_targets, orient_cls_targets, orient_cls_index, orient_reg_targets, ori_reg_targets, fg_masks, num_fgs, valid_id_masks = targets
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        id_targets = torch.cat(id_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        ocls_targets = torch.cat(orient_cls_targets, 0)
        ocls_index = torch.cat(orient_cls_index, 0)
        oreg_targets = torch.cat(orient_reg_targets, 0)
        if noaug_flag is not None and self.dismiss_aug:
            valid_id_masks = [(m & o) for m, o in zip(valid_id_masks, noaug_flag)]
        valid_id_masks = torch.cat(valid_id_masks, 0)
        if self.use_l1:
            ori_reg_targets = torch.cat(ori_reg_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.all_reduce_norm and dist.is_initialized():
            num_fgs = self.get_ave_normalizer(fg_masks)
            num_ids = self.get_ave_normalizer(valid_id_masks)
        else:
            num_fgs = fg_masks.sum().item()
            num_ids = valid_id_masks.sum().item()
        num_fgs = max(num_fgs, 1)
        num_ids = max(num_ids, 1)

        cls_pred = cls_pred.reshape(-1, self.num_classes - 1)
        cls_loss = self.cls_loss(cls_pred[fg_masks], cls_targets, normalizer_override=num_fgs)
        # print(cls_pred[fg_masks].shape, cls_targets.shape)
        id_pred = id_pred.reshape(-1, self.num_ids)
        id_loss = self.id_loss(id_pred[fg_masks][valid_id_masks],
                               id_targets[valid_id_masks][:, 1:], normalizer_override=num_ids)
        orient_pred = ocls_pred.reshape(-1, self.num_orient_class)
        orient_reg = oreg_pred.reshape(-1, self.num_orient_class)
        orient_cls_loss = self.orient_cls_loss(orient_pred[fg_masks][valid_id_masks],
                                               ocls_targets[valid_id_masks], normalizer_override=num_ids)
        orient_reg_loss = self.orient_reg_loss(torch.gather(orient_reg[fg_masks][valid_id_masks], 1, ocls_index[valid_id_masks].unsqueeze(1)).flatten(),
                                               oreg_targets[valid_id_masks], normalizer_override=num_ids)
        # print(id_pred[fg_masks].shape, id_targets.shape)
        acc = self.get_acc(cls_pred[fg_masks], cls_targets)
        acc_id = self.get_acc(id_pred[fg_masks][valid_id_masks], id_targets[valid_id_masks][:, 1:])
        acc_orient = self.get_acc(orient_pred[fg_masks][valid_id_masks], ocls_targets[valid_id_masks])

        loc_target = reg_targets.reshape(-1, 4)
        loc_pred = loc_pred.reshape(-1, 4)
        # loc_preds [xc, yc, w, h] -> [x1, y1, x2, y2]
        loc_pred_x1 = loc_pred[:, 0] - loc_pred[:, 2] / 2
        loc_pred_y1 = loc_pred[:, 1] - loc_pred[:, 3] / 2
        loc_pred_x2 = loc_pred[:, 0] + loc_pred[:, 2] / 2
        loc_pred_y2 = loc_pred[:, 1] + loc_pred[:, 3] / 2
        loc_pred = torch.stack([loc_pred_x1, loc_pred_y1, loc_pred_x2, loc_pred_y2], dim=-1)
        obj_pred = obj_pred.reshape(-1)
        if loc_pred[fg_masks].numel() > 0:
            loc_loss = self.loc_loss(loc_pred[fg_masks], loc_target, normalizer_override=num_fgs)
        else:
            loc_loss = loc_pred[fg_masks].sum()
        obj_loss = self.obj_loss(obj_pred, obj_targets, normalizer_override=num_fgs)
        # add l1 loss
        if self.use_l1:
            if ori_loc_preds.numel() > 0:
                l1_loss = (self.l1_loss(ori_loc_preds.view(-1, 4)[fg_masks], ori_reg_targets)).sum() / num_fgs
            else:
                l1_loss = ori_loc_preds.sum()
        else:
            l1_loss = torch.tensor(0.0).cuda()
        lw_id = self.lw_id
        lw_det = self.lw_det
        if self.balanced_loss_weight == 'auto':
            lw_id = torch.exp(-lw_id)
            lw_det = torch.exp(-lw_det)
            return {
                self.prefix + '.cls_loss': cls_loss * lw_det,
                self.prefix + '.loc_loss': loc_loss * lw_det,
                self.prefix + '.id_loss': id_loss * lw_id,
                self.prefix + '.orient_cls_loss': orient_cls_loss,
                self.prefix + '.orient_reg_loss': orient_reg_loss,
                self.prefix + '.obj_loss': obj_loss * lw_det,
                self.prefix + '.l1_loss': l1_loss * lw_det,
                self.prefix + '.lwnorm_loss': (self.lw_id + self.lw_det) * self.balance_scale,
                self.prefix + '.accuracy': acc,
                self.prefix + '.accuracy_id': acc_id,
                self.prefix + '.accuracy_orient': acc_orient,
            }
        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.id_loss': id_loss,
            self.prefix + '.orient_cls_loss': orient_cls_loss,
            self.prefix + '.orient_reg_loss': orient_reg_loss,
            self.prefix + '.obj_loss': obj_loss,
            self.prefix + '.l1_loss': l1_loss,
            self.prefix + '.accuracy': acc,
            self.prefix + '.accuracy_id': acc_id,
            self.prefix + '.accuracy_orient': acc_orient,
        }
