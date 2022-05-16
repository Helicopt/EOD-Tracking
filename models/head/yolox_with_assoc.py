import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from eod.utils.model import accuracy as A  # noqa F401
from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

from eod.tasks.det.plugins.yolox.models.backbone.cspdarknet import DWConv

from eod.tasks.det.plugins.yolov5.models.components import ConvBnAct
from eod.utils.model.normalize import build_norm_layer
from eod.utils.model.act_fn import build_act_fn
from eod.tasks.det.models.utils.anchor_generator import build_anchor_generator

from eod.tasks.det.models.utils.nms_wrapper import nms
from eod.tasks.det.models.utils.bbox_helper import (
    clip_bbox,
    filter_by_size,
    offset2bbox
)

from eod.utils.model import accuracy as A  # noqa F401
from eod.tasks.det.models.utils.bbox_helper import bbox_iou_overlaps as bbox_overlaps

from ...utils.debug import info_debug, logger_print

__all__ = ['YoloXAssocHead']


@MODULE_ZOO_REGISTRY.register('yolox_assoc')
class YoloXAssocHead(nn.Module):

    def __init__(self,
                 anchor_generator, inplanes, pre_nms_score_thresh, nms, num_classes,
                 framerate_aware=True, auto_framerate=False, control='min', feature_type='high', norm_on_highlvl=True, norm_on_bbox=False,
                 pred_framerate=False, return_extra=False):
        super(YoloXAssocHead, self).__init__()
        self.inplanes = inplanes
        self.point_generator = build_anchor_generator(anchor_generator)
        self.dense_points = self.point_generator.dense_points
        self.pre_nms_score_thresh = pre_nms_score_thresh
        self.nms_cfg = nms
        self.top_n = 300
        self.sin_div = 256
        self.num_classes = num_classes - 1
        self.norm_on_bbox = norm_on_bbox
        self.feature_type = feature_type
        self.norm_on_highlvl = norm_on_highlvl
        self.pred_framerate = pred_framerate and auto_framerate
        self.return_extra = return_extra
        assert feature_type in ['high', 'mid', 'low']
        if self.feature_type == 'high':
            if self.norm_on_highlvl:
                self.aff_net = nn.Sequential(
                    nn.BatchNorm1d(2),
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 64),
                )
            else:
                self.aff_net = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                )

        if self.feature_type == 'mid':
            if self.norm_on_highlvl:
                self.aff_net = nn.Sequential(
                    nn.BatchNorm1d(4),
                    nn.Linear(4, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 64),
                )
            else:
                self.aff_net = nn.Sequential(
                    nn.Linear(4, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                )

        if self.feature_type == 'low':
            inplane = self.inplanes[0]
            if self.norm_on_highlvl:
                self.aff_net_0 = nn.Sequential(
                    nn.BatchNorm1d(4),
                    nn.Linear(4, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 64),
                )
            else:
                self.aff_net_0 = nn.Sequential(
                    nn.Linear(4, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                )
            self.aff_net_1 = nn.Sequential(
                nn.Linear(inplane, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 64),
            )
        if self.pred_framerate:
            self.pfr_net = nn.Sequential(
                nn.Linear(self.top_n, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Linear(64, 1),
            )

        self.framerate_aware = framerate_aware
        self.auto_framerate = auto_framerate
        if self.framerate_aware:
            self.diff_norm = nn.LayerNorm(self.top_n)
            if not self.auto_framerate:
                self.att_net = nn.Sequential(
                    nn.Linear(self.sin_div + self.top_n, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                )
            else:
                self.att_net = nn.Sequential(
                    nn.Linear(self.top_n, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                )
        self.control = control
        assert control in ['min', 'sim']

    def get_outplanes(self):
        return self.inplanes

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

        for l_ix in range(len(mlvl_locations)):
            # add preds for l1 loss
            mlvl_preds[l_ix][1][..., :2] *= strides[l_ix]
            mlvl_preds[l_ix][1][..., :2] += mlvl_locations[l_ix]
            if self.norm_on_bbox:
                mlvl_preds[l_ix][1][..., 2:4] = F.relu(mlvl_preds[l_ix][1][..., 2:4])  # , inplace=True)
                mlvl_preds[l_ix][1][..., 2:4] *= strides[l_ix]
            else:
                mlvl_preds[l_ix][1][..., 2:4] = torch.exp(mlvl_preds[l_ix][1][..., 2:4]) * strides[l_ix]
        return mlvl_preds, mlvl_locations

    def sin_embedding(self, frame_rate, device='cuda', dtype=torch.float32):
        framerates_emb = torch.arange(self.sin_div, device=device, dtype=dtype).reshape(
            1, self.sin_div)
        framerates_emb = torch.cos(framerates_emb * frame_rate * 1.5 / self.sin_div)
        return framerates_emb

    def get_affinity_matrix(self, main, ref, frame_rates=None):
        if not self.training and self.norm_on_highlvl:
            if self.feature_type != 'low':
                aff_net = self.aff_net
            else:
                aff_net = self.aff_net_0
            for mod in aff_net:
                if mod.__class__.__name__ == 'BatchNorm1d':
                    mod.train()
        # logger_print(frame_rates)
        ret = []
        pred_frs = []
        extras = []
        for b_ix, (det_boxes, id_embeddings) in enumerate(zip(main['dt_bboxes'], main['id_embeds'])):
            if det_boxes is None or id_embeddings is None:
                ret.append(None)
                pred_frs.append(0.)
                extras.append(None)
                continue
            det_boxes = det_boxes.detach()
            if frame_rates is not None:
                frame_rate = frame_rates[b_ix]
            else:
                frame_rate = 1.0
            if ref['data'][-1]['dt_bboxes'][b_ix] is None or ref['data'][-1]['id_embeds'][b_ix] is None:
                ret.append(None)
                pred_frs.append(0.)
                extras.append(None)
                continue
            ref_boxes = ref['data'][-1]['dt_bboxes'][b_ix].detach()
            ref_embeds = ref['data'][-1]['id_embeds'][b_ix]
            extended_id_embeds = id_embeddings.unsqueeze(1).repeat(1, ref_boxes.shape[0], 1)
            extedned_ref_embeds = ref_embeds.unsqueeze(0).repeat(id_embeddings.shape[0], 1, 1)
            cos_sim = F.cosine_similarity(extended_id_embeds, extedned_ref_embeds, dim=2)
            extended_boxes = det_boxes.unsqueeze(1).repeat(1, ref_boxes.shape[0], 1)
            extended_ref_boxes = ref_boxes.unsqueeze(0).repeat(det_boxes.shape[0], 1, 1)
            loc_sim = (extended_boxes[..., :2] + extended_boxes[..., 2:4] - extended_ref_boxes[...,
                       :2] - extended_ref_boxes[..., 2:4]).div(2).pow(2).sum(dim=2).sqrt()
            extended_shp = ((extended_boxes[..., 2:4] - extended_boxes[..., :2]).prod(dim=-1)).sqrt()
            extended_ref_shp = ((extended_ref_boxes[..., 2:4] - extended_ref_boxes[..., :2]).prod(dim=-1)).sqrt()
            norm_loc_sim = loc_sim / (torch.min(extended_shp, extended_ref_shp) + 1e-6)
            if self.feature_type == 'high':
                aff_feats = torch.stack([cos_sim, norm_loc_sim], dim=2)
            if self.feature_type == 'mid' or self.feature_type == 'low':
                extended_wh = extended_boxes[..., 2:4] - extended_boxes[..., :2]
                extended_ref_wh = extended_ref_boxes[..., 2:4] - extended_ref_boxes[..., :2]
                scale = extended_wh[..., 1] * extended_ref_wh[..., 0]
                ref_scale = extended_ref_wh[..., 1] * extended_wh[..., 0]
                scale_sim = torch.log((scale + 1e-6) / (ref_scale + 1e-6)).abs()
                shp_sim = torch.log((extended_shp + 1e-6) / (extended_ref_shp + 1e-6)).abs()
                aff_feats = torch.stack([cos_sim, norm_loc_sim, scale_sim, shp_sim], dim=2)
            if self.feature_type == 'low':
                aff_feats_low = extended_id_embeds * extedned_ref_embeds
            if self.framerate_aware:
                if not self.auto_framerate:
                    frame_rate_embeddings = self.sin_embedding(
                        frame_rate, device=aff_feats.device, dtype=aff_feats.dtype)
                if self.control == 'min':
                    diff_embeddings, _ = norm_loc_sim.min(dim=1)
                else:
                    _, max_inds = cos_sim.max(dim=1)
                    diff_embeddings = torch.gather(norm_loc_sim, 1, max_inds.unsqueeze(1)).squeeze(-1)
                extra_ = {'ctrl_emb': diff_embeddings, 'original_cos_sim': (cos_sim, norm_loc_sim)}
                diff_embeddings = self.diff_norm(F.adaptive_avg_pool1d(
                    diff_embeddings.reshape(1, 1, -1), self.top_n).reshape(1, -1))
                extras.append(extra_)
                if not self.auto_framerate:
                    control_embeddings = torch.cat([frame_rate_embeddings, diff_embeddings], dim=1)
                else:
                    control_embeddings = diff_embeddings
            n, m, c = aff_feats.shape
            if self.feature_type == 'high' or self.feature_type == 'mid':
                aff_feats = self.aff_net(aff_feats.reshape(-1, c))
            if self.feature_type == 'low':
                c_low = aff_feats_low.shape[-1]
                aff_feats = self.aff_net_0(aff_feats.reshape(-1, c)) + self.aff_net_1(aff_feats_low.reshape(-1, c_low))
            if self.framerate_aware:
                aff_att = self.att_net(control_embeddings).softmax(dim=1)
                affinity_matrix = (aff_feats * aff_att).sum(dim=1).reshape(n, m)
            else:
                affinity_matrix = aff_feats.mean(dim=1).reshape(n, m)
            # info_debug(det_boxes)
            # info_debug(id_embeddings)
            # info_debug(ref_boxes)
            # info_debug(ref_embeds)
            # info_debug(cos_sim, statistics=True)
            # info_debug(norm_loc_sim, statistics=True)
            ret.append(affinity_matrix)
            if self.pred_framerate:
                pred_fr = self.pfr_net(control_embeddings)
                pred_frs.append(pred_fr)
        # info_debug(ret, statistics=True)
        return_tuples = [ret]
        if self.pred_framerate:
            return_tuples.append(torch.cat(pred_frs, dim=0) if self.training else pred_frs)
        if self.return_extra:
            return_tuples.append(extras)
        return return_tuples

    def forward(self, input):
        frame_rates = input.get('framerate', None)
        if 'main' in input and 'ref' in input:
            ref = input['ref']
            input = input['main']
        else:
            ref = None

        mlvl_preds, mlvl_locations = self.prepare_preds(input)
        id_feats = [lvl_feats[2] for lvl_feats in input['roi_features']]
        mlvl_preds = self.apply_activation(mlvl_preds)
        results = self.predict(mlvl_preds, id_feats)
        if ref is not None and isinstance(ref, list):
            refs = []
            for ref_i in ref:
                mlvl_preds, mlvl_locations = self.prepare_preds(ref_i)
                id_feats = [lvl_feats[2] for lvl_feats in ref_i['roi_features']]
                mlvl_preds = self.apply_activation(mlvl_preds)
                ref_i = self.predict(mlvl_preds, id_feats)
                refs.append(ref_i)
            ref = {'data': refs, 'original': ref}

        if ref is not None and 'data' in ref:
            if self.pred_framerate:
                ret_tupples = self.get_affinity_matrix(results, ref)
                aff_matrix, pred_frs = ret_tupples[0], ret_tupples[1]
                results['log2framerates'] = pred_frs
                if self.return_extra:
                    results['extra'] = ret_tupples[2]
            else:
                ret_tupples = self.get_affinity_matrix(results, ref, frame_rates=frame_rates)
                aff_matrix = ret_tupples[0]
                if self.return_extra:
                    results['extra'] = ret_tupples[1]
            results['affinities'] = aff_matrix
            results['refs'] = ref

        results.update(input)
        # info_debug(results, statistics=True)
        return results

    def predict(self, preds, id_features):
        # id_features = self.fuse_lvl_features(id_features)
        id_features = [id_feat.permute(0, 2, 3, 1).reshape(id_feat.size(
            0), id_feat.size(2) * id_feat.size(3), -1) for id_feat in id_features]

        preds = [(p[0], p[1], p[3]) for p in preds]
        max_wh = 4096
        preds = [torch.cat(p, dim=2) for p in preds]
        preds = torch.cat(preds, dim=1)
        id_features = torch.cat(id_features, dim=1)
        x1 = preds[..., self.num_classes] - preds[..., self.num_classes + 2] / 2
        y1 = preds[..., self.num_classes + 1] - preds[..., self.num_classes + 3] / 2
        x2 = preds[..., self.num_classes] + preds[..., self.num_classes + 2] / 2
        y2 = preds[..., self.num_classes + 1] + preds[..., self.num_classes + 3] / 2
        preds[..., self.num_classes] = x1
        preds[..., self.num_classes + 1] = y1
        preds[..., self.num_classes + 2] = x2
        preds[..., self.num_classes + 3] = y2

        B = preds.shape[0]
        det_results = []
        id_feats_all = []

        # debugger = get_debugger()
        for b_ix in range(B):
            pred_per_img = preds[b_ix]
            class_conf, class_pred = torch.max(pred_per_img[:, :self.num_classes], 1, keepdim=True)
            id_feats_per_img = id_features[b_ix]

            # debugger(class_conf, 'conf')
            conf_mask = (class_conf.squeeze() >= self.pre_nms_score_thresh).squeeze()
            detections = torch.cat((pred_per_img[:, self.num_classes:-1], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            id_feats = id_feats_per_img[conf_mask]

            if not detections.size(0):
                det_results.append(None)
                id_feats_all.append(None)
                continue

            # batch nms
            cls_hash = detections[:, -1].unsqueeze(-1) * max_wh

            boxes = detections[:, :4] + cls_hash
            scores = detections[:, 4:5]  # .unsqueeze(-1)
            res, keep = nms(torch.cat([boxes, scores], 1), self.nms_cfg)
            # debugger(keep, 'predictor_keep')

            rois_keep = detections[keep]
            id_feats_keep = id_feats[keep]

            rois_keep[:, -1] = rois_keep[:, -1] + 1

            # If none remain process next image
            n = rois_keep.shape[0]  # number of boxes
            if not n:
                det_results.append(None)
                id_feats_all.append(None)
                continue

            if n > self.top_n:
                rois_keep = rois_keep[:self.top_n]
                id_feats_keep = id_feats_keep[:self.top_n]

            det_results.append(rois_keep)
            id_feats_all.append(id_feats_keep)

        # if len(det_results) == 0:
        #     det_results.append(preds.new_zeros((1, 6)))
        #     id_feats_all.append(preds.new_zeros((1, 111)))

        return {'dt_bboxes': det_results, 'id_embeds': id_feats_all}
