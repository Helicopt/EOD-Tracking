# Import from third library
import torch
from eod.models.heads.utils.nms_wrapper import nms
from eod.models.heads.utils.bbox_helper import (
    clip_bbox,
    filter_by_size,
    offset2bbox
)
from eod.utils.general.registry_factory import ROI_MERGER_REGISTRY, ROI_PREDICTOR_REGISTRY
from eod.models.postprocess.predictor import build_merger
from eod.utils.general.fp16_helper import to_float32
from ...utils.debug import info_debug, get_debugger

__all__ = ['RoiwIDPredictor', 'YoloXwIDPredictor', 'RetinawIDMerger']


@ROI_PREDICTOR_REGISTRY.register('base_w_id')
class RoiwIDPredictor(object):
    """Predictor for the first stage
    """

    def __init__(self,
                 pre_nms_score_thresh,
                 pre_nms_top_n,
                 post_nms_top_n,
                 roi_min_size,
                 merger=None,
                 nms=None,
                 clip_box=True):
        self.pre_nms_score_thresh = pre_nms_score_thresh
        # self.apply_score_thresh_above = apply_score_thresh_above
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_cfg = nms
        self.roi_min_size = roi_min_size
        self.clip_box = clip_box
        if merger is not None:
            self.merger = build_merger(merger)
        else:
            self.merger = None

    @torch.no_grad()
    @to_float32
    def predict(self, mlvl_anchors, mlvl_preds, id_features, image_info):
        mlvl_resutls = []
        mlvl_embeddings = []
        for anchors, preds, id_feature in zip(mlvl_anchors, mlvl_preds, id_features):
            results, embeddings = self.single_level_predict(anchors, preds, id_feature, image_info)
            mlvl_resutls.append(results)
            mlvl_embeddings.append(embeddings)
        if len(mlvl_resutls) > 0:
            results = torch.cat(mlvl_resutls, dim=0)
            embeddings = torch.cat(mlvl_embeddings, dim=0)
        else:
            results = mlvl_anchors[0].new_zeros((1, 7))
            embeddings = mlvl_anchors[0].new_zeros((1, 256))
        if self.merger is not None:
            results, embeddings = self.merger.merge(results, embeddings)
        return {'dt_bboxes': results, 'id_embeds': embeddings}

    def regression(self, anchors, preds, image_info):
        cls_pred, loc_pred = preds[:2]
        B, K = cls_pred.shape[:2]
        if anchors.dim() == 2:
            concat_anchors = torch.stack([anchors.clone() for _ in range(B)])
        else:
            concat_anchors = anchors.clone()
        rois = offset2bbox(concat_anchors.view(B * K, 4), loc_pred.view(B * K, 4)).view(B, K, 4)  # noqa
        return rois

    def single_level_predict(self, anchors, preds, id_features, image_info):
        """
        Arguments:
            - anchors (FloatTensor, fp32): [K, 4]
            - preds[0] (cls_pred, FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
            - preds[1] (loc_pred, FloatTensor, fp32): [B, K, 4]
            - image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

        Returns:
            rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
        """

        rois = self.regression(anchors, preds, image_info)
        cls_pred = preds[0]
        B, K, C = cls_pred.shape
        roi_min_size = self.roi_min_size
        pre_nms_top_n = self.pre_nms_top_n
        pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
        post_nms_top_n = self.post_nms_top_n
        # if featmap size is too large, filter by score thresh to reduce computation
        if K > 120:
            score_thresh = self.pre_nms_score_thresh
        else:
            score_thresh = 0

        batch_rois = []
        batch_id_feats = []
        for b_ix in range(B):
            # clip rois and filter rois which are too small
            image_rois = rois[b_ix]
            if self.clip_box:
                image_rois = clip_bbox(image_rois, image_info[b_ix])
            image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
            image_cls_pred = cls_pred[b_ix][filter_inds]
            id_feats = id_features[b_ix][filter_inds]
            if image_rois.numel() == 0:
                continue  # noqa E701

            for cls in range(C):
                cls_rois = image_rois
                scores = image_cls_pred[:, cls]
                id_feats_cls = id_feats
                assert not torch.isnan(scores).any()
                if score_thresh > 0:
                    # to reduce computation
                    keep_idx = torch.nonzero(scores > score_thresh).reshape(-1)
                    if keep_idx.numel() == 0:
                        continue  # noqa E701
                    cls_rois = cls_rois[keep_idx]
                    scores = scores[keep_idx]
                    id_feats_cls = id_feats_cls[keep_idx]

                # do nms per image, only one class
                _pre_nms_top_n = min(pre_nms_top_n, scores.shape[0])
                scores, order = scores.topk(_pre_nms_top_n, sorted=True)
                cls_rois = cls_rois[order, :]
                id_feats_cls = id_feats_cls[order, :]
                cls_rois = torch.cat([cls_rois, scores[:, None]], dim=1)

                if self.nms_cfg is not None:
                    cls_rois, keep_idx = nms(cls_rois, self.nms_cfg)
                    id_feats_cls = id_feats_cls[keep_idx]
                if post_nms_top_n > 0:
                    cls_rois = cls_rois[:post_nms_top_n]
                    id_feats_cls = id_feats_cls[:post_nms_top_n]

                ix = cls_rois.new_full((cls_rois.shape[0], 1), b_ix)
                c = cls_rois.new_full((cls_rois.shape[0], 1), cls + 1)
                cls_rois = torch.cat([ix, cls_rois, c], dim=1)
                batch_rois.append(cls_rois)
                batch_id_feats.append(id_feats_cls)

        if len(batch_rois) == 0:
            return anchors.new_zeros((1, 7)), anchors.new_zeros((1, 256))
        return torch.cat(batch_rois, dim=0), torch.cat(batch_id_feats, dim=0)

    def lvl_nms(self, anchors, preds, image_info, preserved=800):
        """
        Arguments:
            - anchors (FloatTensor, fp32): [K, 4]
            - preds[0] (cls_pred, FloatTensor, fp32): [B, K, C], C[i] -> class i+1, background class is excluded
            - preds[1] (loc_pred, FloatTensor, fp32): [B, K, 4]
            - image_info (list of FloatTensor): [B, >=2] (image_h, image_w, ...)

        Returns:
            rois (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
        """
        # preds = (preds[0], preds[1], preds[3])
        rois = self.regression(anchors, preds, image_info)
        cls_pred = preds[0]
        B, K, C = cls_pred.shape
        roi_min_size = self.roi_min_size
        pre_nms_top_n = self.pre_nms_top_n
        pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
        post_nms_top_n = self.post_nms_top_n
        # if featmap size is too large, filter by score thresh to reduce computation
        if K > 120:
            score_thresh = self.pre_nms_score_thresh
        else:
            score_thresh = 0

        batch_keep = []
        for b_ix in range(B):
            # clip rois and filter rois which are too small
            image_rois = rois[b_ix]
            if self.clip_box:
                image_rois = clip_bbox(image_rois, image_info[b_ix])
            image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
            keep_inds = filter_inds.nonzero().flatten()
            image_cls_pred = cls_pred[b_ix][filter_inds]
            if image_rois.numel() == 0:
                for cls in range(C):
                    batch_keep.append(preds[0].new_zeros((0, ), dtype=torch.int64))
                continue  # noqa E701

            for cls in range(C):
                cls_rois = image_rois
                scores = image_cls_pred[:, cls]
                cls_keep = keep_inds
                assert not torch.isnan(scores).any()
                if score_thresh > 0:
                    # to reduce computation
                    keep_idx = torch.nonzero(scores > score_thresh).reshape(-1)
                    if keep_idx.numel() == 0:
                        batch_keep.append(preds[0].new_zeros((0, ), dtype=torch.int64))
                        continue  # noqa E701
                    cls_rois = cls_rois[keep_idx]
                    scores = scores[keep_idx]
                    cls_keep = cls_keep[keep_idx]

                # do nms per image, only one class
                _pre_nms_top_n = min(pre_nms_top_n, scores.shape[0])
                scores, order = scores.topk(_pre_nms_top_n, sorted=True)
                cls_rois = cls_rois[order, :]
                cls_keep = cls_keep[order]
                cls_rois = torch.cat([cls_rois, scores[:, None]], dim=1)

                if self.nms_cfg is not None:
                    cls_rois, keep_idx = nms(cls_rois, self.nms_cfg)
                    cls_keep = cls_keep[keep_idx]
                if post_nms_top_n > 0:
                    cls_rois = cls_rois[:post_nms_top_n]
                    cls_keep = cls_keep[:post_nms_top_n]

                batch_keep.append(cls_keep)
        inds = []
        # info_debug(batch_keep, prefix='C==%d,B==%d' % (C, B))
        for b_ix in range(B):
            bkeep = batch_keep[(b_ix * C): (b_ix + 1) * C]
            # print(bkeep)
            ind = torch.cat(bkeep)
            rest = min(preserved - len(ind), preds[0].size(1))
            if rest > 0:
                ind = torch.cat([ind, torch.arange(rest).to(ind.device)])
            if rest < 0:
                ind = ind[:preserved]
            inds.append(ind)
        if len(inds) == 0:
            return preds[0].new_zeros((0, preserved), dtype=torch.int64)
        else:
            return torch.stack(inds)


@ROI_MERGER_REGISTRY.register('retina_w_id')
class RetinawIDMerger(object):
    """Merge results from multi-levels
    1. concat all results
    2. nms
    3. keep topk
    """

    def __init__(self, top_n, nms):
        self.top_n = top_n
        self.nms_cfg = nms

    @torch.no_grad()
    @to_float32
    def merge(self, mlvl_rois, mlvl_id_feats):
        """
        Merge rois from different levels together

        Note:
            1. do nms for each class when nms_iou_thresh > 0
            2. keep top_n rois for each image, keep all when top_n <= 0

        Arguments:
            - mlvl_rois (FloatTensor): [N, >=7], (batch_index, x1, y1, x2, y2, score, cls)
        """
        merged_rois = []
        merged_embeds = []
        B = int(torch.max(mlvl_rois[:, 0]).item() + 1)
        for b_ix in range(B):
            img_rois = mlvl_rois[mlvl_rois[:, 0] == b_ix]
            img_embeds = mlvl_id_feats[mlvl_rois[:, 0] == b_ix]
            if img_rois.numel() == 0:
                continue  # noqa E701

            classes = torch.unique(img_rois[:, 6].int()).cpu().numpy().tolist()
            all_cls_rois = []
            all_embeds = []
            for cls in classes:
                cls_rois = img_rois[img_rois[:, 6] == cls]
                cls_embeds = img_embeds[img_rois[:, 6] == cls]
                if cls_rois.numel() == 0:
                    continue  # noqa E701
                _, indices = nms(cls_rois[:, 1:6], self.nms_cfg)
                all_cls_rois.append(cls_rois[indices])
                all_embeds.append(cls_embeds[indices])
            if len(all_cls_rois) == 0:
                continue  # noqa E701
            rois = torch.cat(all_cls_rois, dim=0)
            embeds = torch.cat(all_embeds, dim=0)

            if self.top_n < rois.shape[0]:
                _, inds = torch.topk(rois[:, 5], self.top_n)
                rois = rois[inds]
                embeds = embeds[inds]
            merged_rois.append(rois)
            merged_embeds.append(embeds)
        if len(merged_rois) > 0:
            merged_rois = torch.cat(merged_rois, dim=0)
            merged_embeds = torch.cat(merged_embeds, dim=0)
        else:
            merged_rois = mlvl_rois.new_zeros((1, 7))
            merged_embeds = mlvl_rois.new_zeros((1, 256))
        return merged_rois, merged_embeds


@ROI_PREDICTOR_REGISTRY.register('yolox_w_id')
class YoloXwIDPredictor(object):
    def __init__(self, num_classes, pre_nms_score_thresh, nms):
        self.pre_nms_score_thresh = pre_nms_score_thresh
        self.nms_cfg = nms
        self.top_n = 300
        self.num_classes = num_classes - 1

    @torch.no_grad()
    def predict(self, preds, id_features):
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
                continue

            if n > self.top_n:
                rois_keep = rois_keep[:self.top_n]
                id_feats_keep = id_feats_keep[:self.top_n]

            det_results.append(torch.cat([rois_keep.new_full((rois_keep.shape[0], 1), b_ix), rois_keep], dim=1))
            id_feats_all.append(id_feats_keep)
        if len(det_results) == 0:
            det_results.append(preds.new_zeros((1, 7)))
            id_feats_all.append(preds.new_zeros((1, 256)))
        bboxes = torch.cat(det_results, dim=0)
        id_embeds = torch.cat(id_feats_all, dim=0)
        return {'dt_bboxes': bboxes, 'id_embeds': id_embeds}

    def lvl_nms(self, preds, preserved=800):
        preds = (preds[0], preds[1], preds[3])
        max_wh = 4096
        # debugger = get_debugger()
        # debugger(preds)
        preds = torch.cat(preds, dim=2)
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
        for b_ix in range(B):
            pred_per_img = preds[b_ix]
            class_conf, class_pred = torch.max(pred_per_img[:, :self.num_classes], 1, keepdim=True)
            conf_mask = (class_conf.squeeze() >= self.pre_nms_score_thresh).squeeze()
            conf_inds = conf_mask.nonzero().flatten()
            detections = torch.cat((pred_per_img[:, self.num_classes:-1], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                det_results.append(preds.new_zeros((0, ), dtype=torch.int64))
                continue

            # batch nms
            cls_hash = detections[:, -1].unsqueeze(-1) * max_wh

            boxes = detections[:, :4] + cls_hash
            scores = detections[:, 4:5]  # .unsqueeze(-1)
            res, keep = nms(torch.cat([boxes, scores], 1), self.nms_cfg)
            # debugger(keep, 'lvl_keep')
            det_results.append(conf_inds[keep])

        inds = []
        for b_ix in range(B):
            ind = det_results[b_ix]
            rest = min(preserved - len(ind), preds.size(1))
            # print(ind.shape, rest)
            if rest > 0:
                ind = torch.cat([ind, torch.arange(rest).to(ind.device)])
            if rest < 0:
                ind = ind[:preserved]
            inds.append(ind)
        inds = torch.stack(inds)
        return inds
        # pred = torch.gather(preds, 1, inds)
