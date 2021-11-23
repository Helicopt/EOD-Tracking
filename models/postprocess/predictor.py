# Import from third library
import torch
from eod.models.heads.utils.nms_wrapper import nms
from eod.utils.general.registry_factory import ROI_PREDICTOR_REGISTRY
from ...utils.debug import info_debug, get_debugger

__all__ = ['YoloXwIDPredictor']


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
