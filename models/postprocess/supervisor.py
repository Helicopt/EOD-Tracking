# Import from third library
import torch
import torch.nn.functional as F
from eod.tasks.det.models.utils.matcher import build_matcher
from eod.utils.general.registry_factory import MATCHER_REGISTRY, ROI_SUPERVISOR_REGISTRY
from eod.utils.general.log_helper import default_logger as logger
from eod.tasks.det.models.utils.bbox_helper import (
    bbox2offset,
    bbox_iou_overlaps,
    filter_by_size,
    offset2bbox
)
from eod.tasks.det.models.utils.box_sampler import build_roi_sampler
from eod.utils.general.fp16_helper import to_float32


__all__ = ['OTAwIDSupervisor', 'OTAwIDMatcher', 'ATSSwIDSupervisor']


@ROI_SUPERVISOR_REGISTRY.register('ota_w_id')
class OTAwIDSupervisor(object):
    def __init__(self, num_classes, num_ids, matcher, norm_on_bbox=False, return_gts=False):
        self.matcher = build_matcher(matcher)
        self.norm_on_bbox = norm_on_bbox
        self.return_gts = return_gts
        self.center_sample = True
        self.num_classes = num_classes - 1  # 80
        self.num_ids = num_ids

    def get_l1_target(self, gts_xyxy, points_stride, strides, eps=1e-8):
        gts = gts_xyxy.clone()
        gts[:, 0] = (gts_xyxy[:, 0] + gts_xyxy[:, 2]) / 2
        gts[:, 1] = (gts_xyxy[:, 1] + gts_xyxy[:, 3]) / 2
        gts[:, 2] = gts_xyxy[:, 2] - gts_xyxy[:, 0]
        gts[:, 3] = gts_xyxy[:, 3] - gts_xyxy[:, 1]
        points = points_stride.clone() / strides.unsqueeze(-1)
        x_shifts = points[:, 0]
        y_shifts = points[:, 1]
        l1_target = gts.new_zeros((gts.shape[0], 4))
        l1_target[:, 0] = gts[:, 0] / strides - x_shifts
        l1_target[:, 1] = gts[:, 1] / strides - y_shifts
        l1_target[:, 2] = torch.log(gts[:, 2] / strides + eps)
        l1_target[:, 3] = torch.log(gts[:, 3] / strides + eps)
        return l1_target

    @torch.no_grad()
    def get_targets(self, locations, input, mlvl_preds):
        cls_targets = []
        id_targets = []
        reg_targets = []
        obj_targets = []
        ori_reg_targets = []
        fg_masks = []
        valid_id_masks = []

        # process mlvl_preds: --> [B, A, 85]
        mlvl_preds = [torch.cat(preds, dim=-1) for preds in mlvl_preds]
        mlvl_preds = torch.cat(mlvl_preds, dim=1)  # [B, A, 85]

        gt_bboxes = input['gt_bboxes']
        strides = input['strides']

        # center points
        num_points_per_level = [len(p) for p in locations]
        points = torch.cat(locations, dim=0)  # [A, 2]

        # batch size
        B = len(gt_bboxes)

        num_gts = 0.0
        num_fgs = 0.0
        for b_ix in range(B):
            gts = gt_bboxes[b_ix]    # [G, 5]
            num_gts += len(gts)
            preds = mlvl_preds[b_ix]  # [A, 85]
            no_gt_flag = False
            if gts.shape[0] > 0:
                try:
                    img_size = input['image_info'][b_ix][:2]
                    num_fg, gt_matched_classes, gt_matched_ids, pred_ious_this_matching, matched_gt_inds, fg_mask, expanded_strides = \
                        self.matcher.match(gts, preds, points, num_points_per_level, strides, img_size=img_size)
                except RuntimeError:
                    logger.info(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    num_fg, gt_matched_classes, gt_matched_ids, pred_ious_this_matching, matched_gt_inds, fg_mask, expanded_strides = \
                        self.matcher.match(gts, preds, points, num_points_per_level, strides, mode='cpu')
                torch.cuda.empty_cache()
                if num_fg == 0:
                    no_gt_flag = True
                else:
                    cls_target = F.one_hot(
                        (gt_matched_classes - 1).to(torch.int64), self.num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
                    valid_id_mask = gt_matched_ids > 0
                    id_target = F.one_hot((gt_matched_ids).to(torch.int64), self.num_ids + 1) * \
                        pred_ious_this_matching.unsqueeze(-1)
                    reg_target = gts[matched_gt_inds, :4]
                    obj_target = fg_mask.unsqueeze(-1)
                    l1_target = self.get_l1_target(gts[matched_gt_inds, :4],
                                                   points[fg_mask], expanded_strides[0][fg_mask])
                    num_fgs += num_fg
            else:
                no_gt_flag = True
            if no_gt_flag:
                fg_mask = preds.new_zeros(preds.shape[0], dtype=torch.bool)
                cls_target = preds.new_zeros((0, self.num_classes))
                id_target = preds.new_zeros((0, self.num_ids + 1), dtype=torch.int64)
                valid_id_mask = preds.new_zeros((0,), dtype=torch.bool)
                reg_target = preds.new_zeros((0, 4))
                obj_target = preds.new_zeros((fg_mask.shape[0], 1)).to(torch.bool)
                l1_target = preds.new_zeros((0, 4))

            cls_targets.append(cls_target)
            id_targets.append(id_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            ori_reg_targets.append(l1_target)
            fg_masks.append(fg_mask)
            valid_id_masks.append(valid_id_mask)

        return cls_targets, reg_targets, id_targets, obj_targets, ori_reg_targets, fg_masks, num_fgs, valid_id_masks


@MATCHER_REGISTRY.register('ota_w_id')
class OTAwIDMatcher(object):
    def __init__(self, num_classes, center_sample=True, pos_radius=1, candidate_k=10, radius=2.5):
        self.pos_radius = pos_radius
        self.center_sample = center_sample
        self.num_classes = num_classes - 1  # 80
        self.candidate_k = candidate_k
        self.radius = radius

    @torch.no_grad()
    def match(self, gts, preds, points, num_points_per_level, strides, mode='cuda', img_size=[]):
        ''' points: [A, 2] || gts: [G, 5] || preds: [A, 85]
        num_points_per_level:  [15808, 3952, 988, 247, 70]
        strides: [8, 16, 32, 64, 128]'''
        G = len(gts)
        if mode == 'cpu':
            logger.info('------------CPU Mode for This Batch-------------')
            gts = gts.cpu()
            points = points.cpu()

        gt_labels = gts[:, 4]   # [G, 1]
        gt_bboxes = gts[:, :4]  # [G, 4]
        gt_ids = gts[:, 5]

        if self.center_sample:
            fg_mask, is_in_boxes_and_center, expanded_strides = self.get_sample_region(
                gt_bboxes, strides, points, num_points_per_level, img_size)
        else:
            assert "Not implement"

        if fg_mask.sum() <= 1e-6:
            logger.info('no gt in center')
            return 0, None, None, None, None, None, None

        masked_preds = preds[fg_mask]
        if mode == 'cpu':
            masked_preds = masked_preds.cpu()

        preds_cls = masked_preds[:, :self.num_classes]
        preds_box = masked_preds[:, self.num_classes:-1]  # [A', 4]
        preds_obj = masked_preds[:, -1:]

        def decode_ota_box(preds):
            x1 = preds[:, 0] - preds[:, 2] / 2
            y1 = preds[:, 1] - preds[:, 3] / 2
            x2 = preds[:, 0] + preds[:, 2] / 2
            y2 = preds[:, 1] + preds[:, 3] / 2
            return torch.stack([x1, y1, x2, y2], -1)

        with torch.cuda.amp.autocast(enabled=False):
            decoded_preds_box = decode_ota_box(preds_box.float())

            pair_wise_ious = bbox_iou_overlaps(
                gt_bboxes, decoded_preds_box)
            gt_labels_ = gt_labels - 1
            gt_cls_per_image = (
                F.one_hot(gt_labels_.to(torch.int64), self.num_classes).float()
                .unsqueeze(1).repeat(1, masked_preds.shape[0], 1)
            )
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
            cls_preds_ = (
                preds_cls.float().unsqueeze(0).repeat(G, 1, 1).sigmoid_()
                * preds_obj.float().unsqueeze(0).repeat(G, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(),
                gt_cls_per_image,
                reduction="none").sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (num_fg, gt_matched_classes, gt_matched_ids, pred_ious_this_matching, matched_gt_inds
         ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_labels, gt_ids, G, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            gt_matched_ids = gt_matched_ids.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
            expanded_strides = expanded_strides.cuda()

        return num_fg, gt_matched_classes, gt_matched_ids, pred_ious_this_matching, matched_gt_inds, fg_mask, expanded_strides

    def get_sample_region(self, gt_bboxes, strides, points, num_points_per_level, img_size):
        G = gt_bboxes.shape[0]  # num_gts
        A = points.shape[0]     # num_anchors
        expanded_strides = torch.zeros(1, points.shape[0]).to(dtype=points.dtype, device=points.device)

        beg = 0
        for l_ix in range(len(num_points_per_level)):
            end = beg + num_points_per_level[l_ix]
            expanded_strides[0][beg:end] = strides[l_ix]
            beg += num_points_per_level[l_ix]

        x_center = points[:, 0].unsqueeze(0).repeat(G, 1) + 0.5 * expanded_strides  # [G, A]
        y_center = points[:, 1].unsqueeze(0).repeat(G, 1) + 0.5 * expanded_strides  # [G, A]

        # x1, x2, y1, y2
        gt_bboxes_l = (gt_bboxes[:, 0]).unsqueeze(1).repeat(1, A)  # [G, A]
        gt_bboxes_r = (gt_bboxes[:, 2]).unsqueeze(1).repeat(1, A)  # [G, A]
        gt_bboxes_t = (gt_bboxes[:, 1]).unsqueeze(1).repeat(1, A)  # [G, A]
        gt_bboxes_b = (gt_bboxes[:, 3]).unsqueeze(1).repeat(1, A)  # [G, A]

        b_l = x_center - gt_bboxes_l  # [G, A]
        b_r = gt_bboxes_r - x_center  # [G, A]
        b_t = y_center - gt_bboxes_t  # [G, A]
        b_b = gt_bboxes_b - y_center  # [G, A]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)   # [G, A, 4]
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0   # [G, A]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0         # [A]

        center_radius = self.radius

        gt_bboxes_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_bboxes_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2

        if len(img_size) > 0:
            gt_bboxes_cx = torch.clamp(gt_bboxes_cx, min=0, max=img_size[1])
            gt_bboxes_cy = torch.clamp(gt_bboxes_cy, min=0, max=img_size[0])

        gt_bboxes_l = gt_bboxes_cx.unsqueeze(1).repeat(1, A) - center_radius * expanded_strides
        gt_bboxes_r = gt_bboxes_cx.unsqueeze(1).repeat(1, A) + center_radius * expanded_strides
        gt_bboxes_t = gt_bboxes_cy.unsqueeze(1).repeat(1, A) - center_radius * expanded_strides
        gt_bboxes_b = gt_bboxes_cy.unsqueeze(1).repeat(1, A) + center_radius * expanded_strides

        c_l = x_center - gt_bboxes_l
        c_r = gt_bboxes_r - x_center
        c_t = y_center - gt_bboxes_t
        c_b = gt_bboxes_b - y_center
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center, expanded_strides

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, gt_ids, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost)  # [G, A']

        ious_in_boxes_matrix = pair_wise_ious    # [G, A']
        n_candidate_k = min(self.candidate_k, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # , max=A_-1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)    # [A']  wether matched
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        gt_matched_ids = gt_ids[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, gt_matched_ids, pred_ious_this_matching, matched_gt_inds


@ROI_SUPERVISOR_REGISTRY.register('atss_w_id')
class ATSSwIDSupervisor(object):
    '''
    Compuate targets for Adaptive Training Sample Selection
    '''

    def __init__(self, top_n=9, use_centerness=False, use_iou=False, gt_encode=True, return_gts=False):
        self.top_n = top_n
        self.use_centerness = use_centerness
        self.use_iou = use_iou
        self.gt_encode = gt_encode
        self.return_gts = return_gts

    def compuate_centerness_targets(self, loc_target, anchors):
        gts = offset2bbox(anchors, loc_target)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        _l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([_l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness_target = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))  # noqa E501
        assert not torch.isnan(centerness_target).any()
        return centerness_target

    def compuate_iou_targets(self, loc_target, anchors, loc_pred):
        decode_loc_pred = offset2bbox(anchors, loc_pred)
        decode_loc_target = offset2bbox(anchors, loc_target)
        iou_target = bbox_iou_overlaps(decode_loc_pred, decode_loc_target, aligned=True)
        return iou_target

    @torch.no_grad()
    @to_float32
    def get_targets(self, mlvl_anchors, input, mlvl_preds=None):
        r"""Match anchors with gt bboxes and sample batch samples for training

        Arguments:
           - mlvl_anchors (:obj:`list` of :obj:`FloatTensor`, fp32): [[k_0, 4], ..., [k_n,4]],
             for layer :math:`i` in FPN, k_i = h_i * w_i * A
           - input (:obj:`dict`) with:
               - gt_bboxes (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
               - image_info (list of FloatTensor): [B, >=3] (image_h, image_w, scale_factor, ...)
               - ignore_regions (list of FloatTensor): None or [B, I, 4] (x1, y1, x2, y2)

        Returns:
            cls_target (LongTensor): [B, K], {-1, 0, 1} for RPN, {-1, 0, 1...C} for retinanet
            loc_target (FloatTensor): [B, K, 4]
            centerness_target (FloatTensor): [B, K], only for use_centerness
            sample_cls_mask (ByteTensor): [B, K], binary mask, 1 for choosed samples, otherwise 0
            sample_loc_mask (ByteTensor): [B, K], binary mask, 1 for choosed positive samples, otherwise 0
        """
        INF = 100000000
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)
        image_info = input.get('image_info')
        num_anchors_per_level = [len(anchor) for anchor in mlvl_anchors]
        all_anchors = torch.cat(mlvl_anchors, dim=0)
        B = len(gt_bboxes)
        K = all_anchors.shape[0]
        neg_targets = input.get('neg_targets', None)
        if ignore_regions is None:
            ignore_regions = [None] * B
        if neg_targets is None:
            neg_targets = [0] * B
        cls_target = all_anchors.new_full((B, K), 0, dtype=torch.int64)
        loc_target = all_anchors.new_zeros((B, K, 4))
        id_target = all_anchors.new_full((B, K), -1, dtype=torch.int64)
        sample_cls_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)
        sample_loc_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)
        sample_id_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)

        anchors_cx = (all_anchors[:, 2] + all_anchors[:, 0]) / 2.0
        anchors_cy = (all_anchors[:, 3] + all_anchors[:, 1]) / 2.0
        anchor_num = anchors_cx.shape[0]
        anchor_points = torch.stack((anchors_cx, anchors_cy), dim=1)
        if self.use_iou:
            mlvl_loc_pred = list(zip(*mlvl_preds))[1]
            loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        for b_ix in range(B):
            gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)
            num_gt = gt.shape[0]
            if gt.shape[0] == 0:
                cls_target[b_ix][:] = neg_targets[b_ix]
                continue
            else:
                bbox = gt[:, :4]
                labels = gt[:, 4]
                id_labels = gt[:, 5]
                ious = bbox_iou_overlaps(all_anchors, gt)
                gt_cx = (bbox[:, 2] + bbox[:, 0]) / 2.0
                gt_cy = (bbox[:, 3] + bbox[:, 1]) / 2.0
                gt_cx = torch.clamp(gt_cx, min=0, max=image_info[b_ix][1])
                gt_cy = torch.clamp(gt_cy, min=0, max=image_info[b_ix][0])
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)
                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                # Selecting candidates based on the center distance between anchor box and object
                candidate_idxs = []
                star_idx = 0
                for num in num_anchors_per_level:
                    end_idx = star_idx + num
                    distances_per_level = distances[star_idx:end_idx]
                    topk = min(self.top_n, num)
                    _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                    candidate_idxs.append(topk_idxs_per_level + star_idx)
                    star_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=0)

                # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
                candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                iou_mean_per_gt = candidate_ious.mean(0)
                iou_std_per_gt = candidate_ious.std(0)
                iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                # Limiting the final positive samples’ center to object
                for ng in range(num_gt):
                    candidate_idxs[:, ng] += ng * anchor_num
                e_anchors_cx = anchors_cx.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                e_anchors_cy = anchors_cy.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                candidate_idxs = candidate_idxs.view(-1)
                _l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bbox[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bbox[:, 1]
                r = bbox[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = bbox[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([_l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)

                labels = labels[anchors_to_gt_indexs]
                id_labels = id_labels[anchors_to_gt_indexs]
                neg_index = anchors_to_gt_values == -INF
                pos_index = ~neg_index
                labels[anchors_to_gt_values == -INF] = neg_targets[b_ix]
                id_labels[anchors_to_gt_values == -INF] = -1
                matched_gts = bbox[anchors_to_gt_indexs]
                if self.gt_encode:
                    reg_targets_per_im = bbox2offset(all_anchors, matched_gts)
                else:
                    reg_targets_per_im = matched_gts
                cls_target[b_ix] = labels
                loc_target[b_ix] = reg_targets_per_im
                id_target[b_ix] = id_labels
                sample_cls_mask[b_ix] = pos_index
                sample_loc_mask[b_ix] = pos_index
                sample_id_mask[b_ix] = pos_index & (id_labels != 0)

            if ignore_regions[b_ix] is not None and ignore_regions[b_ix].shape[0] > 0:
                ig_bbox = ignore_regions[b_ix]
                if ig_bbox.sum() > 0:
                    ig_left = anchors_cx[:, None] - ig_bbox[..., 0]
                    ig_right = ig_bbox[..., 2] - anchors_cx[:, None]
                    ig_top = anchors_cy[:, None] - ig_bbox[..., 1]
                    ig_bottom = ig_bbox[..., 3] - anchors_cy[:, None]
                    ig_targets = torch.stack((ig_left, ig_top, ig_right, ig_bottom), -1)
                    ig_inside_bbox_mask = (ig_targets.min(-1)[0] > 0).max(-1)[0]
                    cls_target[b_ix][ig_inside_bbox_mask] = -1
                    id_target[b_ix][ig_inside_bbox_mask] = 0
                    sample_cls_mask[b_ix][ig_inside_bbox_mask] = False
                    sample_loc_mask[b_ix][ig_inside_bbox_mask] = False
                    sample_id_mask[b_ix][ig_inside_bbox_mask] = False
        if self.use_centerness or self.use_iou:
            batch_anchor = all_anchors.view(1, -1, 4).expand((sample_loc_mask.shape[0], sample_loc_mask.shape[1], 4))
            sample_anchor = batch_anchor[sample_loc_mask].contiguous().view(-1, 4)
            sample_loc_target = loc_target[sample_loc_mask].contiguous().view(-1, 4)
            if sample_loc_target.numel():
                if self.use_iou:
                    centerness_target = self.compuate_iou_targets(sample_loc_target, sample_anchor, loc_pred[sample_loc_mask].view(-1, 4))  # noqa
                else:
                    centerness_target = self.compuate_centerness_targets(
                        sample_loc_target, sample_anchor)
            else:
                centerness_target = sample_loc_target.new_zeros(sample_loc_target.shape[0])
            return cls_target, loc_target, id_target, centerness_target, sample_cls_mask, sample_loc_mask, sample_id_mask

        if self.return_gts:
            return cls_target, loc_target, id_target, sample_cls_mask, sample_loc_mask, sample_id_mask, gt_bboxes
        return cls_target, loc_target, id_target, sample_cls_mask, sample_loc_mask, sample_id_mask
