import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.models.losses import build_loss
from ..utils.debug import info_debug

__all__ = ['RelationYOLOX']


@MODULE_ZOO_REGISTRY.register('relation_yolox')
class RelationYOLOX(nn.Module):

    def __init__(self, relation_cfgs, yolox_kwargs, inplanes, num_to_refine=1000, num_as_ref=500):
        super().__init__()
        self.prefix = self.__class__.__name__
        self.roi_features_mappings = {0: 'cls', 1: 'loc', 2: 'id'}
        self.inplanes = inplanes
        self.num_to_refine = num_to_refine
        self.num_as_ref = num_as_ref
        yolox_kwargs.update({'inplanes': self.inplanes})
        self.post_module = MODULE_ZOO_REGISTRY.build(dict(
            type='yolox_post_w_id',
            kwargs=yolox_kwargs,
        ))
        self.roi_pred_dims = {0: self.post_module.num_classes - 1,
                              1: 4, 2: self.post_module.num_ids}
        self.relation_modules = nn.ModuleList()
        self.relation_indices = {}
        for lvl_idx in range(len(self.inplanes)):
            lvl_relations = nn.ModuleList()
            for idx, relation_cfg in enumerate(relation_cfgs):
                relation_cfg['kwargs']['embed_dim'] = self.inplanes[lvl_idx]
                relation_module = MODULE_ZOO_REGISTRY.build(dict(
                    type=relation_cfg['type'],
                    kwargs=relation_cfg['kwargs'],
                ))
                lvl_relations.append(relation_module)
                self.relation_indices[relation_cfg['index']] = idx
            self.relation_modules.append(lvl_relations)
        prior_prob = 1e-2
        for relation_idx in self.relation_indices:
            mod_list = nn.ModuleList()
            for lvl_idx, lvl_c in enumerate(self.inplanes):
                mod_list.append(
                    nn.Conv2d(
                        in_channels=lvl_c,
                        out_channels=self.roi_pred_dims[relation_idx],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            for conv in mod_list:
                b = conv.bias.view(1, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            loss = build_loss(relation_cfg['loss'])
            setattr(self, self.roi_features_mappings[relation_idx] + '_loss', loss)
            setattr(self, self.roi_features_mappings[relation_idx] + '_preds', mod_list)
        self.obj_preds = nn.ModuleList()
        for lvl_idx, lvl_c in enumerate(self.inplanes):
            self.obj_preds.append(nn.Conv2d(
                in_channels=lvl_c,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            ))
        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def get_targets(self, input, return_preds=False):
        features = input['features']
        strides = input['strides']
        mlvl_preds = input['preds']

        # [B, hi*wi*A, :]
        mlvl_preds, mlvl_shapes = self.post_module.permute_preds(mlvl_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]

        # [hi*wi*A, 2], for C4 there is only one layer, for FPN generate anchors for each layer
        mlvl_locations = self.post_module.point_generator.get_anchors(mlvl_shapes, device=features[0].device)

        mlvl_ori_loc_preds = None
        if self.post_module.use_l1:
            mlvl_ori_loc_preds = []
        for l_ix in range(len(mlvl_locations)):
            # add preds for l1 loss
            if self.post_module.use_l1:
                mlvl_ori_loc_preds.append(mlvl_preds[l_ix][1].clone())
            mlvl_preds[l_ix][1][..., :2] *= strides[l_ix]
            mlvl_preds[l_ix][1][..., :2] += mlvl_locations[l_ix]
            if self.post_module.norm_on_bbox:
                mlvl_preds[l_ix][1][..., 2:4] = F.relu(mlvl_preds[l_ix][1][..., 2:4])  # , inplace=True)
                mlvl_preds[l_ix][1][..., 2:4] *= strides[l_ix]
            else:
                mlvl_preds[l_ix][1][..., 2:4] = torch.exp(mlvl_preds[l_ix][1][..., 2:4]) * strides[l_ix]

        if self.training:
            targets = self.post_module.supervisor.get_targets(mlvl_locations, input, mlvl_preds)
        else:
            targets = None
        if return_preds:
            return targets, mlvl_preds, mlvl_ori_loc_preds
        else:
            return targets

    def get_fg_masks(self, fg_masks):
        factor = 0
        sizes = []
        for i in range(len(self.inplanes)):
            factor += 1 << (i << 1)
            sizes.insert(0, 1 << (i << 1))
        prev = 0
        prev_num = [0 for mask in fg_masks]
        lvl_masks = []
        target_range = []
        for i in range(len(self.inplanes)):
            lvl_batch = []
            one_range = []
            for j, mask in enumerate(fg_masks):
                s2 = mask.size(0) // factor
                now = s2 * sizes[i]
                lvl_mask = mask[prev: prev + now]
                lvl_batch.append(lvl_mask)
                now_num = int((lvl_mask).sum())
                # print(i, j, now_num)
                one_range.append((prev_num[j], prev_num[j] + now_num))
                prev_num[j] += now_num
            target_range.append(one_range)
            lvl_masks.append(torch.stack(lvl_batch))
            prev += now
        return lvl_masks, target_range

    def get_selected_indices(self, fg_mask, target_range, objness, mode='main'):
        if mode == 'main':
            if fg_mask is not None:
                num = min(fg_mask.size(1), self.num_to_refine)  # must larger than fg_num
            else:
                num = self.num_to_refine
        else:
            if fg_mask is not None:
                num = min(fg_mask.size(1), self.num_as_ref)
            else:
                num = self.num_as_ref
        if mode == 'main' and fg_mask is not None:
            fg_max = int(fg_mask.sum(dim=1).max())
            num = max(num, fg_max)
        new_masks = []
        new_fg_masks = []
        new_gt_inds = []
        if fg_mask is not None:
            for i in range(fg_mask.size(0)):
                fg_num = int(fg_mask[i].sum())
                assert mode != 'main' or fg_num <= num, str(fg_num) + ', ' + str(fg_mask[i].shape)
                bg_num = num - fg_num
                assert fg_mask[i].nonzero().numel() == target_range[i][1] - target_range[i][0]
                obj = objness[i].reshape(fg_mask[i].numel())
                if bg_num > 0:
                    new_mask = fg_mask[i].clone()
                    bg_inds = (~fg_mask[i]).nonzero().flatten()
                    bgs = obj[~fg_mask[i]]
                    # print(bgs.shape, bg_num, mode, 'bg')
                    _, inds = bgs.topk(bg_num)
                    bg_inds = bg_inds[inds]
                    new_mask[bg_inds] = True
                    gt_inds = torch.arange(target_range[i][0], target_range[i][1]).to(fg_mask.device)
                else:
                    new_mask = fg_mask[i].new_zeros(fg_mask[i].shape, dtype=torch.bool)
                    fg_inds = (fg_mask[i]).nonzero().flatten()
                    fgs = obj[fg_mask[i]]
                    # print(fgs.shape, fg_num, mode, 'fg')
                    _, inds = fgs.topk(num)
                    fg_inds = fg_inds[inds]
                    new_mask[fg_inds] = True
                    inds = inds.sort()[0]
                    gt_inds = target_range[i][0] + inds
                new_gt_inds.append(gt_inds)
                new_masks.append(new_mask.nonzero().flatten())
                new_fg_masks.append(fg_mask[i][new_mask])
            new_masks = torch.stack(new_masks)
            new_fg_masks = torch.stack(new_fg_masks)
            return new_masks, new_fg_masks, new_gt_inds
        else:
            objness = objness.reshape(objness.size(0), -1)
            _, inds = objness.topk(num, dim=1)
            # mask = objness.new_zeros(objness.shape, dtype=torch.bool)
            return inds

    def get_top_feats(self, feats, indices):
        if feats is None:
            return None
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return torch.gather(feats, 1, indices.unsqueeze(-1).repeat(1, 1, c))

    # def get_range(self, target, ranges):
    #     ret = []
    #     for t, r in zip(target, ranges):
    #         ret.append(t[r[0]: r[1]])
    #     return ret

    def forward(self, data):
        m = len(data['ref'])
        if self.training:
            target_main, mlvl_preds, mlvl_ori_loc_preds = self.get_targets(data['main'], return_preds=True)
            # info_debug(data['main'])
            # info_debug(target_main)
            # info_debug(mlvl_preds)
            targets_ref = [self.get_targets(ref) for ref in data['ref']]
            ref_pred0 = data['ref'][0]['preds'][0][0][0].reshape(-1)
            # print(ref_pred0[targets_ref[0][5][0][:ref_pred0.size(0)]][:50], targets_ref[0][0][0][:50], '0~~~')
            fg_masks_main, target_range_main = self.get_fg_masks(target_main[5])
            fg_masks_ref, target_range_ref = zip(*[self.get_fg_masks(ref[5]) for ref in targets_ref])
        else:
            _, mlvl_preds, mlvl_ori_loc_preds = self.get_targets(data['main'], return_preds=True)
        all_refined_preds = []
        all_refined_feats = []
        all_relation_stuffs = {}
        all_fg_masks = []
        for lvl_idx, lvl_c in enumerate(self.inplanes):
            refined_lvl_preds = []
            refined_lvl_feats = []
            main_roi_feats = data['main']['roi_features'][lvl_idx]
            if self.training:
                selected_main, selected_fg_masks, selected_gt_idx = self.get_selected_indices(
                    fg_masks_main[lvl_idx], target_range_main[lvl_idx], data['main']['preds'][lvl_idx][3])
                selected_ref, selected_fg_masks_ref, selected_gt_idx_ref = zip(
                    *[self.get_selected_indices(fg_masks_ref[i][lvl_idx], target_range_ref[i][lvl_idx], data['ref'][i]['preds'][lvl_idx][3], mode='ref') for i in range(m)])
                # if lvl_idx == 0:
                #     ref_pred0 = data['ref'][0]['preds'][0][0]
                #     ref_pred0 = self.get_top_feats(ref_pred0, selected_ref[0])
                #     print(selected_gt_idx_ref[0][0].shape, targets_ref[0][0][0].shape)
                #     print(ref_pred0[0][selected_fg_masks_ref[0][0]][:50],
                #           targets_ref[0][0][0][selected_gt_idx_ref[0][0]][:50], 're~~~')
                selected_fg_masks_ref = torch.cat(selected_fg_masks_ref, dim=1)
                all_fg_masks.append(selected_fg_masks)
            else:
                selected_main = self.get_selected_indices(None, None, data['main']['preds'][lvl_idx][3])
                selected_ref = [self.get_selected_indices(
                    None, None, data['ref'][i]['preds'][lvl_idx][3], mode='ref') for i in range(m)]
            for idx, roi_feat in enumerate(main_roi_feats):
                relation_idx = self.relation_indices.get(idx, -1)
                roi_feat = self.get_top_feats(roi_feat, selected_main)
                roi_preds = self.get_top_feats(data['main']['preds'][lvl_idx][idx], selected_main)
                if relation_idx < 0:
                    refined_lvl_preds.append(roi_preds.permute(0, 2, 1).unsqueeze(-1))
                    refined_lvl_feats.append(roi_feat.permute(0, 2, 1).unsqueeze(-1))
                else:
                    roi_feat_ref = [self.get_top_feats(data['ref'][i]['roi_features']
                                                       [lvl_idx][idx], selected_ref[i]) for i in range(m)]
                    roi_feat_ref = torch.cat(roi_feat_ref, dim=1)
                    if self.training:
                        target_this = target_main[idx]
                        # target_this = self.get_range(target_this, target_range_main[lvl_idx])
                        target_this = [t[gi] for t, gi in zip(target_this, selected_gt_idx)]
                        # target_ref = [torch.cat(o, dim=0) for o in zip(
                        #     *[self.get_range(targets_ref[i][idx], target_range_ref[i][lvl_idx]) for i in range(m)])]
                        target_ref_splits = [[t[gi] for t, gi in zip(
                            targets_ref[i][idx], selected_gt_idx_ref[i])] for i in range(m)]
                        target_ref = [torch.cat(o, dim=0) for o in zip(*target_ref_splits)]
                        refined_feats, relation_stuffs = self.relation_modules[lvl_idx][relation_idx](roi_feat, roi_feat_ref, target_main=(
                            selected_fg_masks, target_this), target_ref=(selected_fg_masks_ref, target_ref), original_preds=roi_preds)
                        for k, v in relation_stuffs.items():
                            all_relation_stuffs['relation.%d.%d.%s' % (lvl_idx, relation_idx, k)] = v
                    else:
                        refined_feats, _ = self.relation_modules[lvl_idx][relation_idx](
                            roi_feat, roi_feat_ref, original_preds=roi_preds)
                    refined_feats = refined_feats.permute(0, 2, 1).unsqueeze(-1)
                    refined_lvl_pred = getattr(
                        self, self.roi_features_mappings[idx] + '_preds')[lvl_idx](refined_feats)
                    refined_lvl_preds.append(refined_lvl_pred)
                    refined_lvl_feats.append(refined_feats)
            refined_lvl_preds.append(self.obj_preds[lvl_idx](refined_lvl_feats[0]))
            all_refined_preds.append(refined_lvl_preds)
            all_refined_feats.append(refined_lvl_feats)
        if self.training:
            if not self.post_module.use_l1:
                losses_rpn = self.post_module.get_loss(target_main, mlvl_preds)
            else:
                losses_rpn = self.post_module.get_loss(target_main, mlvl_preds, mlvl_ori_loc_preds)
            losses_refined = self.get_loss(target_main, all_fg_masks, all_refined_preds, all_refined_feats)
            losses = {}
            losses.update(all_relation_stuffs)
            losses.update(losses_rpn)
            losses.update(losses_refined)
            # print(losses)
            return losses
        else:
            with torch.no_grad():
                all_refined_preds = self.post_module.apply_activation(mlvl_preds)
                results = self.post_module.predictor.predict(mlvl_preds)
                return results

    def get_loss(self, target, fg_masks, mlvl_preds, features):
        # info_debug(mlvl_preds)
        fg_masks = torch.cat(fg_masks, dim=1)
        mlvl_preds, mlvl_shapes = self.post_module.permute_preds(mlvl_preds)
        preds = list(zip(*mlvl_preds))
        losses = {}
        if self.post_module.all_reduce_norm and dist.is_initialized():
            num_fgs = self.post_module.get_ave_normalizer(fg_masks)
        else:
            num_fgs = fg_masks.sum().item()
        num_fgs = max(num_fgs, 1)
        for idx in self.relation_indices:
            feat_name = self.roi_features_mappings[idx]
            pred = preds[idx]
            pred = torch.cat(pred, dim=1)
            target_this = torch.cat(target[idx], dim=0)
            # print(pred.shape, feat_name, pred[fg_masks].shape, target_this.shape)
            # print(pred.mean(), target_this.mean())
            loss = getattr(self, feat_name + '_loss')(pred[fg_masks], target_this, normalizer_override=num_fgs)
            losses[self.prefix + '.' + feat_name + '_loss'] = loss
        losses[self.prefix + '.obj_loss'] = self.post_module.obj_loss(
            torch.cat(preds[3], dim=1).flatten(),
            fg_masks.flatten(),
            normalizer_override=num_fgs)
        return losses
