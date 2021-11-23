import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.models.losses import build_loss
from ..utils.debug import info_debug, get_debugger
from ..utils import map_transpose

__all__ = ['RelationYOLOX']


@MODULE_ZOO_REGISTRY.register('relation_yolox')
class RelationYOLOX(nn.Module):

    def __init__(self, relation_cfgs, yolox_kwargs, inplanes, num_to_refine=1000, num_as_ref=500):
        super().__init__()
        self.vis = True
        self.prefix = self.__class__.__name__
        self.roi_features_mappings = {0: 'cls', 1: 'loc', 2: 'id'}
        self.inplanes = inplanes
        self.num_to_refine = num_to_refine
        self.num_as_ref = num_as_ref
        self.size_factors = []
        for i in range(len(self.inplanes)):
            self.size_factors.insert(0, 1 << (i << 1))
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
            idx = self.relation_indices[relation_idx]
            if self.roi_features_mappings[relation_idx] == 'id':
                mod_list = nn.Conv2d(
                    in_channels=lvl_c,
                    out_channels=self.roi_pred_dims[relation_idx],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for conv in [mod_list]:
                    b = conv.bias.view(1, -1)
                    b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                    conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            else:
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
            loss = build_loss(relation_cfgs[idx]['loss'])
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

    def get_targets(self, input):
        mlvl_preds, mlvl_locations, mlvl_ori_loc_preds = self.post_module.prepare_preds(input)
        if self.training:
            targets = self.post_module.supervisor.get_targets(mlvl_locations, input, mlvl_preds)
        else:
            targets = None
        return targets, mlvl_preds, mlvl_ori_loc_preds

    @torch.no_grad()
    def get_fg_masks(self, fg_masks):
        sizes = self.size_factors
        factor = sum(sizes)
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

    @torch.no_grad()
    def get_selected_indices(self, fg_mask, target_range, mlvl_preds, lvl, mode='main'):
        # info_debug(mlvl_preds)
        if mode == 'main':
            # if fg_mask is not None:
            #     num = min(fg_mask.size(1), self.num_to_refine)  # must larger than fg_num
            # else:
            #     num = min(self.num_to_refine, objness.size(2) * objness.size(3))
            num = self.num_to_refine * self.size_factors[lvl]
            num = min(num, 1200)
        else:
            # if fg_mask is not None:
            #     num = min(fg_mask.size(1), self.num_as_ref)
            # else:
            #     num = self.num_as_ref
            num = self.num_as_ref * self.size_factors[lvl]
            num = min(num, 300)
        # if mode == 'main' and fg_mask is not None:
        #     fg_max = int(fg_mask.sum(dim=1).max())
        #     num = max(num, fg_max)
        keeps = self.post_module.predictor.lvl_nms(mlvl_preds, preserved=num)

        # info_debug(keeps, prefix='<%d>' % (self.num_to_refine))
        # new_masks = []
        # new_fg_masks = []
        # new_gt_inds = []
        if fg_mask is not None:
            new_fg_mask = torch.gather(fg_mask, 1, keeps)
            gt_inds_ = fg_mask.new_zeros(fg_mask.shape, dtype=torch.int64)
            new_gt_inds = []
            for i in range(fg_mask.size(0)):
                gt_inds = torch.arange(target_range[i][0], target_range[i][1]).to(fg_mask.device)
                gt_inds_[i][fg_mask[i]] = gt_inds + 1
                gathered_gts_inds = torch.gather(gt_inds_[i], 0, keeps[i])
                new_gt_inds_i = gathered_gts_inds[gathered_gts_inds > 0] - 1
                new_gt_inds.append(new_gt_inds_i)
            # info_debug([keeps, new_fg_mask, new_gt_inds])
            # info_debug([new_fg_mask.nonzero(), new_gt_inds])
            # info_debug(new_gt_inds_i.unique(), prefix='<%d>' % (target_range[i][1] - target_range[i][0]))
            return keeps, new_fg_mask, new_gt_inds

            # for i in range(fg_mask.size(0)):
            #     fg_num = int(fg_mask[i].sum())
            #     assert mode != 'main' or fg_num <= num, str(fg_num) + ', ' + str(fg_mask[i].shape)
            #     bg_num = num - fg_num
            #     assert fg_mask[i].nonzero().numel() == target_range[i][1] - target_range[i][0]
            #     obj = objness[i].reshape(fg_mask[i].numel())
            #     if bg_num > 0:
            #         new_mask = fg_mask[i].clone()
            #         bg_inds = (~fg_mask[i]).nonzero().flatten()
            #         bgs = obj[~fg_mask[i]]
            #         # print(bgs.shape, bg_num, mode, 'bg')
            #         _, inds = bgs.topk(bg_num)
            #         bg_inds = bg_inds[inds]
            #         new_mask[bg_inds] = True
            #         gt_inds = torch.arange(target_range[i][0], target_range[i][1]).to(fg_mask.device)
            #     else:
            #         new_mask = fg_mask[i].new_zeros(fg_mask[i].shape, dtype=torch.bool)
            #         fg_inds = (fg_mask[i]).nonzero().flatten()
            #         fgs = obj[fg_mask[i]]
            #         # print(fgs.shape, fg_num, mode, 'fg')
            #         _, inds = fgs.topk(num)
            #         fg_inds = fg_inds[inds]
            #         new_mask[fg_inds] = True
            #         inds = inds.sort()[0]
            #         gt_inds = target_range[i][0] + inds
            #     new_gt_inds.append(gt_inds)
            #     new_masks.append(new_mask.nonzero().flatten())
            #     new_fg_masks.append(fg_mask[i][new_mask])
            # new_masks = torch.stack(new_masks)
            # new_fg_masks = torch.stack(new_fg_masks)
            # return new_masks, new_fg_masks, new_gt_inds
        else:
            # if stride < 16:
            #     ker = 3
            #     objness_nms = F.max_pool2d(objness, ker, 1, 1, 1)
            #     mask = objness_nms == objness
            # else:
            #     mask = objness.new_ones(objness.shape, dtype=torch.bool)
            # all_inds = []
            # for i in range(objness.size(0)):
            #     nms_indices = mask[i].flatten().nonzero().flatten()
            #     other = (~mask[i]).flatten().nonzero().flatten()
            #     objness_i = objness[i].flatten()[nms_indices]
            #     # print(nms_indices.shape, objness_i.shape)
            #     _, inds = objness_i.topk(min(num, objness_i.size(0)))
            #     inds = nms_indices[inds]
            #     if inds.size(0) < num:
            #         inds = torch.cat([inds, other[: num - inds.size(0)]])
            #     all_inds.append(inds)
            # return torch.stack(all_inds)
            # objness = objness.reshape(objness.size(0), -1)
            # _, inds = objness.topk(num, dim=1)
            # mask = objness.new_zeros(objness.shape, dtype=torch.bool)
            return keeps

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
        self.vis = data.get('vis_flag', False)
        m = len(data['ref'])
        if self.training:
            target_main, mlvl_preds, mlvl_ori_loc_preds = self.get_targets(data['main'])
            # info_debug(data['main'])
            # info_debug(target_main)
            # info_debug(mlvl_preds)
            targets_ref, mlvl_preds_ref, _ = map_transpose([self.get_targets(ref) for ref in data['ref']])
            # ref_pred0 = data['ref'][0]['preds'][0][0][0].reshape(-1)
            # print(ref_pred0[targets_ref[0][5][0][:ref_pred0.size(0)]][:50], targets_ref[0][0][0][:50], '0~~~')
            fg_masks_main, target_range_main = self.get_fg_masks(target_main[5])
            fg_masks_ref, target_range_ref = zip(*[self.get_fg_masks(ref[5]) for ref in targets_ref])
        else:
            mlvl_preds, _, mlvl_ori_loc_preds = self.post_module.prepare_preds(data['main'])
            mlvl_preds_ref, _, mlvl_ori_loc_preds_ref = map_transpose(
                [self.post_module.prepare_preds(o) for o in data['ref']])
        mlvl_preds_activated = self.post_module.apply_activation(mlvl_preds)
        mlvl_preds_ref_activated = [self.post_module.apply_activation(o) for o in mlvl_preds_ref]
        all_refined_preds = []
        all_refined_feats = []
        original_lvl_preds = []
        all_relation_stuffs = {}
        all_fg_masks = []
        all_target_main = []
        all_fg_masks_ref = []
        all_target_ref = []
        mlvl_selected_gt = []
        for lvl_idx, lvl_c in enumerate(self.inplanes):
            original_preds_main = []
            refined_lvl_preds = []
            refined_lvl_feats = []
            main_roi_feats = data['main']['roi_features'][lvl_idx]
            all_target_main.append([])
            all_target_ref.append([])
            if self.training:
                selected_main, selected_fg_masks, selected_gt_idx = self.get_selected_indices(
                    fg_masks_main[lvl_idx], target_range_main[lvl_idx], mlvl_preds_activated[lvl_idx], lvl_idx)
                selected_ref, selected_fg_masks_ref, selected_gt_idx_ref = map_transpose(
                    [self.get_selected_indices(
                        fg_masks_ref[i][lvl_idx],
                        target_range_ref[i][lvl_idx],
                        mlvl_preds_ref_activated[i][lvl_idx],
                        lvl_idx,
                        mode='ref',
                    ) for i in range(m)])
                # if lvl_idx == 0:
                #     ref_pred0 = data['ref'][0]['preds'][0][0]
                #     ref_pred0 = self.get_top_feats(ref_pred0, selected_ref[0])
                #     print(selected_gt_idx_ref[0][0].shape, targets_ref[0][0][0].shape)
                #     print(ref_pred0[0][selected_fg_masks_ref[0][0]][:50],
                #           targets_ref[0][0][0][selected_gt_idx_ref[0][0]][:50], 're~~~')
                selected_fg_masks_ref = torch.cat(selected_fg_masks_ref, dim=1)
                all_fg_masks_ref.append(selected_fg_masks_ref)
                all_fg_masks.append(selected_fg_masks)
                mlvl_selected_gt.append(selected_gt_idx)
            else:
                with get_debugger().no_debug():
                    selected_main = self.get_selected_indices(
                        None, None, mlvl_preds_activated[lvl_idx], lvl_idx)
                    selected_ref = [self.get_selected_indices(
                        None, None, mlvl_preds_ref_activated[i][lvl_idx], lvl_idx, mode='ref') for i in range(m)]
            for idx, roi_feat in enumerate(main_roi_feats):
                relation_idx = self.relation_indices.get(idx, -1)
                roi_feat = self.get_top_feats(roi_feat, selected_main)
                if self.roi_features_mappings[idx] == 'loc':
                    c = self.roi_pred_dims[idx]
                    roi_preds = torch.gather(mlvl_preds[lvl_idx][idx], 1,
                                             selected_main.unsqueeze(-1).repeat(1, 1, c))
                else:
                    roi_preds = self.get_top_feats(data['main']['preds'][lvl_idx][idx], selected_main)
                if relation_idx < 0:
                    refined_lvl_preds.append(roi_preds)
                    original_preds_main.append(roi_preds)
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
                        if self.vis:
                            all_target_main[-1].append(target_this)
                            all_target_ref[-1].append(target_ref)
                        if self.roi_features_mappings[idx] != 'id':
                            target_this = [torch.cat([o.new_zeros((o.size(0), 1)), o], dim=1) for o in target_this]
                            target_ref = [torch.cat([o.new_zeros((o.size(0), 1)), o], dim=1) for o in target_ref]
                        refined_feats, relation_stuffs = self.relation_modules[lvl_idx][relation_idx](roi_feat, roi_feat_ref, target_main=(
                            selected_fg_masks, target_this), target_ref=(selected_fg_masks_ref, target_ref), original_preds=roi_preds)
                        for k, v in relation_stuffs.items():
                            all_relation_stuffs['relation.%d.%d.%s' % (lvl_idx, relation_idx, k)] = v
                    else:
                        refined_feats, _ = self.relation_modules[lvl_idx][relation_idx](
                            roi_feat, roi_feat_ref, original_preds=roi_preds)
                    refined_feats = refined_feats.permute(0, 2, 1).unsqueeze(-1)
                    pred_func = getattr(self, self.roi_features_mappings[idx] + '_preds')
                    if isinstance(pred_func, nn.ModuleList):
                        refined_lvl_pred = pred_func[lvl_idx](refined_feats)
                    else:
                        refined_lvl_pred = pred_func(refined_feats)
                    refined_lvl_preds.append(refined_lvl_pred.squeeze(-1).permute(0, 2, 1))
                    refined_lvl_feats.append(refined_feats)
                    original_preds_main.append(roi_preds)
            refined_lvl_preds.append(self.obj_preds[lvl_idx](refined_lvl_feats[0]).squeeze(-1).permute(0, 2, 1))
            original_preds_main.append(self.get_top_feats(data['main']['preds'][lvl_idx][3], selected_main))
            all_refined_preds.append(refined_lvl_preds)
            all_refined_feats.append(refined_lvl_feats)
            original_lvl_preds.append(original_preds_main)
        # refined_mlvl_preds, refined_mlvl_locations, refined_mlvl_ori_loc_preds = self.post_module.prepare_preds(
        #     {
        #         'features': data['main']['features'],
        #         'preds': all_refined_preds,
        #         'strides': data['main']['strides'],
        #     }
        # )
        refined_mlvl_preds = all_refined_preds
        if self.training:
            losses_rpn = self.post_module.get_loss(target_main, mlvl_preds, mlvl_ori_loc_preds)
            losses_refined = self.get_loss(target_main, all_fg_masks, refined_mlvl_preds, mlvl_selected_gt)
            losses = {}
            losses.update(all_relation_stuffs)
            losses.update(losses_rpn)
            losses.update(losses_refined)
            if self.vis:
                losses['refined_pred_main'] = refined_mlvl_preds
                losses['original_pred_main'] = original_lvl_preds
                losses['targets_main'] = (all_target_main, all_fg_masks)
                losses['original_pred_ref'] = []
                losses['targets_ref'] = (all_target_ref, all_fg_masks_ref)
            # print(losses)
            return losses
        else:
            # print(mlvl_preds[0][1].shape, mlvl_preds[0][1][0, :10])
            # print(refined_mlvl_preds[0][1].shape, refined_mlvl_preds[0][1][0, :10])
            # info_debug(mlvl_preds)
            rpn_results = self.get_results(
                mlvl_preds,
                data['main']['roi_features'],
            )
            # info_debug(refined_mlvl_preds)
            refined_results = self.get_results(
                refined_mlvl_preds,
                all_refined_feats,
            )
            for k in rpn_results:
                refined_results[k + '_rpn'] = rpn_results[k]
                # refined_results[k] = rpn_results[k]
            if self.vis:
                refined_results['refined_pred_main'] = refined_mlvl_preds
                refined_results['original_pred_main'] = mlvl_preds
                refined_results['targets_main'] = None
                refined_results['original_pred_ref'] = []
                refined_results['targets_ref'] = None
            return refined_results

    @torch.no_grad()
    def get_results(self, mlvl_preds, mlvl_feats):
        id_feats = [mlvl_feats[lvl_idx][2] for lvl_idx in range(len(self.inplanes))]
        mlvl_preds = self.post_module.apply_activation(mlvl_preds)
        with get_debugger().no_debug():
            results = self.post_module.predictor.predict(mlvl_preds, id_feats)
        return results

    def get_loss(self, target, fg_masks, mlvl_preds, gt_indices):
        # info_debug(mlvl_preds)
        fg_masks = torch.cat(fg_masks, dim=1)
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
            cat_gt_inds = [torch.cat([o[i] for o in gt_indices]) for i in range(fg_masks.size(0))]
            target_this = torch.cat([o[gti] for o, gti in zip(target[idx], cat_gt_inds)], dim=0)
            # print(pred.shape, feat_name, pred[fg_masks].shape, target_this.shape)
            # print(pred.mean(), target_this.mean())
            if self.roi_features_mappings[idx] == 'id':
                valid = torch.cat([o[gti] for o, gti in zip(target[7], cat_gt_inds)], dim=0)
                # valid = torch.cat(target[7], dim=0)
                if self.post_module.all_reduce_norm and dist.is_initialized():
                    num_ids = self.post_module.get_ave_normalizer(valid)
                else:
                    num_ids = valid.sum().item()
                num_ids = max(num_ids, 1)
                loss = getattr(self, feat_name + '_loss')(pred[fg_masks][valid],
                                                          target_this[valid][:, 1:], normalizer_override=num_ids)
            else:
                # info_debug([pred, fg_masks, pred[fg_masks], gt_indices, target[idx]])
                loss = getattr(self, feat_name + '_loss')(pred[fg_masks], target_this, normalizer_override=num_fgs)
            losses[self.prefix + '.' + feat_name + '_loss'] = loss
        losses[self.prefix + '.obj_loss'] = self.post_module.obj_loss(
            torch.cat(preds[3], dim=1).flatten(),
            fg_masks.flatten(),
            normalizer_override=num_fgs)
        return losses
