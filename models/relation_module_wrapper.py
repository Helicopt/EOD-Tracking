import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.models.losses import build_loss
from ..utils.debug import info_debug, get_debugger, logger_print
from eod.utils.env.dist_helper import env
from ..utils import map_transpose

__all__ = ['RelationYOLOX', 'RelationRetina']


@MODULE_ZOO_REGISTRY.register('relation_yolox')
class RelationYOLOX(nn.Module):

    def __init__(self, relation_cfgs, yolox_kwargs, inplanes, num_to_refine=1000, num_as_ref=500, share=False, refine_objness=True, dismiss_aug=False, init_prior=0.01, normalize_id=False, balanced_loss_weight='none', balance_scale=1.):
        super().__init__()
        self.vis = True
        self.prefix = self.__class__.__name__
        self.roi_features_mappings = {0: 'cls', 1: 'loc', 2: 'id'}
        self.inplanes = inplanes
        self.num_to_refine = num_to_refine
        self.num_as_ref = num_as_ref
        self.size_factors = []
        self.dismiss_aug = dismiss_aug
        self.balanced_loss_weight = balanced_loss_weight
        self.balance_scale = balance_scale
        if self.balanced_loss_weight == 'none':
            self.lw_id = 1.
            self.lw_det = 1.
        elif self.balanced_loss_weight == 'auto':
            self.lw_id = nn.Parameter(-0.5 * torch.ones(1).float())
            self.lw_det = nn.Parameter(-0.6 * torch.ones(1).float())
        else:
            assert False, '%s not defined' % self.balanced_loss_weight
        for i in range(len(self.inplanes)):
            self.size_factors.insert(0, 1 << (i << 1))
        yolox_kwargs.update({'inplanes': self.inplanes})
        self.post_module = MODULE_ZOO_REGISTRY.build(dict(
            type='yolox_post_w_id',
            kwargs=yolox_kwargs,
        ))
        self.normalize_id = normalize_id
        self.emb_scale = math.sqrt(2) * math.log(self.post_module.num_ids)
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
        prior_prob = init_prior
        for relation_idx in self.relation_indices:
            idx = self.relation_indices[relation_idx]
            if self.roi_features_mappings[relation_idx] == 'id' or share:
                mod_list = nn.Conv2d(
                    in_channels=self.inplanes[0],
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
        self.refine_obj = refine_objness
        if refine_objness:
            self.obj_preds = nn.ModuleList()
            for lvl_idx, lvl_c in enumerate(self.inplanes):
                if share:
                    self.obj_preds = nn.Conv2d(
                        in_channels=lvl_c,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    mod_list = [self.obj_preds]
                else:
                    self.obj_preds.append(nn.Conv2d(
                        in_channels=lvl_c,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ))
                    mod_list = self.obj_preds
            for conv in mod_list:
                b = conv.bias.view(1, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def get_targets(self, input, force=False):
        mlvl_preds, mlvl_locations, mlvl_ori_loc_preds = self.post_module.prepare_preds(input)
        if self.training or force:
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
            num = self.num_to_refine * self.size_factors[lvl]
            num = min(num, 1200)
        else:
            num = self.num_as_ref * self.size_factors[lvl]
            num = min(num, 300)
        keeps = self.post_module.predictor.lvl_nms(mlvl_preds, preserved=num)

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

        else:
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
        noaug_flag = data['noaug_flag']
        self.vis = data.get('vis_flag', False)
        m = len(data['ref'])
        if self.training:
            target_main, mlvl_preds, mlvl_ori_loc_preds = self.get_targets(data['main'])
            # info_debug(data['main'])
            # info_debug(target_main)
            # info_debug(mlvl_preds)
            targets_ref, mlvl_preds_ref, _ = map_transpose([self.get_targets(ref) for ref in data['ref']])
            fg_masks_main, target_range_main = self.get_fg_masks(target_main[5])
            fg_masks_ref, target_range_ref = map_transpose([self.get_fg_masks(ref[5]) for ref in targets_ref])
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
                            selected_fg_masks, target_this), target_ref=(selected_fg_masks_ref, target_ref), original_preds=roi_preds,
                            framerates=data.get('framerate', None))
                        for k, v in relation_stuffs.items():
                            all_relation_stuffs['relation.%d.%d.%s' % (lvl_idx, relation_idx, k)] = v
                    else:
                        refined_feats, _ = self.relation_modules[lvl_idx][relation_idx](
                            roi_feat, roi_feat_ref, original_preds=roi_preds, framerates=data.get('framerate', None))
                    refined_feats = refined_feats.permute(0, 2, 1).unsqueeze(-1)
                    if self.roi_features_mappings[idx] == 'id' and self.normalize_id:
                        refined_feats = F.normalize(refined_feats, dim=1) * self.emb_scale
                    pred_func = getattr(self, self.roi_features_mappings[idx] + '_preds')
                    if isinstance(pred_func, nn.ModuleList):
                        refined_lvl_pred = pred_func[lvl_idx](refined_feats)
                    else:
                        refined_lvl_pred = pred_func(refined_feats)
                    refined_lvl_preds.append(refined_lvl_pred.squeeze(-1).permute(0, 2, 1))
                    refined_lvl_feats.append(refined_feats)
                    original_preds_main.append(roi_preds)
            ori_obj_pred = self.get_top_feats(data['main']['preds'][lvl_idx][3], selected_main)
            original_preds_main.append(ori_obj_pred)
            if self.refine_obj:
                refined_lvl_preds.append(self.obj_preds[lvl_idx](refined_lvl_feats[0]).squeeze(-1).permute(0, 2, 1))
            else:
                refined_lvl_preds.append(ori_obj_pred)
            all_refined_preds.append(refined_lvl_preds)
            all_refined_feats.append(refined_lvl_feats)
            original_lvl_preds.append(original_preds_main)
        refined_mlvl_preds = all_refined_preds
        if self.training:
            del targets_ref
            del target_ref_splits
        del mlvl_preds_ref
        del mlvl_preds_ref_activated
        if self.training:
            losses_rpn = self.post_module.get_loss(target_main, mlvl_preds, mlvl_ori_loc_preds, noaug_flag=noaug_flag)
            # logger_print(losses_rpn, rk=1)
            # logger_print(noaug_flag, rk=1)
            # logger_print(data['main']['image_id'], rk=3)
            # logger_print(data['main']['image'].shape, rk=3)
            losses_refined = self.get_loss(target_main, all_fg_masks, refined_mlvl_preds,
                                           mlvl_selected_gt, noaug_flag=noaug_flag)
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
            # target_main, mlvl_preds, mlvl_ori_loc_preds = self.get_targets(data['main'], force=True)
            # if 'MOT20-01' in data['main']['image_id'][0] and ('000068' in data['main']['image_id'][0] or '000067' in data['main']['image_id'][0] or '000069' in data['main']['image_id'][0]):
            #     aaaaa = target_main[2][0]
            #     logger_print(data['main']['image_id'][0], aaaaa.shape, rk=3)
            #     mmmmm = torch.cat([p[2] for p in mlvl_preds], dim=1)[0]
            #     logger_print(mmmmm.shape, rk=3)
            #     canvas = mmmmm.new_zeros(mmmmm.shape[0])
            #     logger_print(canvas.shape, rk=3)
            #     amax = aaaaa.argmax(dim=1)
            #     logger_print(amax.shape, target_main[5][0].shape, rk=3)
            #     canvas[target_main[5][0]] = amax.float()
            #     torch.save(canvas, './idmap.%s.pkl' % (data['main']['image_id'][0].split('/')[-1],))
            #     torch.save(mmmmm, './idpred.%s.pkl' % (data['main']['image_id'][0].split('/')[-1],))
            #     for ii, one in enumerate(aaaaa):
            #         k = (one > 0.05).nonzero()
            #         if k == 12 or k == 7 or k == 8:
            #             logger_print(ii, k, rk=3)
            # losses_rpn = self.post_module.get_loss(
            #     target_main, mlvl_preds, mlvl_ori_loc_preds, noaug_flag=[True] * len(noaug_flag))
            # logger_print(losses_rpn, rk=1)
            # logger_print(data['main']['image'].shape, rk=3)
            # logger_print(data['main']['image_id'], rk=3)
            rpn_results = self.get_results(
                mlvl_preds,
                data['main']['roi_features'],
            )
            refined_results = self.get_results(
                refined_mlvl_preds,
                all_refined_feats,
            )
            # logger_print(refined_results['dt_bboxes'].shape, rk=3)
            # logger_print(refined_results['id_embeds'].shape, rk=3)
            # logger_print(refined_results['dt_bboxes'].mean(), rk=3)
            # logger_print(refined_results['id_embeds'].mean(), rk=3)
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
        # info_debug(mlvl_preds)
        with get_debugger().no_debug():
            results = self.post_module.predictor.predict(mlvl_preds, id_feats)
        # info_debug(results)
        # logger_print(results['id_info'])
        return results

    def get_loss(self, target, fg_masks, mlvl_preds, gt_indices, noaug_flag=None):
        # info_debug(mlvl_preds)
        lw_id = self.lw_id
        lw_det = self.lw_det
        if self.balanced_loss_weight == 'auto':
            lw_id = torch.exp(-lw_id)
            lw_det = torch.exp(-lw_det)
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
            if self.roi_features_mappings[idx] == 'id':
                if noaug_flag is None or not self.dismiss_aug:
                    valid = torch.cat([o[gti] for o, gti in zip(target[7], cat_gt_inds)], dim=0)
                else:
                    valid = torch.cat([(o[gti] & no) for o, gti, no in zip(target[7], cat_gt_inds, noaug_flag)], dim=0)
                # valid = torch.cat(target[7], dim=0)
                if self.post_module.all_reduce_norm and dist.is_initialized():
                    num_ids = self.post_module.get_ave_normalizer(valid)
                else:
                    num_ids = valid.sum().item()
                num_ids = max(num_ids, 1)
                loss = getattr(self, feat_name + '_loss')(pred[fg_masks][valid],
                                                          target_this[valid][:, 1:], normalizer_override=num_ids)
                # loss = loss * lw_id
            else:
                # info_debug([pred, fg_masks, pred[fg_masks], gt_indices, target[idx]])
                loss = getattr(self, feat_name + '_loss')(pred[fg_masks], target_this, normalizer_override=num_fgs)
                # loss = loss * lw_det
            losses[self.prefix + '.' + feat_name + '_loss'] = loss
        if self.refine_obj:
            losses[self.prefix + '.obj_loss'] = self.post_module.obj_loss(
                torch.cat(preds[3], dim=1).flatten(),
                fg_masks.flatten(),
                normalizer_override=num_fgs)  # * lw_det
        if self.balanced_loss_weight == 'auto':
            losses[self.prefix + '.lwnorm_loss'] = (self.lw_id + self.lw_det) * self.balance_scale
        return losses


@MODULE_ZOO_REGISTRY.register('relation_retina')
class RelationRetina(nn.Module):

    def __init__(self, relation_cfgs, retina_kwargs, inplanes, num_to_refine=1000, num_as_ref=500, share=False, init_prior=0.01):
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
        retina_kwargs.update({'inplanes': self.inplanes})
        self.post_module = MODULE_ZOO_REGISTRY.build(dict(
            type='retina_post_iou_w_id',
            kwargs=retina_kwargs,
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
        prior_prob = init_prior
        for relation_idx in self.relation_indices:
            idx = self.relation_indices[relation_idx]
            if self.roi_features_mappings[relation_idx] == 'id' or share:
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

    def get_targets(self, input, mode='main'):
        save_anchors = mode == 'main'
        mlvl_preds, mlvl_anchors, mlvl_shapes = self.post_module.prepare_preds(input, save_anchors=save_anchors)
        if self.training:
            targets_all_ = self.post_module.supervisor.get_targets(mlvl_anchors, input, mlvl_preds)
            targets_all = list(targets_all_)
            del targets_all[3]
            targets = []
            cum = 0
            for lvl_anchors in mlvl_anchors:
                num = lvl_anchors.size(0)
                targets.append([x[:, cum:cum + num] for x in targets_all])
                cum += num
            targets_all = targets_all_
        else:
            targets_all = None
            targets = None
        return targets_all, targets, mlvl_preds, mlvl_anchors, mlvl_shapes

    @torch.no_grad()
    def get_selected_indices(self, lvl_anchors, lvl_preds, image_info, lvl, mode='main'):
        # info_debug(lvl_preds)
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
        keeps = self.post_module.predictor.lvl_nms(lvl_anchors, lvl_preds, image_info, preserved=num)
        return keeps

    def get_top_feats(self, feats, indices, is_pred=False):
        # get_debugger()([feats, indices], tag='top')
        if feats is None:
            return None
        need_reshape = False
        if feats.dim() == 4:
            b, c, h, w = feats.shape
            if not is_pred:
                feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
                indices = indices // 2
            else:
                k = self.post_module.num_anchors
                c //= k
                feats = feats.permute(0, 2, 3, 1).reshape(b, h * w * k, c)
        else:
            if feats.dim() == 3:
                c = feats.size(2)
            else:
                need_reshape = True
                c = 1
                feats = feats.unsqueeze(-1)
        ret = torch.gather(feats, 1, indices.unsqueeze(-1).repeat(1, 1, c))
        if need_reshape and ret.size(-1) == 1:
            ret = ret.squeeze(-1)
        # info_debug(indices, prefix='indices')
        return ret

    def forward(self, data):
        # info_debug(data)
        self.vis = data.get('vis_flag', False)
        m = len(data['ref'])
        image_info_main = data['main']['image_info']
        image_info_ref = [o['image_info'] for o in data['ref']]
        if self.training:
            target_main, mlvl_target_main, mlvl_preds, mlvl_anchors, mlvl_shapes = self.get_targets(data['main'])
            # info_debug(data['main'])
            # info_debug(target_main)
            # info_debug(mlvl_preds)
            targets_ref, mlvl_targets_ref, mlvl_preds_ref, mlvl_anchors_ref, _ = map_transpose(
                [self.get_targets(ref, mode='ref') for ref in data['ref']])
        else:
            mlvl_preds, mlvl_anchors, _ = self.post_module.prepare_preds(data['main'])
            mlvl_preds_ref, mlvl_anchors_ref, _ = map_transpose(
                [self.post_module.prepare_preds(o, save_anchors=False) for o in data['ref']])
        mlvl_preds_activated = self.post_module.apply_activation(mlvl_preds)
        mlvl_preds_ref_activated = [self.post_module.apply_activation(o) for o in mlvl_preds_ref]
        all_refined_preds = []
        all_refined_feats = []
        original_lvl_preds = []
        all_relation_stuffs = {}
        all_target_main = []
        all_target_ref = []
        all_target_main_map = []
        all_selected_anchors = []
        for lvl_idx, lvl_c in enumerate(self.inplanes):
            original_preds_main = []
            refined_lvl_preds = []
            refined_lvl_feats = []
            main_roi_feats = data['main']['roi_features'][lvl_idx]
            all_target_main.append([])
            all_target_main_map.append({})
            all_target_ref.append([])
            with get_debugger().no_debug():
                selected_main = self.get_selected_indices(
                    mlvl_anchors[lvl_idx], mlvl_preds_activated[lvl_idx], image_info_main, lvl_idx)
                all_selected_anchors.append(self.get_top_feats(
                    mlvl_anchors[lvl_idx].unsqueeze(0).repeat(2, 1, 1), selected_main))
                selected_ref = [self.get_selected_indices(
                    mlvl_anchors_ref[i][lvl_idx], mlvl_preds_ref_activated[i][lvl_idx], image_info_ref[i], lvl_idx, mode='ref') for i in range(m)]
            for idx, roi_feat in enumerate(main_roi_feats):
                relation_idx = self.relation_indices.get(idx, -1)
                with get_debugger().no_debug():
                    roi_feat = self.get_top_feats(roi_feat, selected_main)
                if self.roi_features_mappings[idx] == 'loc':
                    c = self.roi_pred_dims[idx]
                    roi_preds = torch.gather(mlvl_preds[lvl_idx][idx], 1,
                                             selected_main.unsqueeze(-1).repeat(1, 1, c))
                else:
                    with get_debugger().no_debug():
                        roi_preds = self.get_top_feats(data['main']['preds'][lvl_idx][idx], selected_main, is_pred=True)
                if relation_idx < 0:
                    refined_lvl_preds.append(roi_preds)
                    original_preds_main.append(roi_preds)
                    refined_lvl_feats.append(roi_feat.permute(0, 2, 1).unsqueeze(-1))
                else:
                    with get_debugger().no_debug():
                        roi_feat_ref = [self.get_top_feats(data['ref'][i]['roi_features']
                                                           [lvl_idx][idx], selected_ref[i]) for i in range(m)]
                    roi_feat_ref = torch.cat(roi_feat_ref, dim=1)
                    if self.training:
                        target_this = mlvl_target_main[lvl_idx][idx]
                        # target_this = self.get_range(target_this, target_range_main[lvl_idx])
                        with get_debugger().no_debug():
                            target_this = self.get_top_feats(target_this, selected_main)
                        # target_ref = [torch.cat(o, dim=0) for o in zip(
                        #     *[self.get_range(targets_ref[i][idx], target_range_ref[i][lvl_idx]) for i in range(m)])]
                        # info_debug(mlvl_targets_ref)
                            target_ref_splits = [self.get_top_feats(
                                mlvl_targets_ref[i][lvl_idx][idx], selected_ref[i]) for i in range(m)]

                        target_ref = torch.cat(target_ref_splits, dim=1)
                        all_target_main_map[-1][idx] = target_this
                        if self.vis:
                            all_target_main[-1].append(target_this)
                            all_target_ref[-1].append(target_ref)
                        if self.roi_features_mappings[idx] != 'id':  # TODO: inspect
                            target_this = target_this - 1
                            target_ref = target_ref - 1
                        # if self.roi_features_mappings[idx] != 'id':
                        #     target_this = [torch.cat([o.new_zeros((o.size(0), 1)), o], dim=1) for o in target_this]
                        #     target_ref = [torch.cat([o.new_zeros((o.size(0), 1)), o], dim=1) for o in target_ref]
                        refined_feats, relation_stuffs = self.relation_modules[lvl_idx][relation_idx](roi_feat, roi_feat_ref, target_main=(
                            None, target_this), target_ref=(None, target_ref), original_preds=roi_preds)
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
            iou_pred = self.get_top_feats(
                data['main']['preds'][lvl_idx][3], selected_main, is_pred=True)
            refined_lvl_preds.append(iou_pred)
            original_preds_main.append(iou_pred)
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
            # info_debug([target_main, mlvl_preds])
            losses_rpn = self.post_module.get_loss(target_main, mlvl_preds, mlvl_shapes)
            losses_refined = self.get_loss(all_target_main_map, refined_mlvl_preds)
            losses = {}
            losses.update(all_relation_stuffs)
            losses.update(losses_rpn)
            losses.update(losses_refined)
            if self.vis:
                losses['refined_pred_main'] = refined_mlvl_preds
                losses['original_pred_main'] = original_lvl_preds
                losses['targets_main'] = (all_target_main, None)
                losses['original_pred_ref'] = []
                losses['targets_ref'] = (all_target_ref, None)
            # print(losses)
            return losses
        else:
            # print(mlvl_preds[0][1].shape, mlvl_preds[0][1][0, :10])
            # print(refined_mlvl_preds[0][1].shape, refined_mlvl_preds[0][1][0, :10])
            # info_debug(mlvl_preds)
            rpn_results = self.get_results(
                mlvl_anchors,
                mlvl_preds,
                data['main']['roi_features'],
                image_info_main,
                raw=True,
            )
            # info_debug(refined_mlvl_preds)
            refined_results = self.get_results(
                all_selected_anchors,
                refined_mlvl_preds,
                all_refined_feats,
                image_info_main,
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
    def get_results(self, mlvl_anchors, mlvl_preds, mlvl_feats, image_info, raw=False):
        id_feats = [mlvl_feats[lvl_idx][2] for lvl_idx in range(len(self.inplanes))]
        id_feats = [id_features.permute(0, 2, 3, 1).reshape(id_features.size(
            0), id_features.size(2) * id_features.size(3), -1) for id_features in id_feats]
        if raw:
            k = self.post_module.num_anchors
            id_feats = [id_features.unsqueeze(2).repeat(1, 1, k, 1).reshape(id_features.size(
                0), id_features.size(1) * k, -1) for id_features in id_feats]
        mlvl_preds = self.post_module.apply_activation(mlvl_preds)
        with get_debugger().no_debug():
            results = self.post_module.predictor.predict(mlvl_anchors, mlvl_preds, id_feats, image_info)
        return results

    def get_loss(self, target, mlvl_preds):
        # info_debug([target, mlvl_preds], prefix='loss')
        preds = map_transpose(mlvl_preds)
        losses = {}
        for idx in self.relation_indices:
            feat_name = self.roi_features_mappings[idx]
            pred = preds[idx]
            pred = torch.cat(pred, dim=1)
            target_this = torch.cat([o[idx] for o in target], dim=1)
            pos_normalizer = max(1, torch.sum((target_this > 0).float()).item())
            # info_debug([pred, target_this], prefix='%s: [%.6f]' % (feat_name, pos_normalizer))
            # info_debug([target_this > 0, pred, target_this], statistics=True)
            # logger_print(target_this[:, :20])
            if self.roi_features_mappings[idx] == 'id':
                loss = getattr(self, feat_name + '_loss')(pred[target_this > 0],
                                                          target_this[target_this > 0], normalizer_override=pos_normalizer)
            else:
                loss = getattr(self, feat_name + '_loss')(pred,
                                                          target_this, normalizer_override=pos_normalizer)
            losses[self.prefix + '.' + feat_name + '_loss'] = loss
        return losses
