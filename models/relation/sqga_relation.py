import torch
import torch.nn as nn
import torch.nn.functional as F

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.models.losses import build_loss
from .custom_mha import MultiheadAttention as CustomMHA
from ...utils.debug import info_debug, logger_print

__all__ = ['SQGARelaton']


@MODULE_ZOO_REGISTRY.register('sqga')
class SQGARelaton(nn.Module):

    def __init__(self, embed_dim, loss, detach=False, beta=0.1, np_ratio=3, ** kwargs):
        super().__init__()
        self.relation_layer = CustomMHA(embed_dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)
        self.qlt_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        self.u = nn.Parameter(torch.FloatTensor(1))
        self.u.data.fill_(-10)
        self.beta = beta
        self.np_ratio = np_ratio
        self.loss = build_loss(loss)
        self.vis = True
        self.detach = detach

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None):
        b, m, c = ref_feats.shape
        if self.detach:
            qlt_main = self.qlt_net(main_feats.detach())
            qlt_ref = self.qlt_net(ref_feats.detach())
        else:
            qlt_main = self.qlt_net(main_feats)
            qlt_ref = self.qlt_net(ref_feats)
        real_qlt_ref = torch.cat([qlt_ref, qlt_main], dim=1)
        real_ref_feats = torch.cat([ref_feats, main_feats], dim=1)
        qmasks = []
        for i in range(b):
            qlt_mask_i = qlt_main[i].reshape(-1, 1) < real_qlt_ref[i].reshape(1, -1) + self.beta
            qmasks.append(qlt_mask_i)
        qmasks = torch.stack(qmasks)
        valid_mask = ~qmasks
        if self.detach:
            rf = real_ref_feats.detach().permute(1, 0, 2)
            attention, affinities = self.relation_layer(
                main_feats.detach().permute(1, 0, 2), rf, rf, attn_mask=valid_mask)
        else:
            rf = real_ref_feats.permute(1, 0, 2)
            attention, affinities = self.relation_layer(
                main_feats.permute(1, 0, 2), rf, rf, attn_mask=valid_mask)
        attention = attention.permute(1, 0, 2)
        refined_feats = self.norm(main_feats + self.dropout(attention))
        sims = affinities[:, :, :m]
        if self.training:
            qlt_loss = self.get_qlt_loss(qlt_main, original_preds, target_main)
            sim_loss, sim_target = self.get_sim_loss(sims, target_main, target_ref)
            # {'affinities': affinities}
            stuff = {'sim_loss': sim_loss, 'qlt_loss': qlt_loss}
            if self.vis:
                stuff.update({'sims': sims, 'sim_target': sim_target, 'qlt_main': qlt_main, 'qlt_ref': real_qlt_ref})
        else:
            stuff = {}
            if self.vis:
                stuff.update({'sims': sims, 'qlt_main': qlt_main, 'qlt_ref': real_qlt_ref})
        return refined_feats, stuff

    def get_qlt_loss(self, qlt_preds, ori_preds, target):
        # ori_preds_ = ori_preds
        # info_debug(qlt_preds)
        # info_debug(ori_preds)
        # info_debug(target)
        with torch.no_grad():
            ori_preds = torch.sigmoid(ori_preds).float()
            qlt_target = 1 - ori_preds.max(dim=2)[0]
            b, n, c = ori_preds.shape
            if target[0] is not None:
                for i in range(b):
                    fg_mask = target[0][i].clone()
                    mask = target[1][i][:, 0] <= 1e-12
                    fg_mask[fg_mask.clone()] = mask
                    if isinstance(target[1], list):
                        diff = torch.abs(ori_preds[i][fg_mask] - target[1][i][mask][:, 1:])
                        # print(ori_preds_[i][target[0][i]][:50], target[1][i][:50])
                        if diff.size(0) > 0:
                            diff_max, _ = diff.max(dim=1)
                        else:
                            diff_max = diff.new_zeros((0, ))
                        qlt_target[i][fg_mask] = 1 - diff_max
                    else:
                        raise NotImplementedError()
            else:
                if target[1].dtype != torch.int64:
                    diff = torch.abs(ori_preds - target[1][:, :, 1:])
                    # print(ori_preds_[i][target[0][i]][:50], target[1][i][:50])
                    diff_max, _ = diff.max(dim=2)
                    qlt_target[i][fg_mask] = 1 - diff_max
                else:
                    mask = target[1] > 0
                    pos_preds = ori_preds[mask]
                    pos_targets = target[1][mask]
                    pos_scores = torch.gather(pos_preds, 1, pos_targets.unsqueeze(-1) - 1).squeeze(-1)
                    qlt_target[mask] = 1 - pos_scores
            qlt_target[qlt_target < 0] = 0
        # print(qlt_preds.mean(), qlt_target.mean())
        # print(qlt_preds[:1, :10])
        loss = self.loss(qlt_preds, qlt_target)
        return loss

    def get_sim_loss(self, affs, target_main, target_ref):
        eps = 1e-12
        # info_debug(affs)
        # info_debug(target_main)
        # info_debug(target_ref)
        # print(affs[0, :3, :3])
        with torch.no_grad():
            b, n, m = affs.shape
            label_main = affs.new_zeros((b, n), dtype=torch.int64)
            label_ref = affs.new_zeros((b, m), dtype=torch.int64)
            if isinstance(target_main[1], list):
                if target_main[0] is not None:
                    fg_labels_main = [(l.max(dim=1)[1]) if l.size(0) > 0
                                      else l.new_zeros((0, ), dtype=torch.int64) for l in target_main[1]]
                    fg_labels_ref = [(l.max(dim=1)[1]) if l.size(0) > 0
                                     else l.new_zeros((0, ), dtype=torch.int64) for l in target_ref[1]]
                    label_main[target_main[0]] = torch.cat(fg_labels_main, dim=0)
                    label_ref[target_ref[0]] = torch.cat(fg_labels_ref, dim=0)
                else:
                    label_main = target_main[1].max(dim=2)[1]
                    label_ref = target_ref[1].max(dim=2)[1]
            else:
                if target_main[0] is not None:
                    raise NotImplementedError()
                else:
                    label_main = target_main[1].clone()
                    label_ref = target_ref[1].clone()
                    label_main[label_main < 0] = 0
                    label_ref[label_ref < 0] = 0
            sim_target = affs.new_zeros(affs.shape)
            mask = affs.new_ones(affs.shape, dtype=torch.bool)
            for i in range(b):
                eqm = label_main[i].reshape(-1, 1) == label_ref[i].reshape(1, -1)
                sim_target[i] = eqm
                unknown = ((label_main[i].reshape(-1, 1) == 0) | (label_ref[i].reshape(1, -1) == 0)) & eqm
                mask[i] = ~ unknown
        # print(affs.mean(), affs.numel(), sim_target.mean(), sim_target.numel())
        # print(affs[:1, :3])
        # logger_print(sim_target[mask].nonzero().numel() / max(sim_target[mask].numel(), 1),
        #              sim_target[mask].numel(), sim_target.shape)
        loss = self.loss(affs[mask] + self.u, sim_target[mask], normalizer_override=max(sim_target[mask].sum(), 1))
        return loss, sim_target
