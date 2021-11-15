import torch
import torch.nn as nn
import torch.nn.functional as F

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.models.losses import build_loss
from .custom_mha import MultiheadAttention as CustomMHA
from ...utils.debug import info_debug

__all__ = ['SQGARelaton']


@MODULE_ZOO_REGISTRY.register('sqga')
class SQGARelaton(nn.Module):

    def __init__(self, embed_dim, loss, beta=0.1, np_ratio=3, ** kwargs):
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
        self.beta = beta
        self.np_ratio = np_ratio
        self.loss = build_loss(loss)

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None):
        b, m, c = ref_feats.shape
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
        attention, affinities = self.relation_layer(
            main_feats.permute(1, 0, 2), real_ref_feats.permute(1, 0, 2), real_ref_feats.permute(1, 0, 2), attn_mask=valid_mask)
        attention = attention.permute(1, 0, 2)
        refined_feats = self.norm(main_feats + self.dropout(attention))
        if self.training:
            qlt_loss = self.get_qlt_loss(qlt_main, original_preds, target_main)
            affinities = affinities[:, :, :m]
            sim_loss = self.get_sim_loss(affinities, target_main, target_ref)
            return refined_feats, {'sim_loss': sim_loss, 'qlt_loss': qlt_loss}  # {'affinities': affinities}
        else:
            return refined_feats, {}

    def get_qlt_loss(self, qlt_preds, ori_preds, target):
        # ori_preds_ = ori_preds
        # info_debug(qlt_preds)
        # info_debug(ori_preds)
        # info_debug(target)
        with torch.no_grad():
            ori_preds = torch.sigmoid(ori_preds).float()
            qlt_target = ori_preds.max(dim=2)[0]
            b, n, c = ori_preds.shape
            for i in range(b):
                fg_mask = target[0][i].clone()
                mask = target[1][i][:, 0] <= 0
                fg_mask[fg_mask.clone()] = mask
                diff = torch.abs(ori_preds[i][fg_mask] - target[1][i][mask][:, 1:])
                # print(ori_preds_[i][target[0][i]][:50], target[1][i][:50])
                if diff.size(0) > 0:
                    diff_max, _ = diff.max(dim=1)
                else:
                    diff_max = diff.new_zeros((0, ))
                qlt_target[i][fg_mask] = 1 - diff_max
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
            fg_labels_main = [(l.max(dim=1)[1]) if l.size(0) > 0
                              else l.new_zeros((0, ), dtype=torch.int64) for l in target_main[1]]
            fg_labels_ref = [(l.max(dim=1)[1]) if l.size(0) > 0
                             else l.new_zeros((0, ), dtype=torch.int64) for l in target_ref[1]]
            label_main[target_main[0]] = torch.cat(fg_labels_main, dim=0)
            label_ref[target_ref[0]] = torch.cat(fg_labels_ref, dim=0)
            sim_target = affs.new_zeros(affs.shape)
            mask = affs.new_ones(affs.shape, dtype=torch.bool)
            for i in range(b):
                eqm = label_main[i].reshape(-1, 1) == label_ref[i].reshape(1, -1)
                sim_target[i] = eqm
                unknown = ((label_main[i].reshape(-1, 1) == 0) | (label_ref[i].reshape(1, -1) == 0)) & eqm
                mask[i] = ~ unknown
        # print(affs.mean(), affs.numel(), sim_target.mean(), sim_target.numel())
        # print(affs[:1, :3])
        loss = self.loss(affs[mask], sim_target[mask])
        return loss
