import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ...utils.debug import info_debug

__all__ = ['NonRelaton', 'VanillaRelaton']


@MODULE_ZOO_REGISTRY.register('nonrelation')
class NonRelaton(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None):
        # info_debug(main_feats)
        # info_debug(ref_feats)
        # info_debug(target_main)
        # info_debug(target_ref)
        # info_debug(original_preds)
        # print(target_main[0][0].nonzero().numel(), target_main[0][1].nonzero().numel())
        # print(target_ref[0][0].nonzero().numel(), target_ref[0][1].nonzero().numel())
        # fg_mask_main, target_main = target_main
        # fg_mask_ref, target_ref = target_ref
        return main_feats, {}


@MODULE_ZOO_REGISTRY.register('transformer')
class TransformerRelaton(nn.Module):

    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super().__init__()
        self.relation_layer = nn.Transformer(d_model=embed_dim, nhead=num_heads,
                                             num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=embed_dim)

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None, **kwargs):
        refined_feats = self.relation_layer(
            ref_feats.permute(1, 0, 2), main_feats.permute(1, 0, 2))
        refined_feats = refined_feats.permute(1, 0, 2)
        return refined_feats, {}


@MODULE_ZOO_REGISTRY.register('vanilla')
class VanillaRelaton(nn.Module):

    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super().__init__()
        self.relation_layer = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None, **kwargs):
        attention, affinities = self.relation_layer(
            main_feats.permute(1, 0, 2), ref_feats.permute(1, 0, 2), ref_feats.permute(1, 0, 2))
        attention = attention.permute(1, 0, 2)
        refined_feats = self.norm(main_feats + self.dropout(attention))
        # {'affinities': affinities}
        return refined_feats, {'sims': affinities.log() + math.log(affinities.size(2)), 'sim_target': affinities.new_full(affinities.shape, -1)}
