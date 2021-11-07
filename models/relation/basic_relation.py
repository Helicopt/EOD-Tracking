import torch
import torch.nn as nn

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ...utils.debug import info_debug

__all__ = ['NonRelaton', 'VanillaRelaton', 'SQGARelaton']


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
        fg_mask_main, target_main = target_main
        fg_mask_ref, target_ref = target_main
        return main_feats, {}


@MODULE_ZOO_REGISTRY.register('vanilla')
class VanillaRelaton(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None):
        fg_mask_main, target_main = target_main
        fg_mask_ref, target_ref = target_main
        return main_feats, {}


@MODULE_ZOO_REGISTRY.register('sqga')
class SQGARelaton(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None):
        fg_mask_main, target_main = target_main
        fg_mask_ref, target_ref = target_main
        return main_feats, {}
