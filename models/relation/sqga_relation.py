import torch
import torch.nn as nn
import torch.nn.functional as F

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ...utils.debug import info_debug

__all__ = ['SQGARelaton']


@MODULE_ZOO_REGISTRY.register('sqga')
class SQGARelaton(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, main_feats, ref_feats, target_main=None, target_ref=None, original_preds=None):
        if target_main is not None:
            fg_mask_main, target_main = target_main
            fg_mask_ref, target_ref = target_ref
        return main_feats, {}
