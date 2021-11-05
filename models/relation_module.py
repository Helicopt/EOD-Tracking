import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ..utils.debug import info_debug

__all__ = ['SQGARelation']


@MODULE_ZOO_REGISTRY.register('sqga')
class SQGARelation(nn.Module):

    def __init__(self, stages, **kwargs):
        super().__init__()
        self.stages = stages

    def forward(self, data):
        # info_debug(data)
        return {
            'refined_features': data['main']['features'],
            'refined_preds': data['main']['preds'],
        }
