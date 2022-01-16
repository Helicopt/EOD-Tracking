import torch
import torch.nn as nn
import torch.nn.functional as F
from eod.models.losses import build_loss

import math

from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ...utils.debug import info_debug

__all__ = ['SoftLabelKD']


@MODULE_ZOO_REGISTRY.register('soft_label')
class SoftLabelKD(nn.Module):

    def __init__(self, mimic_keys, mimic_loss, **kwargs):
        super().__init__()
        if isinstance(mimic_keys, str):
            mimic_keys = [mimic_keys]
        self.mimic_keys = set(mimic_keys)
        self.loss = build_loss(mimic_loss)
        self.prefix = self.__class__.__name__

    def forward(self, output, output_teacher):
        losses = {}
        # info_debug(output, statistics=True, prefix='student_all')
        # info_debug(output_teacher, statistics=True, prefix='teacher_all')
        for k in self.mimic_keys:
            if k == 'cls_pred':
                tensor_s = [lvl[0] for lvl in output['preds']]
                tensor_t = [lvl[0] for lvl in output_teacher['preds']]
            elif k == 'loc_pred':
                tensor_s = [lvl[1] for lvl in output['preds']]
                tensor_t = [lvl[1] for lvl in output_teacher['preds']]
            elif k == 'obj_pred':
                tensor_s = [lvl[3] for lvl in output['preds']]
                tensor_t = [lvl[2] for lvl in output_teacher['preds']]
            elif k == 'cls_feat':
                tensor_s = [lvl[0] for lvl in output['roi_features']]
                tensor_t = [lvl[0] for lvl in output_teacher['roi_features']]
            elif k == 'loc_feat':
                tensor_s = [lvl[1] for lvl in output['roi_features']]
                tensor_t = [lvl[1] for lvl in output_teacher['roi_features']]
            else:
                assert False, 'Unknown key [%s] for kd' % k
            # info_debug(tensor_s, prefix='stu')
            # info_debug(tensor_t, prefix='tea')
            tensor_s = torch.cat([lvl.permute(0, 2, 3, 1).reshape(-1, lvl.shape[1]) for lvl in tensor_s], dim=0)
            tensor_t = torch.cat([lvl.permute(0, 2, 3, 1).reshape(-1, lvl.shape[1]) for lvl in tensor_t], dim=0)
            tag = 'loss_mimic_%s' % k
            losses[self.prefix + '.' + tag] = self.loss(tensor_s, tensor_t)
        return losses
