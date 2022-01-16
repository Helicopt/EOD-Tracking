# Import from third library
import torch
import torch.nn.functional as F

# Import from local
from eod.models.losses.loss import BaseLoss, _reduce
from eod.utils.general.registry_factory import LOSSES_REGISTRY
from eod.utils.general.fp16_helper import to_float32

from ...utils.debug import info_debug


__all__ = ['DistillKL', 'MSELoss']


@to_float32
def kl_loss(y_s, y_t, T=3, scale_type='linear', reduction='none', normalizer=None):
    """Distilling the Knowledge in a Neural Network"""
    if y_s.shape[1] == 1:
        extreme_inds = (y_s < -30).long()
        p_s_normal = (y_s / T).sigmoid().log()
        p_s_extreme = y_s / T
        p_s = torch.gather(torch.cat([p_s_normal, p_s_extreme], dim=1), 1, extreme_inds)
    else:
        p_s = F.log_softmax(y_s / T, dim=1)
    if y_t.shape[1] == 1:
        extreme_inds = (y_t < -30).long()
        p_t_normal = (y_t / T).sigmoid().log()
        p_t_extreme = y_t / T
        p_t = torch.gather(torch.cat([p_t_normal, p_t_extreme], dim=1), 1, extreme_inds)
    else:
        p_t = F.log_softmax(y_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, log_target=True, reduction='none') * (T**2)
    loss = _reduce(loss, reduction=reduction, normalizer=normalizer)
    return loss


@to_float32
def mse_loss(x, y, scale_type='linear', reduction='none', normalizer=None):
    loss = F.mse_loss(x, y, reduction='none')
    loss = _reduce(loss, reduction=reduction, normalizer=normalizer)
    return loss


@LOSSES_REGISTRY.register('kl_loss')
class DistillKL(BaseLoss):
    def __init__(self,
                 name='kl_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 temp=3.0,
                 scale_type='linear'):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - scale_type index (:obj:`str`): choice of 'linear', 'log'
        """
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.scale_type = scale_type
        self.key_fields = []
        self.temp = temp

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (:obj:`FloatTenosr`):
            - output (:obj:`FloatTenosr`): same shape as input
        """
        loss = kl_loss(input, target, T=self.temp, scale_type=self.scale_type, reduction=reduction,
                       normalizer=normalizer)
        return loss


@LOSSES_REGISTRY.register('mse_loss')
class MSELoss(BaseLoss):
    def __init__(self,
                 name='mse_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 scale_type='linear'):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - scale_type index (:obj:`str`): choice of 'linear', 'log'
        """
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.scale_type = scale_type
        self.key_fields = []

    def forward(self, input, target, reduction, normalizer=None):
        """
        Arguments:
            - input (:obj:`FloatTenosr`):
            - output (:obj:`FloatTenosr`): same shape as input
        """
        loss = mse_loss(input, target, scale_type=self.scale_type, reduction=reduction,
                        normalizer=normalizer)
        return loss
