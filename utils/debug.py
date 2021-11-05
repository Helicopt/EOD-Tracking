from collections import deque
import torch

from eod.utils.general.log_helper import default_logger as logger


def info_debug(x, prefix=''):
    if isinstance(x, torch.Tensor):
        logger.info('%s %s %s %s' % (prefix, x.shape, x.dtype, x.device))
    if isinstance(x, (list, tuple, deque)):
        for i, v in enumerate(x):
            info_debug(v, prefix=prefix + '[%d]' % i)
    if isinstance(x, dict):
        for i, v in x.items():
            info_debug(v, prefix=prefix + '.%s' % str(i))
