from collections import deque
import torch

from eod.utils.general.log_helper import default_logger as logger


def info_debug(x, statistics=False, prefix=''):
    if isinstance(x, torch.Tensor):
        if statistics:
            if x.numel() > 0:
                xmean = float(x.float().mean())
                xstd = float(x.float().std())
                xmin = float(x.float().min())
                xmax = float(x.float().max())
            else:
                xmean = xstd = xmin = xmax = 0.0
            logger.info('%s %s %s %s | mean: %.6f std: %.6f min: %.6f max: %.6f' %
                        (prefix, x.shape, x.dtype, x.device, xmean, xstd, xmin, xmax))
        else:
            logger.info('%s %s %s %s' % (prefix, x.shape, x.dtype, x.device))
    if isinstance(x, (list, tuple, deque)):
        for i, v in enumerate(x):
            info_debug(v, statistics=statistics, prefix=prefix + '[%d]' % i)
    if isinstance(x, dict):
        for i, v in x.items():
            info_debug(v, statistics=statistics, prefix=prefix + '.%s' % str(i))


class NotEnabledClass(object):

    def __init__(self, cls):
        self.cls = cls

    def __enter__(self):
        self.cls.enable = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cls.enable = True


class Debugger(object):

    def __init__(self, name='', statistics=True):
        self._name = name
        self._show_sts = statistics
        self._enable = True

    def __call__(self, x, tag=''):
        if self.enable:
            info_debug(x, statistics=self._show_sts,
                       prefix=self._name + (('[%s]' % tag) if (tag != '') else ''))

    def no_debug(self):
        return NotEnabledClass(self)

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, v):
        self._enable = v


global debugger_pool
debugger_pool = dict()


def get_debugger(name='root', create_one=False, show_sts=True):
    global debugger_pool
    if name not in debugger_pool:
        if name == 'root' or create_one:
            debugger_pool[name] = Debugger(name, statistics=show_sts)
        else:
            raise KeyError(name)
    return debugger_pool[name]


def logger_print(*args):
    text = ' '.join(map(str, args))
    logger.info(text)
