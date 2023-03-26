from torch.nn import Module
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.env.gene_env import patterns_match
from ..utils.debug import info_debug


__all__ = ['LossSelector']


@MODULE_ZOO_REGISTRY.register('loss_selector')
class LossSelector(Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if not isinstance(self.cfg, list):
            self.cfg = [self.cfg]

    def match(self, key, pattern):
        return patterns_match(pattern, key)

    def forward(self, data):
        losses = {k for k in data if 'loss' in k}
        includes = set()
        excludes = set()
        for cond in self.cfg:
            key = cond['key']
            pattern = cond['pattern']
            if self.match(key, pattern):
                for k in data:
                    if self.match(k, cond['includes']):
                        if k in excludes:
                            excludes.remove(k)
                        includes.add(k)
                    if self.match(k, cond['excludes']):
                        if k in includes:
                            includes.remove(k)
                        excludes.add(k)
        if not excludes and includes:
            return {k: data[k] for k in losses if k in includes}
        else:
            return {k: data[k] for k in losses if k not in excludes}
