import torch
from eod.utils.general.registry_factory import RUNNER_REGISTRY
from eod.utils.env.gene_env import to_device

from eod.runner.fp16_runner import FP16Runner


__all__ = ['MOTFP16Runner']


@RUNNER_REGISTRY.register('motfp16')
class MOTFP16Runner(FP16Runner):

    def batch2device(self, batch):
        if batch['main']['image'].device != torch.device('cuda') or batch['main']['image'].dtype != torch.float32:
            batch = to_device(batch, device=torch.device('cuda'), dtype=torch.float32)
        return batch
