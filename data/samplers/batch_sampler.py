# Standard Library
import bisect
import itertools

# Import from third library
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler
from eod.utils.general.registry_factory import BATCH_SAMPLER_REGISTRY


__all__ = ['SequenceBatchSampler']


@BATCH_SAMPLER_REGISTRY.register('sequence')
class SequenceBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        assert batch_size == 1, 'Sequence batch mode not implemented'
        super(SequenceBatchSampler, self).__init__(sampler, batch_size, drop_last)
