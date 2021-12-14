import copy
from eod.utils.general.log_helper import default_logger as logger
from eod.data.data_builder import BaseDataLoaderBuilder

from eod.utils.general.registry_factory import (
    SAMPLER_REGISTRY,
    DATA_BUILDER_REGISTY)


__all__ = ['CustomDataLoaderBuilder']


@DATA_BUILDER_REGISTY.register('custom')
class CustomDataLoaderBuilder(BaseDataLoaderBuilder):

    def build_sampler(self, cfg_sampler, dataset):
        """Only works for training. We support dist_test for testing for now
        """
        cfg_sampler = copy.deepcopy(cfg_sampler)
        cfg_sampler['kwargs']['dataset'] = dataset
        if self.phase == 'test':
            original_type = cfg_sampler['type']
            cfg_sampler['type'] += '_test'
            logger.warning('We use {} instead of {} for test'.format(cfg_sampler['type'], original_type))
        return SAMPLER_REGISTRY.build(cfg_sampler)
