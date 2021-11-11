from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .no_tracking import NoTracking
from eod.utils.general.log_helper import default_logger as logger
import torch
import numpy as np
from ..utils.debug import info_debug

__all__ = ['MotionAppearanceOnlineTracker']


@MODULE_ZOO_REGISTRY.register('ma_online')
class MotionAppearanceOnlineTracker(NoTracking):

    def __init__(self, output_thr=0.5, **kwargs):
        super().__init__()
        self.output_thr = output_thr

    def initialize(self, state):
        super().initialize(state)
        state.tracklets = []

    def forward(self, state, inputs):
        info_debug(inputs)
        return inputs
