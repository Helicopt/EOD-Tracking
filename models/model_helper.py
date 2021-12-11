# Standard Library
import copy
import importlib

# Import from third library
import torch
import torch.nn as nn
from collections import OrderedDict

from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY
from eod.utils.env.gene_env import to_device, patterns_match
from eod.models.model_helper import ModelHelper
from eod.utils.env.dist_helper import env

__all__ = ['MOTModelHelper']


@MODEL_HELPER_REGISTRY.register('mot')
class MOTModelHelper(ModelHelper):

    def __init__(self, cfg, nonref=False, **kwargs):
        super().__init__(cfg, **kwargs)
        self.nonref = nonref

    def forward(self, input):
        """
        Note:
            Input should not be updated inplace !!!
            In Mimic task, input may be fed into teacher & student networks respectivly,
            inplace update may cause the input dict only keep the last forward results, which is unexpected.
        """
        if not self.quant:
            input = copy.copy(input)
            if input['main']['image'].device != self.device or input['main']['image'].dtype != self.dtype:
                input = to_device(input, device=self.device, dtype=self.dtype)
        for submodule in self.children():
            if self.nonref:
                if 'main' in input and 'ref' in input:
                    main = input['main']
                    del input['main']
                    del input['ref']
                    input.update(main)
            output = submodule(input)
            if 'main' in output and 'ref' in output:
                input['main'].update(output['main'])
                for u, v in zip(input['ref'], output['ref']):
                    kept_keys = ['strides', 'gt_bboxes', 'image_info']
                    kept_dict = {k: u[k] for k in kept_keys if k in u}
                    u.clear()
                    u.update(v)
                    u.update(kept_dict)
                del output['main']
                del output['ref']
            else:
                if 'main' in input and 'ref' in input:
                    main = input['main']
                    del input['main']
                    del input['ref']
                    input.update(main)
                input.update(output)
        return input
