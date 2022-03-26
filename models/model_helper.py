# Standard Library
import copy
import importlib

# Import from third library
import torch
import torch.nn as nn
from collections import OrderedDict

from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.env.gene_env import to_device, patterns_match
from eod.models.model_helper import ModelHelper
from eod.utils.env.dist_helper import env
from ..utils.debug import info_debug

__all__ = ['MOTModelHelper']


@MODEL_HELPER_REGISTRY.register('mot')
class MOTModelHelper(ModelHelper):

    def __init__(self, cfg, nonref=False, keep_ref_image=False, **kwargs):
        super().__init__(cfg, **kwargs)
        self.nonref = nonref
        self.keep_ref_image = keep_ref_image

    def forward(self, input):
        """
        Note:
            Input should not be updated inplace !!!
            In Mimic task, input may be fed into teacher & student networks respectivly,
            inplace update may cause the input dict only keep the last forward results, which is unexpected.
        """
        input = copy.copy(input)
        if not self.quant:
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
                    if not ('ref_cache' in input and input['ref_cache'][0]):
                        kept_keys = ['strides', 'gt_bboxes', 'image_info']
                        if self.keep_ref_image:
                            kept_keys.append('image')
                        kept_dict = {k: u[k] for k in kept_keys if k in u}
                    else:
                        kept_dict = dict(u)
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

    @staticmethod
    def prefix_suffix_match(k1, k2):
        """
        Check if k2 is a prefix-suffix match of k1
        """
        splits = k2.split('.')
        for i in range(len(splits) + 1):
            prefix = '.'.join(splits[:i]) + '.'
            suffix = '.' + '.'.join(splits[i:])
            if k1.startswith(prefix) and k1.endswith(suffix):
                return len(splits)
        return 0

    def try_rename_unexpected_keys(self, model_state_dict, other_state_dict, missing_keys, unexpected_keys):
        """
        try to rename unexpected keys in other_state_dict
        """
        renamed = set()
        renamed_to = set()
        for k in missing_keys:
            max_matched = 0
            max_matched_key = None
            for k_ in unexpected_keys:
                if model_state_dict[k].shape != other_state_dict[k_].shape:
                    continue
                matched_length = self.prefix_suffix_match(k, k_)
                if matched_length > max_matched:
                    max_matched_key = k_
                    max_matched = matched_length
            if max_matched_key is not None:
                renamed.add(max_matched_key)
                renamed_to.add(k)
                other_state_dict[k] = other_state_dict[max_matched_key]
                logger.info(f'Rename {max_matched_key} to {k}')
        for k in renamed:
            other_state_dict.pop(k)
        return renamed, renamed_to

    def load(self, other_state_dict, strict=False):
        """
        1. load resume model or pretained detection model
        2. load pretrained clssification model
        """
        logger.info("Try to load the whole resume model or pretrained detection model...")
        other_state_dict.pop('model_cfg', None)
        model_keys = self.state_dict().keys()
        other_keys = other_state_dict.keys()
        shared_keys, unexpected_keys, missing_keys \
            = self.check_keys(model_keys, other_keys, 'model')
        renamed, renamed_to = self.try_rename_unexpected_keys(
            self.state_dict(), other_state_dict, missing_keys, unexpected_keys)
        shared_keys = shared_keys | set(renamed_to)
        missing_keys = missing_keys - set(renamed_to)
        unexpected_keys = unexpected_keys - set(renamed)

        # check shared_keys size
        new_state_dict = self.check_share_keys_size(self.state_dict(), other_state_dict, shared_keys, model_keys)
        # get share keys info from new_state_dict
        shared_keys, _, missing_keys \
            = self.check_keys(model_keys, new_state_dict.keys(), 'model')
        self.load_state_dict(new_state_dict, strict=strict)

        num_share_keys = len(shared_keys)
        if num_share_keys == 0:
            logger.info(
                'Failed to load the whole detection model directly, '
                'trying to load each part seperately...'
            )
            for mname, module in self.named_children():
                module_keys = module.state_dict().keys()
                other_keys = other_state_dict.keys()
                # check and display info module by module
                shared_keys, unexpected_keys, missing_keys, \
                    = self.check_keys(module_keys, other_keys, mname)
                module_ns_dict = self.check_share_keys_size(
                    module.state_dict(), other_state_dict, shared_keys, module_keys)
                shared_keys, _, missing_keys \
                    = self.check_keys(module_keys, module_ns_dict.keys(), mname)

                module.load_state_dict(module_ns_dict, strict=strict)

                self.display_info(mname, shared_keys, unexpected_keys, missing_keys)
                num_share_keys += len(shared_keys)
        else:
            self.display_info("model", shared_keys, unexpected_keys, missing_keys)
        return num_share_keys
