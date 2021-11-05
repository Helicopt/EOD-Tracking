import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

__all__ = ['MOTModuleWrapper']


@MODULE_ZOO_REGISTRY.register('mot_wrapper')
class MOTModuleWrapper(nn.Module):

    def __sub_mod_get_outplanes(self):
        return self.inner_module.get_outplanes()

    def __init__(self, cfg, **kwargs):
        super().__init__()
        if 'kwargs' not in cfg:
            cfg['kwargs'] = dict()
        cfg['kwargs'].update(kwargs)
        self.inner_module = MODULE_ZOO_REGISTRY.build(cfg)
        if hasattr(self.inner_module, 'get_outplanes'):
            setattr(self, 'get_outplanes', self.__sub_mod_get_outplanes)

    @staticmethod
    def __get_input(args, kwargs, tag='main', idx=0):
        new_args = []
        new_kwargs = {}
        for arg in args:
            if isinstance(arg, dict) and 'main' in arg and 'ref' in arg:
                if tag == 'main':
                    new_args.append(arg[tag])
                else:
                    new_args.append(arg[tag][idx])
            else:
                new_args.append(arg)
        for kw in kwargs:
            value = kwargs[kw]
            if isinstance(value, dict) and 'main' in value and 'ref' in value:
                if tag == 'main':
                    new_kwargs[kw] = value[tag]
                else:
                    new_kwargs[kw] = value[tag][idx]
            else:
                new_kwargs[kw] = value
        return new_args, new_kwargs

    @staticmethod
    def __get_num_ref(args, kwargs):
        num = None
        for arg in args:
            if isinstance(arg, dict):
                if 'main' in arg and 'ref' in arg:
                    if num is None:
                        num = len(arg['ref'])
                    else:
                        assert num == len(arg['ref'])
        for kw in kwargs:
            value = kwargs[kw]
            if isinstance(value, dict):
                if 'main' in value and 'ref' in value:
                    if num is None:
                        num = len(value['ref'])
                    else:
                        assert num == len(value['ref'])
        return num

    def forward(self, *args, **kwargs):
        num_ref = self.__get_num_ref(args, kwargs)
        new_args, new_kwargs = self.__get_input(args, kwargs, tag='main')
        output_main = self.inner_module(*new_args, **kwargs)
        output_ref = []
        with torch.no_grad():
            for i in range(num_ref):
                new_args, new_kwargs = self.__get_input(args, kwargs, tag='ref', idx=i)
                output_ref.append(self.inner_module(*new_args, **new_kwargs))
        return {'main': output_main, 'ref': output_ref}
