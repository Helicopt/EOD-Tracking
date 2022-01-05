import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ..utils.debug import info_debug

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
        output_main = self.inner_module(*new_args, **new_kwargs)
        output_ref = []
        with torch.no_grad():
            for i in range(num_ref):
                new_args, new_kwargs = self.__get_input(args, kwargs, tag='ref', idx=i)
                output_ref.append(self.inner_module(*new_args, **new_kwargs))
        return {'main': output_main, 'ref': output_ref}


@MODULE_ZOO_REGISTRY.register('dual_wrapper')
class DualModuleWrapper(nn.Module):

    AUTO_GEN_FLAG = 'DUAL_MOD_AUTO_GEN_FLAG'
    AUTO_GEN_PREFIX = 'DUAL_MOD_##'

    def __sub_mod_get_outplanes(self):
        return self.mod0.get_outplanes()

    def __init__(self, cfg, cfg1=None, metas=[], **kwargs):
        super().__init__()
        cfg0 = cfg
        self.metas = set(metas)
        if cfg1 is None:
            cfg1 = copy.deepcopy(cfg0)
        for cfg in [cfg0, cfg1]:
            if 'kwargs' not in cfg:
                cfg['kwargs'] = dict()
            cfg['kwargs'].update(kwargs)
        self.mod0 = MODULE_ZOO_REGISTRY.build(cfg0)
        self.mod1 = MODULE_ZOO_REGISTRY.build(cfg1)
        self.n = 2
        if hasattr(self.mod0, 'get_outplanes'):
            setattr(self, 'get_outplanes', self.__sub_mod_get_outplanes)

    @staticmethod
    def get(idx):
        return '%s.%d' % (DualModuleWrapper.AUTO_GEN_PREFIX, idx)

    @staticmethod
    def __wrap_dict(arg, idx):
        tag = DualModuleWrapper.get(idx)
        if isinstance(arg, dict):
            if arg.get(DualModuleWrapper.AUTO_GEN_FLAG, False):
                return arg[tag]
            else:
                tmp = dict()
                for k, v in arg.items():
                    if isinstance(v, dict) and v.get(DualModuleWrapper.AUTO_GEN_FLAG, False):
                        tmp[k] = v[tag]
                    else:
                        tmp[k] = v
                return tmp
        else:
            return arg

    @staticmethod
    def __get_input(args, kwargs, idx=0):
        new_args = []
        new_kwargs = {}
        for arg in args:
            new_args.append(DualModuleWrapper.__wrap_dict(arg, idx))
        for kw in kwargs:
            value = kwargs[kw]
            new_kwargs[kw] = DualModuleWrapper.__wrap_dict(value, idx)
        return new_args, new_kwargs

    def forward(self, *args, **kwargs):
        ret = {}
        for i in range(self.n):
            new_args, new_kwargs = self.__get_input(args, kwargs, idx=i)
            mod = getattr(self, 'mod%d' % i)
            outdata = mod(*new_args, **new_kwargs)
            tag = DualModuleWrapper.get(i)
            if isinstance(outdata, dict):
                for k, v in outdata.items():
                    if k in self.metas:
                        if i == 0:
                            ret[k] = v
                        continue
                    if k not in ret:
                        ret[k] = {self.AUTO_GEN_FLAG: True}
                    ret[k][tag] = v
            else:
                ret[self.AUTO_GEN_FLAG] = True
                ret[tag] = outdata
        if ret.get(self.AUTO_GEN_FLAG, False):
            assert len(ret) == self.n + 1
        return ret
