import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.tasks.det.plugins.yolov5.models.components import ConvBnAct
from ..utils.debug import info_debug

__all__ = ['MOTModuleWrapper', 'DualModuleWrapper', 'AffiliatedNetWrapper', 'MultiTaskHeadsWrapper']


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
        if 'ref_cache' in args[0] and args[0]['ref_cache'][0]:
            return {'main': output_main, 'ref': args[0]['ref']}
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


@MODULE_ZOO_REGISTRY.register('affiliated_net_wrapper')
class AffiliatedNetWrapper(nn.Module):

    def __init__(self,
                 inplanes,
                 backbone_cfg,
                 neck_cfg,
                 task_tag,
                 act_fn={'type': 'Silu'},
                 normalize={'type': 'solo_bn'},
                 **kwargs):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.neck_cfg = neck_cfg
        self.task_tag = task_tag
        self.kwargs = kwargs
        self.backbone = MODULE_ZOO_REGISTRY.build(backbone_cfg)
        outplanes = self.backbone.get_outplanes()
        neck_cfg['kwargs']['inplanes'] = outplanes
        self.neck = MODULE_ZOO_REGISTRY.build(neck_cfg)
        self.outplanes = self.neck.get_outplanes()
        self.dims = [(main_dim + a_dim) for main_dim, a_dim in zip(inplanes, self.outplanes)]
        self.trans = nn.ModuleList()
        for in_dim, out_dim in zip(self.dims, self.outplanes):
            mod = ConvBnAct(
                in_dim, out_dim,
                kernel_size=1, stride=1, act_fn=act_fn, normalize=normalize,
            )
            self.trans.append(mod)

    def get_outplanes(self):
        return self.outplanes

    def forward(self, input):
        a_net_out = self.neck(self.backbone(input))
        main_net_features = input['features']
        a_net_features = a_net_out['features']
        task_features = [torch.cat([main_net_features_i.detach(), a_net_features_i], dim=1)
                         for main_net_features_i, a_net_features_i in zip(main_net_features, a_net_features)]
        task_features = [self.trans[i](feat_i) for i, feat_i in enumerate(task_features)]
        return {('features_w_%s' % (self.task_tag, )): task_features, 'features': main_net_features}


@MODULE_ZOO_REGISTRY.register('multi_task_heads_wrapper')
class MultiTaskHeadsWrapper(nn.Module):

    def __init__(self, heads, concat={}, num_point=1, width=0.375, inplanes=[256, 512, 1024], outplanes=256):
        super().__init__()
        self.tasks = list(heads.keys())
        self.cfgs = list(heads.values())
        for head_cfg in self.cfgs:
            if 'num_point' not in head_cfg['kwargs']:
                head_cfg['kwargs']['num_point'] = num_point
            if 'width' not in head_cfg['kwargs']:
                head_cfg['kwargs']['width'] = width
            if 'inplanes' not in head_cfg['kwargs']:
                head_cfg['kwargs']['inplanes'] = inplanes
            if 'outplanes' not in head_cfg['kwargs']:
                head_cfg['kwargs']['outplanes'] = outplanes
        self.concats = concat
        self.heads = nn.ModuleList([MODULE_ZOO_REGISTRY.build(head_cfg) for head_cfg in self.cfgs])

    def forward(self, input):
        ret = {}
        for task, head in zip(self.tasks, self.heads):
            out = head(input)
            for key in self.concats:
                if key in out:
                    if key not in ret:
                        ret[key] = {}
                    ret[key][task] = out[key]
                    del out[key]
            ret.update(out)
        for key in self.concats:
            if key in ret:
                tmp = []
                for one in self.concats[key]:
                    if not tmp:
                        tmp = [[] for _ in range(len(ret[key][one]))]
                    for i, v in enumerate(ret[key][one]):
                        tmp[i].extend(v)
                ret[key] = tmp
        # info_debug(ret)
        return ret

    def get_outplanes(self):
        return {task: head.get_outplanes() for task, head in zip(self.tasks, self.heads)}
