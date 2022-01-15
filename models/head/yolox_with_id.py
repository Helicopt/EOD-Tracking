import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from eod.utils.model import accuracy as A  # noqa F401
from eod.utils.model.initializer import initialize_from_cfg
from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY

from eod.tasks.det.plugins.yolox.models.backbone.cspdarknet import DWConv

from eod.tasks.det.plugins.yolov5.models.components import ConvBnAct
from eod.utils.model.normalize import build_norm_layer
from eod.utils.model.act_fn import build_act_fn

from ..mot_module_wrapper import DualModuleWrapper
from ...utils.debug import info_debug


__all__ = ['YoloXHeadwID', 'YoloXHeadwIDLessShare', 'YoloXHeadwIDShare']


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        x_ = x.permute(0, 2, 3, 1)
        y = self.excitation(x_).permute(0, 3, 1, 2)
        return x * y


@MODULE_ZOO_REGISTRY.register('YoloXHeadwID')
class YoloXHeadwID(nn.Module):
    def __init__(self,
                 num_classes,
                 num_ids,
                 num_point=1,
                 width=0.375,
                 inplanes=[256, 512, 1024],
                 outplanes=256,
                 act_fn={'type': 'Silu'},
                 depthwise=False,
                 initializer=None,
                 class_activation='sigmoid',
                 normalize={'type': 'solo_bn'},
                 fuse_lvls_for_id=False,
                 fuse_mode='nearest',
                 se_block=0,
                 init_prior=0.01):
        super(YoloXHeadwID, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_levels = len(inplanes)
        self.num_point = num_point
        class_channel = {'sigmoid': -1, 'softmax': 0}[class_activation] + num_classes
        self.class_channel = class_channel
        self.fuse_id_lvls = fuse_lvls_for_id
        self.fuse_mode = fuse_mode

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.id_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else ConvBnAct
        if self.fuse_id_lvls:
            outplane = int(outplanes * width)
            self.id_convs = nn.Sequential(
                *[
                    Conv(
                        outplane * self.num_levels,
                        outplane,
                        kernel_size=3,
                        stride=1,
                        act_fn=act_fn,
                        normalize=normalize
                    ),
                    Conv(
                        outplane,
                        outplane,
                        kernel_size=3,
                        stride=1,
                        act_fn=act_fn,
                        normalize=normalize
                    ),
                ]
            )

        else:
            self.id_convs = nn.ModuleList()

        self.out_planes = []
        for i in range(self.num_levels):
            inplane = int(inplanes[i])
            outplane = int(outplanes * width)
            self.out_planes.append(outplane)
            self.stems.append(
                ConvBnAct(
                    inplane,
                    outplane,
                    kernel_size=1,
                    stride=1,
                    act_fn=act_fn,
                    normalize=normalize
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                        Conv(
                            outplane,
                            outplane,
                            kernel_size=3,
                            stride=1,
                            act_fn=act_fn,
                            normalize=normalize
                        ),
                    ]
                )
            )
            if not self.fuse_id_lvls:
                self.id_convs.append(
                    nn.Sequential(
                        *[
                            Conv(
                                outplane,
                                outplane,
                                kernel_size=3,
                                stride=1,
                                act_fn=act_fn,
                                normalize=normalize
                            ),
                            Conv(
                                outplane,
                                outplane,
                                kernel_size=3,
                                stride=1,
                                act_fn=act_fn,
                                normalize=normalize
                            ),
                        ]
                    )
                )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=outplane,
                    out_channels=self.num_point * class_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=outplane,
                    out_channels=self.num_point * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=outplane,
                    out_channels=self.num_point,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        self.id_preds = nn.Conv2d(
            in_channels=outplane,
            out_channels=self.num_point * num_ids,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.se_block = se_block
        if self.se_block > 0:
            self.id_se = SE_Block(outplane, self.se_block)
        if initializer is not None:
            initialize_from_cfg(self, initializer)
        self.initialize_biases(init_prior)

    def fuse_features(self, features):
        ret = []
        for lvl_i, feature in enumerate(features):
            target_shape = feature.shape[-2:]
            fused = []
            for lvl_j, other in enumerate(features):
                if lvl_i != lvl_j:
                    fused.append(F.interpolate(other, size=target_shape, mode=self.fuse_mode))
                else:
                    fused.append(feature)
            fused = torch.cat(fused, dim=1)
            ret.append(fused)
        return ret

    def forward_net(self, features, idx=0):
        mlvl_preds = []
        mlvl_roi_features = []
        stems = []
        for i in range(self.num_levels):
            feat = self.stems[i](features[i])
            stems.append(feat)
            cls_feat = self.cls_convs[i](feat)
            loc_feat = self.reg_convs[i](feat)
            if not self.fuse_id_lvls:
                id_feat = self.id_convs[i](feat)
                if self.se_block > 0:
                    id_feat = self.id_se(id_feat)
                id_pred = self.id_preds(id_feat)
            else:
                id_feat = None
                id_pred = None
            cls_pred = self.cls_preds[i](cls_feat)
            loc_pred = self.reg_preds[i](loc_feat)
            obj_pred = self.obj_preds[i](loc_feat)
            mlvl_preds.append([cls_pred, loc_pred, id_pred, obj_pred])
            mlvl_roi_features.append([cls_feat, loc_feat, id_feat])
        if self.fuse_id_lvls:
            id_feats = self.fuse_features(stems)
            for i in range(self.num_levels):
                id_feat = self.id_convs(id_feats[i])
                if self.se_block > 0:
                    id_feat = self.id_se(id_feat)
                id_pred = self.id_preds(id_feat)
                mlvl_roi_features[i][2] = id_feat
                mlvl_preds[i][2] = id_pred
                mlvl_roi_features[i] = tuple(mlvl_roi_features[i])
                mlvl_preds[i] = tuple(mlvl_preds[i])

        return mlvl_preds, mlvl_roi_features

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.num_point, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in [self.id_preds]:
            b = conv.bias.view(self.num_point, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.num_point, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds, mlvl_roi_features = self.forward_net(features)
        output = {}
        output['preds'] = mlvl_raw_preds
        output['roi_features'] = mlvl_roi_features
        return output

    def get_outplanes(self):
        return copy.copy(self.out_planes)


@MODULE_ZOO_REGISTRY.register('YoloXHeadwIDLessShare')
class YoloXHeadwIDLessShare(YoloXHeadwID):

    def __init__(self,
                 num_classes,
                 num_ids,
                 num_point=1,
                 width=0.375,
                 inplanes=[256, 512, 1024],
                 outplanes=256,
                 act_fn={'type': 'Silu'},
                 depthwise=False,
                 initializer=None,
                 class_activation='sigmoid',
                 normalize={'type': 'solo_bn'},
                 init_prior=0.01):
        super().__init__(num_classes,
                         num_ids,
                         num_point=num_point,
                         width=width,
                         inplanes=inplanes,
                         outplanes=outplanes,
                         act_fn=act_fn,
                         depthwise=depthwise,
                         initializer=initializer,
                         class_activation=class_activation,
                         normalize=normalize,
                         init_prior=init_prior)
        self.stems_id = nn.ModuleList()
        Conv = DWConv if depthwise else ConvBnAct

        for i in range(self.num_levels):
            inplane = int(inplanes[i])
            outplane = int(outplanes * width)
            self.stems_id.append(
                ConvBnAct(
                    inplane,
                    outplane,
                    kernel_size=1,
                    stride=1,
                    act_fn=act_fn,
                    normalize=normalize
                )
            )

    def forward_net(self, features, idx=0):
        mlvl_preds = []
        mlvl_roi_features = []
        for i in range(self.num_levels):
            feat = self.stems[i](features[i])
            feat_id = self.stems_id[i](features[i])
            cls_feat = self.cls_convs[i](feat)
            loc_feat = self.reg_convs[i](feat)
            id_feat = self.id_convs[i](feat_id)
            cls_pred = self.cls_preds[i](cls_feat)
            loc_pred = self.reg_preds[i](loc_feat)
            id_pred = self.id_preds(id_feat)
            obj_pred = self.obj_preds[i](loc_feat)
            mlvl_preds.append((cls_pred, loc_pred, id_pred, obj_pred))
            mlvl_roi_features.append((cls_feat, loc_feat, id_feat))
        return mlvl_preds, mlvl_roi_features


@MODULE_ZOO_REGISTRY.register('YoloXHeadwIDDual')
class YoloXHeadwIDDual(YoloXHeadwIDLessShare):

    def __init__(self, *args, **kwargs):
        self.det_idx = kwargs.pop('det_idx', 0)
        self.trk_idx = kwargs.pop('trk_idx', 1)
        super().__init__(*args, **kwargs)

    def forward_net(self, features, idx=0):
        mlvl_preds = []
        mlvl_roi_features = []
        for i in range(self.num_levels):
            feat = self.stems[i](features[DualModuleWrapper.get(self.det_idx)][i])
            feat_id = self.stems_id[i](features[DualModuleWrapper.get(self.trk_idx)][i])
            cls_feat = self.cls_convs[i](feat)
            loc_feat = self.reg_convs[i](feat)
            id_feat = self.id_convs[i](feat_id)
            cls_pred = self.cls_preds[i](cls_feat)
            loc_pred = self.reg_preds[i](loc_feat)
            id_pred = self.id_preds(id_feat)
            obj_pred = self.obj_preds[i](loc_feat)
            mlvl_preds.append((cls_pred, loc_pred, id_pred, obj_pred))
            mlvl_roi_features.append((cls_feat, loc_feat, id_feat))
        return mlvl_preds, mlvl_roi_features


@MODULE_ZOO_REGISTRY.register('YoloXHeadwIDShare')
class YoloXHeadwIDShare(nn.Module):
    def __init__(self,
                 num_classes,
                 num_ids,
                 num_point=1,
                 width=0.375,
                 inplanes=[256, 512, 1024],
                 outplanes=256,
                 act_fn={'type': 'Silu'},
                 depthwise=False,
                 initializer=None,
                 class_activation='sigmoid',
                 normalize={'type': 'solo_bn'},
                 num_conv=2):
        super(YoloXHeadwIDShare, self).__init__()
        self.prefix = self.__class__.__name__
        self.num_levels = len(inplanes)
        self.num_point = num_point
        self.num_ids = num_ids
        class_channel = {'sigmoid': -1, 'softmax': 0}[class_activation] + num_classes
        self.class_channel = class_channel

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.id_convs = nn.ModuleList()
        self.stems = nn.ModuleList()

        self.out_planes = []
        outplane = int(outplanes * width)
        for i in range(self.num_levels):
            inplane = int(inplanes[i])
            self.out_planes.append(outplane)
            self.stems.append(
                ConvBnAct(
                    inplane,
                    outplane,
                    kernel_size=1,
                    stride=1,
                    act_fn=act_fn,
                    normalize=normalize
                )
            )
        self.cls_convs = self.build(num_conv, outplane, outplane, normalize, act_fn)
        self.reg_convs = self.build(num_conv, outplane, outplane, normalize, act_fn)
        self.id_convs = self.build(num_conv, outplane, outplane, normalize, act_fn)
        self.cls_pred = nn.Conv2d(in_channels=outplane,
                                  out_channels=self.num_point * class_channel,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.reg_pred = nn.Conv2d(in_channels=outplane,
                                  out_channels=self.num_point * 4,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.id_pred = nn.Conv2d(in_channels=outplane,
                                 out_channels=self.num_point * num_ids,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.obj_pred = nn.Conv2d(in_channels=outplane,
                                  out_channels=self.num_point,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        if initializer is not None:
            initialize_from_cfg(self, initializer)
        self.initialize_biases(1e-2)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def initialize_biases(self, prior_prob):
        b = self.cls_pred.bias.view(self.num_point, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        b = self.id_pred.bias.view(self.num_point, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.id_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        b = self.obj_pred.bias.view(self.num_point, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def build(self, num_conv, input_planes, feat_planes, normalize, act_fn):
        mlvl_heads = nn.ModuleList()
        for lvl in range(self.num_levels):
            layers = []
            inplanes = input_planes
            for conv_idx in range(num_conv):
                if lvl == 0:
                    layers.append(nn.Sequential(
                        nn.Conv2d(inplanes, feat_planes, kernel_size=3, stride=1, padding=1, bias=False),
                        build_norm_layer(feat_planes, normalize)[1],
                        build_act_fn(act_fn)[1]))
                    inplanes = feat_planes
                else:
                    layers.append(nn.Sequential(
                        mlvl_heads[-1][conv_idx][0],
                        build_norm_layer(feat_planes, normalize)[1],
                        build_act_fn(act_fn)[1]))
                    inplanes = feat_planes
            mlvl_heads.append(nn.Sequential(*layers))
        return mlvl_heads

    def forward_net(self, features, idx=0):
        mlvl_preds = []
        mlvl_roi_features = []
        for i in range(self.num_levels):
            feat = self.stems[i](features[i])
            cls_feat = self.cls_convs[i](feat)
            loc_feat = self.reg_convs[i](feat)
            id_feat = self.id_convs[i](feat)
            cls_pred = self.cls_pred(cls_feat)
            loc_pred = self.reg_pred(loc_feat)
            id_pred = self.id_preds(id_feat)
            obj_pred = self.obj_pred(loc_feat)
            mlvl_preds.append((cls_pred, loc_pred, id_pred, obj_pred))
            mlvl_roi_features.append((cls_feat, loc_feat, id_feat))
        return mlvl_preds, mlvl_roi_features

    def forward(self, input):
        features = input['features']
        mlvl_raw_preds, mlvl_roi_features = self.forward_net(features)
        output = {}
        output['preds'] = mlvl_raw_preds
        output['roi_features'] = mlvl_roi_features
        return output

    def get_outplanes(self):
        return copy.copy(self.out_planes)
