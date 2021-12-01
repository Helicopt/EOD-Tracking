from typing import AnyStr
import os
import torch
import yaml


model_to_convert = './pretrained/yolox_x_retina_sgd.pth'
config_path = './configs/tracking/base_retina_x_m20_noaug.yaml'


def _rename_keys(state, modules):
    new_state = state.__class__()
    for k in state.keys():
        if k.startswith('module.'):
            mname = k[7:]
            prefix = 'module.'
        else:
            mname = k
            prefix = ''
        new_key = k
        for module in modules:
            if mname.startswith(module + '.'):
                new_key = prefix + module + '.inner_module' + mname[len(module):]
                # print('rename %s => %s' % (k, new_key))
                break
        new_state[new_key] = state[k]
    return new_state


def convert_to_mot(model_path, config_path, destination=None):
    with open(config_path) as fd:
        cfg = yaml.safe_load(fd)
    wrapper_modules = []
    for module in cfg['net']:
        if module['type'] == 'mot_wrapper':
            wrapper_modules.append(module['name'])
    if destination is None:
        fn, ext = os.path.splitext(model_path)
        destination = fn + '.mot' + ext
    ckpt = torch.load(model_path, map_location='cpu')
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt
    else:
        state_dict = ckpt
    state_dict = _rename_keys(state_dict, wrapper_modules)
    ckpt['model'] = state_dict
    if 'ema' in ckpt:
        ema_state_dict = _rename_keys(ckpt['ema']['ema_state_dict'], wrapper_modules)
        ckpt['ema']['ema_state_dict'] = ema_state_dict
    torch.save(ckpt, destination)


convert_to_mot(model_to_convert, config_path)
