from typing import AnyStr
import os
import torch
import yaml


model_to_convert = './pretrained/yolox_x_m20_sgd.pth'
config_path = './configs/tracking/sqga_yolox_x_m20_noaug.yaml'


def _rename_keys(state, modules, verbose=False):
    new_state = state.__class__()
    for k in state.keys():
        if k.startswith('module.'):
            mname = k[7:]
            prefix = 'module.'
        else:
            mname = k
            prefix = ''
        new_key = k
        found = False
        for module, typ, cfg in modules:
            if mname.startswith(module + '.'):
                found = True
                tmp_state = {}
                if typ == 'mot_wrapper':
                    wrapper_key = module + '.inner_module'
                    new_key = prefix + wrapper_key + mname[len(module):]
                    tmp_state = {new_key: state[k]}
                    sub_mods = [(wrapper_key, cfg['type'], cfg['kwargs'].get('cfg', None))]
                if typ == 'dual_wrapper':
                    wrapper_key = module + '.mod0'
                    new_key = prefix + wrapper_key + mname[len(module):]
                    tmp_state = {new_key: state[k]}
                    sub_mods = [(wrapper_key, cfg['type'], cfg['kwargs'].get('cfg', None))]
                    wrapper_key = module + '.mod1'
                    new_key = prefix + wrapper_key + mname[len(module):]
                    tmp_state.update({new_key: state[k]})
                    sub_mods.append((wrapper_key, cfg['type'], cfg['kwargs'].get('cfg', None)))
                before = list(tmp_state.keys())
                if cfg['type'].endswith('wrapper'):
                    tmp_state = _rename_keys(tmp_state, sub_mods)
                new_state.update(tmp_state)
            if found:
                if verbose:
                    print('rename %s => %s' % (before, list(tmp_state.keys())))
                break
        if not found:
            new_state[new_key] = state[k]
    return new_state


def convert_to_mot(model_path, config_path, destination=None):
    with open(config_path) as fd:
        cfg = yaml.safe_load(fd)
    wrapper_modules = []
    for module in cfg['net']:
        if module['type'].endswith('wrapper'):
            wrapper_modules.append((module['name'], module['type'], module['kwargs']['cfg']))
    if destination is None:
        fn, ext = os.path.splitext(model_path)
        destination = fn + '.mot2' + ext
    ckpt = torch.load(model_path, map_location='cpu')
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt
    else:
        state_dict = ckpt
    state_dict = _rename_keys(state_dict, wrapper_modules, verbose=False)
    ckpt['model'] = state_dict
    if 'ema' in ckpt:
        ema_state_dict = _rename_keys(ckpt['ema']['ema_state_dict'], wrapper_modules)
        ckpt['ema']['ema_state_dict'] = ema_state_dict
    torch.save(ckpt, destination)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        model_to_convert = sys.argv[1]
        config_path = sys.argv[2]
    convert_to_mot(model_to_convert, config_path)
