from eod.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from eod.utils.general.log_helper import default_logger as logger
import torch
import numpy as np
from ..utils.debug import info_debug

__all__ = ['NoTracking']


class TrackState(object):

    def __init__(self):
        self.inited = False


@MODULE_ZOO_REGISTRY.register('no_tracking')
class NoTracking(object):

    masked_keys = set(['dt_bboxes', 'id_embeds'])
    common_keys = set(['strides'])
    ignored_keys = set(['image', 'features', 'preds', 'roi_features', 'refs'])

    def __init__(self, **kwargs):
        self.states = {}

    def initialize(self, state):
        state.inited = True

    def finalize(self, state):
        return NotImplemented

    def wrap(self, data):
        ret = {}
        for k, v in data.items():
            if k in self.masked_keys:
                if isinstance(v, torch.Tensor):
                    v = torch.cat([v.new_zeros((v.size(0), 1)), v], dim=1)
                elif isinstance(v, np.ndarray):
                    v = torch.from_numpy(v).unsqueeze(0)
                    v = torch.cat([v.new_zeros((v.size(0), 1)), v], dim=1)
            elif k in self.common_keys:
                pass
            elif k in self.ignored_keys:
                continue
            else:
                v = [v]
            ret[k] = v
        return ret

    def get_ith(self, data, i):
        mask = data['dt_bboxes'][:, 0] == i
        ret = {}
        for key in data:
            if key in self.masked_keys:
                ret[key] = data[key][mask]
            elif key in self.common_keys:
                ret[key] = data[key]
            elif key in self.ignored_keys:
                continue
            else:
                ret[key] = data[key][i]
        ret['dt_bboxes'] = ret['dt_bboxes'][:, 1:]
        return ret

    def __call__(self, data):
        batch_size = data['image'].shape[0]
        all_returns = []
        for i in range(batch_size):
            if i not in self.states:
                self.states[i] = TrackState()
            state = self.states[i]
            if 'begin_flag' in data and data['begin_flag'][i] or 'begin_flag' not in data:
                self.initialize(state)
            assert state.inited, 'track state not init'
            inputs = self.get_ith(data, i)
            output = self.wrap(self.forward(state, inputs))
            if output is not None:
                all_returns.append(output)
            if 'end_flag' in data and data['end_flag'][i]:
                output_final = self.finalize(state)  # noqa
                assert output_final is NotImplemented or isinstance(output_final, list),\
                    'list expected'
                if isinstance(output_final, list):
                    all_returns += list(map(self.wrap, output_final))
                self.states[i] = TrackState()

        return all_returns

    def forward(self, state, inputs):
        # bboxes = inputs['dt_bboxes']
        # id_embeds = inputs['id_embeds']
        # logger.info(inputs['image_id'])
        # logger.info('%s' % str(bboxes.shape))
        # logger.info('%s' % str(id_embeds.shape))
        return inputs
