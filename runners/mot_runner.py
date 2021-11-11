import torch
import copy
from eod.utils.general.registry_factory import RUNNER_REGISTRY
from eod.utils.env.gene_env import to_device
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY, MODULE_ZOO_REGISTRY
from eod.utils.env.dist_helper import barrier, all_gather, env

from eod.runner.fp16_runner import FP16Runner


__all__ = ['MOTFP16Runner']


@RUNNER_REGISTRY.register('motfp16')
class MOTFP16Runner(FP16Runner):

    def batch2device(self, batch):
        if batch['main']['image'].device != torch.device('cuda') or \
                batch['main']['image'].dtype != torch.float32:
            batch = to_device(batch, device=torch.device('cuda'), dtype=torch.float32)
        return batch

    def build_fake_model(self):
        '''
        pt_sync_bn can't be deepcopy, replace with solo bn for ema
        '''
        net_cfg = copy.deepcopy(self.config['net'])
        normalize = {"type": "solo_bn"}
        flag = False
        for n in net_cfg:
            if 'normalize' in n['kwargs']:
                if n['kwargs']['normalize']['type'] == 'pt_sync_bn':
                    flag = True
                    n['kwargs']['normalize'] = normalize
            if n['type'] == 'mot_wrapper':
                if 'normalize' in n['kwargs']['cfg']['kwargs'] \
                        and n['kwargs']['cfg']['kwargs']['normalize']['type'] == 'pt_sync_bn':
                    flag = True
                    n['kwargs']['cfg']['kwargs']['normalize'] = normalize
        if flag:
            logger.info("pt_sync_bn can't be deepcopy, replace with solo bn for ema, \
                load model state to fake model state")
            model_helper_type = self.config['runtime']['model_helper']['type']
            model_helper_kwargs = self.config['runtime']['model_helper']['kwargs']
            model_helper_ins = MODEL_HELPER_REGISTRY[model_helper_type]

            model = model_helper_ins(net_cfg, **model_helper_kwargs)
            if self.device == 'cuda':
                model = model.cuda()
            model.load(self.model.state_dict())
        else:
            model = self.model
        return model

    def build(self):
        super().build()
        self.build_trackers(self.config.get('tracker', None))

    def build_trackers(self, cfg):
        if cfg is not None:
            self.tracker = MODULE_ZOO_REGISTRY[cfg['type']](**cfg['kwargs'])
        else:
            self.tracker = None

    def forward_eval(self, batch):
        self._hooks('before_eval_forward', self.local_eval_iter(), batch)
        assert not self.model.training
        output = self.forward_model(batch)
        if self.tracker is not None:
            output = self.tracker(output)
        else:
            output = [output]
        if len(output) > 0:
            self._hooks('after_eval_forward', self.local_eval_iter(), output[0])
        else:
            self._hooks('after_eval_forward', self.local_eval_iter(), None)
        return output

    @torch.no_grad()
    def _inference(self):
        self.model.cuda().eval()
        test_loader = self.data_loaders['test']
        all_results_list = []
        for _ in range(test_loader.get_epoch_size()):
            batch = self.get_batch('test')
            output = self.forward_eval(batch)
            for one in output:
                dump_results = test_loader.dataset.dump(one)
                all_results_list.append(dump_results)
        barrier()
        all_device_results_list = all_gather(all_results_list)
        return all_device_results_list
