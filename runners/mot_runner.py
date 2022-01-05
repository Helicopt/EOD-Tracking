import os
import json
import torch
import copy
import time
from collections import deque
from eod.utils.general.registry_factory import RUNNER_REGISTRY
from eod.utils.env.gene_env import to_device
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.registry_factory import MODEL_HELPER_REGISTRY, MODULE_ZOO_REGISTRY
from eod.utils.env.dist_helper import barrier, all_gather, env

from eod.runner.base_runner import BaseRunner


__all__ = ['MOTFP16Runner']


@RUNNER_REGISTRY.register('motfp16')
class MOTFP16Runner(BaseRunner):

    def __init__(self, config, cache=False, estimate_time=False, **kwargs):
        super().__init__(config, **kwargs)
        self.cache = cache
        self.estimate_time = estimate_time

    def batch2device(self, batch):
        model_dtype = torch.float32
        if self.fp16 and self.backend == 'linklink':
            model_dtype = self.model.dtype
        if 'main' in batch:
            if batch['main']['image'].device != torch.device('cuda') or \
                    batch['main']['image'].dtype != model_dtype:

                batch = to_device(batch, device=torch.device('cuda'), dtype=model_dtype)
        else:
            if batch['image'].device != torch.device('cuda') or \
                    batch['image'].dtype != model_dtype:
                batch = to_device(batch, device=torch.device('cuda'), dtype=model_dtype)
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
            if self.fp16 and self.backend == 'linklink':
                model = model.half()
            if self.config['runtime']['special_bn_init']:
                for m in model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm):
                        m.eps = 1e-3
                        m.momentum = 0.03
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
        batch_size = self.data_loaders['test'].batch_sampler.batch_size
        assert batch_size == 1, 'only size 1 supported'
        if self.cache:
            if not hasattr(self, 'cache_queue'):
                self.cache_queue = deque(maxlen=self.data_loaders['test'].dataset.ref_num)
            batch['ref_cache'] = [False] * batch_size
            if batch['begin_flag'][0]:
                self.cache_queue.clear()
            else:
                batch['ref_cache'][0] = True
                tmp = list(self.cache_queue)
                batch['ref'] = tmp

        assert not self.model.training
        if self.estimate_time:
            torch.cuda.synchronize()
        model_begin = time.time()
        output = self.forward_model(batch)
        if self.estimate_time:
            model_end = time.time()
            torch.cuda.synchronize()
        if self.cache:
            needed_keys = ['roi_features', 'preds', 'strides', 'image_info', 'gt_bboxes']
            self.cache_queue.append({
                k: output[k] for k in needed_keys
            })
            while len(self.cache_queue) < self.cache_queue.maxlen:
                self.cache_queue.append({
                    k: output[k] for k in needed_keys
                })

        if self.tracker is not None:
            if self.estimate_time:
                torch.cuda.synchronize()
                tracker_begin = time.time()
            output = self.tracker(output)
            if self.estimate_time:
                tracker_end = time.time()
                torch.cuda.synchronize()
        else:
            output = [output]
        if self.estimate_time:
            logger.info('model time: %.3f, tracker time: %.3f' % (model_end - model_begin, tracker_end - tracker_begin))
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
        if self.config['saver'].get('save_result', False):
            os.makedirs(self.results_dir, exist_ok=True)
            res_file = os.path.join(self.results_dir, 'results.rk%d.txt' % env.rank)
            logger.info(f'saving partial inference results into {res_file}')
            writer = open(res_file, 'w')
            for results in all_results_list:
                for item in results:
                    print(json.dumps(item), file=writer)
                    writer.flush()
            writer.close()
        barrier()
        all_device_results_list = all_gather(all_results_list)
        return all_device_results_list
