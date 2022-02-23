from eod.utils.general.registry_factory import HOOK_REGISTRY
from eod.utils.general.hook_helper import Hook
from eod.utils.env.dist_helper import env
from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.hook_helper import get_summary_writer_class
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
from ..utils.debug import info_debug


__all__ = ['MOTYoloxNoaug', 'RelMapVis']


@HOOK_REGISTRY.register('yolox_mot_noaug')
class MOTYoloxNoaug(Hook):
    def __init__(self, runner, no_aug_epoch=15, max_epoch=300, transformer=[], test_freq=1, save_freq=1):
        super(MOTYoloxNoaug, self).__init__(runner)
        self.no_aug_epoch = no_aug_epoch
        self.max_epoch = max_epoch
        self.transformer = transformer
        self.flag = False
        self.test_freq = test_freq
        self.save_freq = save_freq

    def before_forward(self, cur_iter, input):
        runner = self.runner_ref()
        if cur_iter >= runner.data_loaders['train'].get_epoch_size() * (self.max_epoch - self.no_aug_epoch):
            if not self.flag:
                logger.info(f"rebuild dataset transformer cfg {self.transformer}")
                runner.config['dataset']['train']['dataset']['kwargs']['transformer_noaug'] = self.transformer
                runner.config['dataset']['train']['dataset']['kwargs']['noaug_ratio'] = 1
                del runner.data_loaders, runner.data_iterators['train']
                import gc
                gc.collect()
                if not hasattr(self, 'data_iterators'):
                    runner.data_iterators = {}
                logger.info("rebuild dataloader")
                runner.build_dataloaders()
                runner.data_iterators['train'] = iter(runner.data_loaders["train"])
                try:
                    if env.world_size > 1:
                        if hasattr(runner.model.module, 'relation_module'):
                            runner.model.module.relation_module.post_module.use_l1 = True
                        else:
                            runner.model.module.yolox_post.use_l1 = True
                    else:
                        if hasattr(runner.model, 'relation_module'):
                            runner.model.relation_module.post_module.use_l1 = True
                        else:
                            runner.model.yolox_post.use_l1 = True
                except:  # noqa
                    pass
                runner.test_freq = self.test_freq
                runner.save_freq = self.save_freq
                self.flag = True


@HOOK_REGISTRY.register('relmap_vis')
class RelMapVis(Hook):

    def __init__(self, runner, logdir='log', summary_writer='tensorboard'):
        super(RelMapVis, self).__init__(runner)
        if env.is_master():
            self.summary_writer = get_summary_writer_class(summary_writer)(
                os.path.join(runner.work_dir, logdir, 'vis'))
        self._is_first = True

    def need_vis(self, cur_iter):
        runner = self.runner_ref()
        epoch_size = runner.data_loaders['train'].get_epoch_size()
        # info_debug(torch.zeros((1,)), prefix='<%s %s>' % (cur_iter, self._is_first))
        return cur_iter % epoch_size == 0 or self._is_first

    def get_vis_pred(self, data):
        # info_debug(data, statistics=True)
        clspred = data[0][0].sigmoid()
        if len(data) > 3:
            objness = data[3][0].sigmoid()
            scores = (objness * clspred).flatten()
        else:
            objness = scores = clspred[:, 0]
        bboxes = data[1][0].clone()
        idpred = data[2][0].sigmoid()
        bboxes[:, 0] -= bboxes[:, 2] / 2.
        bboxes[:, 1] -= bboxes[:, 3] / 2.
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        return objness, clspred, bboxes, idpred, scores

    def plot_affinity_matrix(self, aff, target, class_names_y, class_names_x):
        """
        Returns a matplotlib figure containing the plotted affinity matrix.

        Args:
            aff (array, shape = [n, n]): a affinity matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(28, 12))
        plt.imshow(aff, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("affinity matrix")
        plt.colorbar()
        tick_marks_y = np.arange(len(class_names_y))
        tick_marks_x = np.arange(len(class_names_x))
        plt.xticks(tick_marks_x, class_names_x, rotation=75)
        plt.yticks(tick_marks_y, class_names_y)

        # Compute the labels from the normalized affinity matrix.
        # exp_aff = np.exp(aff.astype('float'))
        # labels = np.around(aff / aff.sum(axis=1)[:, np.newaxis], decimals=2)
        labels = np.around(aff, decimals=3)
        if target is not None:
            target = target.astype('int')

        # Use white text if squares are dark; otherwise black.
        threshold = aff.max() / 2.
        for i, j in itertools.product(range(aff.shape[0]), range(aff.shape[1])):
            color = "white" if aff[i, j] > threshold else "black"
            if target is not None:
                plt.text(j, i - 0.1, labels[i, j], horizontalalignment="center", color=color)
                plt.text(j, i + 0.3, target[i, j], horizontalalignment="center", color=color)
            else:
                plt.text(j, i + 0.1, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('Main')
        plt.xlabel('Reference')
        return figure

    def before_forward(self, cur_iter, input):
        if self.need_vis(cur_iter) and env.is_master():
            input['vis_flag'] = True
        else:
            input['vis_flag'] = False

    def after_forward(self, cur_iter, output):
        if self.need_vis(cur_iter) and env.is_master():
            self._is_first = False
            image = output['image'][0][[2, 1, 0]] / 255
            # info_debug(output)
            for lvl_to_show in range(len(output['refined_pred_main'])):
                # info_debug(output)
                # refines
                obj_re, cls_re, bboxes_re, id_re, scores_re = self.get_vis_pred(
                    output['refined_pred_main'][lvl_to_show])
                obj_ori, cls_ori, bboxes_ori, id_ori, scores_ori = self.get_vis_pred(
                    output['original_pred_main'][lvl_to_show])
                num_to_show = 64
                num_to_ref = 32
                topscores, keep = torch.topk(scores_re, k=num_to_show)
                # info_debug(keep)
                # print(topscores.shape, keep.shape)
                bboxes_re = bboxes_re[keep]
                obj_re = obj_re[keep]
                cls_re = cls_re[keep]
                id_re = id_re[keep]
                # original
                obj_ori = obj_ori[keep]
                bboxes_ori = bboxes_ori[keep]
                cls_ori = cls_ori[keep]
                id_ori = id_ori[keep]
                # print(bboxes_re)
                self.summary_writer.add_image_with_boxes(
                    'pred.%d' % lvl_to_show, image,
                    bboxes_re,
                    labels=list(map(str, map(int, keep))),
                    global_step=cur_iter,
                )
                text_string = ''
                logs = []
                id_main = output['targets_main'][0][lvl_to_show][1][0]
                if output['targets_main'][1] is None:
                    fg_main = output['targets_main'][0][lvl_to_show][0][0] > 0
                    fg_ind = torch.arange(id_main.size(0)).to(fg_main.device)
                else:
                    fg_main = output['targets_main'][1][lvl_to_show][0]
                    fg_ind = fg_main.new_full((fg_main.size(0), ), -1, dtype=torch.int64)
                    fg_ind[fg_main] = torch.arange(id_main.size(0)).to(fg_main.device)
                ptr = 0
                for ni, i in enumerate(keep):
                    i = int(i)
                    ori_cls = float(cls_ori[ni])
                    re_cls = float(cls_re[ni])
                    ori_obj = float(obj_ori[ni])
                    re_obj = float(obj_re[ni])
                    ori_id = int(id_ori[ni].argmax())
                    ori_id_score = float(id_ori[ni][ori_id])
                    re_id = int(id_re[ni].argmax())
                    re_id_score = float(id_re[ni][re_id])
                    one_log = '%d: cls(%.3f/%.3f), obj(%.3f/%.3f), id(%d:%.3f/%d:%.3f)' %\
                        (i, ori_cls, re_cls, ori_obj, re_obj, ori_id + 1, ori_id_score, re_id + 1, re_id_score)
                    if fg_main[i]:
                        one_log += ' | fg'
                        assert fg_ind[i] >= 0
                        if id_main[fg_ind[i]].dtype == torch.int64:
                            id_ind = id_main[fg_ind[i]]
                        else:
                            id_ind = id_main[fg_ind[i]].argmax()
                        one_log += '(%d)' % int(id_ind)
                    else:
                        one_log += ' | bg'
                    text_string += one_log + '\n'
                    logs.append(one_log)
                self.summary_writer.add_text('diff_log', text_string, global_step=cur_iter)
                x_logs = []
                ptr = 0
                if output['targets_ref'][1] is None:
                    fgs = output['targets_ref'][0][lvl_to_show][0][0] > 0
                else:
                    fgs = output['targets_ref'][1][lvl_to_show][0]
                for i, fg in enumerate(fgs):
                    if not fg:
                        one_log = 'bg'
                    else:
                        cls_label = float(output['targets_ref'][0][lvl_to_show][0][0][ptr])
                        if isinstance(output['targets_ref'][0][lvl_to_show][1], torch.Tensor):
                            id_label = int(output['targets_ref'][0][lvl_to_show][1][0][ptr])
                            id_label_score = 1.
                        else:
                            id_label = int(output['targets_ref'][0][lvl_to_show][1][0][ptr].argmax())
                            id_label_score = float(output['targets_ref'][0][lvl_to_show][1][0][ptr].max())
                        ptr += 1
                        one_log = 'cls(%.3f), id(%d:%.3f)' % (cls_label, id_label, id_label_score)
                    x_logs.append(one_log)

                for i in range(2):
                    relmap_tag = 'relation.%d.%d.sims' % (lvl_to_show, i)
                    heatmap = output[relmap_tag][0]
                    heatmap_target = output.get('relation.%d.%d.sim_target' % (lvl_to_show, i), [None])[0]
                    heatmap = heatmap[keep].softmax(dim=1)
                    if heatmap_target is not None:
                        heatmap_target = heatmap_target[keep]
                    class_names_y = [x for xi, x in enumerate(logs)]
                    if heatmap.size(0) > 16:
                        heatmap = heatmap[:16]
                        if heatmap_target is not None:
                            heatmap_target = heatmap_target[:16]
                        class_names_y = class_names_y[:16]
                    if i == 0 or heatmap_target is None:
                        _, ref_idx = heatmap.sum(dim=0).topk(num_to_ref)
                    else:
                        mxs, _ = heatmap_target.max(dim=0)
                        _, ref_idx = mxs.topk(num_to_ref)
                    ref_idx = ref_idx.flatten()
                    class_names_x = [x_logs[int(k)] for k in ref_idx]
                    heatmap = heatmap[:, ref_idx]
                    heatmap = heatmap.detach().cpu().numpy()
                    if heatmap_target is not None:
                        heatmap_target = heatmap_target[:, ref_idx]
                        heatmap_target = heatmap_target.detach().cpu().numpy()
                    # info_debug([heatmap, heatmap_target])
                    fig = self.plot_affinity_matrix(heatmap, heatmap_target, class_names_y, class_names_x)
                    self.summary_writer.add_figure(relmap_tag, fig, global_step=cur_iter)
                self.summary_writer.add_image_with_boxes(
                    'ori.%d' % lvl_to_show, image,
                    bboxes_ori,
                    labels=list(map(str, map(int, keep))),
                    global_step=cur_iter,
                )
            try:
                self.summary_writer.flush()
            except Exception as e:
                pass


@HOOK_REGISTRY.register('custom_auto_save')
class CustomAutoSave(Hook):
    """save some epochs
    """

    def __init__(self, runner, save_epochs=[]):
        super(CustomAutoSave, self).__init__(runner)
        self.save_epochs = save_epochs

    def _save_ckpt(self, prefix='best'):
        cur_epoch = self.runner_ref().cur_epoch()
        cur_iter = self.runner_ref().cur_iter
        if env.is_master():
            if self.runner_ref().ema is not None:
                ema = self.runner_ref().ema.state_dict()
            else:
                ema = {}
            self.runner_ref().saver.save(epoch=cur_epoch,
                                         iter=cur_iter,
                                         lns=False,
                                         auto_save=prefix,
                                         metric_val=0,
                                         state_dict=self.runner_ref().model.state_dict(),
                                         optimizer=self.runner_ref().optimizer.state_dict(),
                                         ema=ema,
                                         lr_scheduler=self.runner_ref().lr_scheduler.state_dict())

    # def after_epoch(self, cur_epoch):
    #     self.cur_epoch = cur_epoch
    #     if self.cur_epoch in self.save_epochs:
    #         self._save_ckpt(f'e{self.cur_epoch}')
    #         logger.info(f'custom saved epoch: {self.cur_epoch}')

    def after_eval(self, metrics):
        if not self.runner_ref().model.training:
            return
        self.cur_epoch = self.runner_ref().cur_epoch()
        if self.cur_epoch in self.save_epochs:
            self._save_ckpt(f'e{self.cur_epoch}')
            logger.info(f'custom saved epoch: {self.cur_epoch}')
