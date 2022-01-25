# Standard Library
import json
import copy
import os
from collections import Counter, OrderedDict

# Import from third library
import numpy as np
import pandas
import yaml
from prettytable import PrettyTable

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.yaml_loader import IncludeLoader
from eod.utils.general.registry_factory import EVALUATOR_REGISTRY
from eod.tasks.det.data.metrics.custom_evaluator import CustomEvaluator, Metric

from ...utils.read_helper import read_lines

__all__ = ['TrackingEvaluator']


@EVALUATOR_REGISTRY.register('tracking')
class TrackingEvaluator(CustomEvaluator):
    """Calculate mAP&MR@FPPI for custom dataset"""

    def __init__(self,
                 gt_file,
                 num_classes,
                 iou_thresh,
                 formatter='{root}/{seq}/img1/{fr}.{ext}',
                 class_names=None,
                 fppi=np.array([0.1, 0.5, 1]),
                 metrics_csv='metrics.csv',
                 cmp_key='mDT91',
                 label_mapping=None,
                 ignore_mode=0,
                 ign_iou_thresh=0.5,
                 iou_types=['bbox'],
                 eval_class_idxs=[]):

        super(TrackingEvaluator, self).__init__(gt_file,
                                                num_classes,
                                                iou_thresh,
                                                metrics_csv=metrics_csv,
                                                label_mapping=label_mapping,
                                                ignore_mode=ignore_mode,
                                                ign_iou_thresh=ign_iou_thresh)
        self.formatter = formatter
        if len(eval_class_idxs) == 0:
            eval_class_idxs = list(range(1, num_classes))
        self.eval_class_idxs = eval_class_idxs

        self.fppi = np.array(fppi)
        self.class_names = class_names
        self.metrics_csv = metrics_csv
        if self.class_names is None:
            self.class_names = eval_class_idxs
        self.iou_types = iou_types
        self._idcnt = 0
        self._idmapping = {}
        self.cmp_key = cmp_key

    def next_id(self, seq_name, uid):
        if seq_name not in self._idmapping:
            self._idmapping[seq_name] = {}
        if uid < 0:
            return uid
        if uid not in self._idmapping[seq_name]:
            self._idcnt += 1
            self._idmapping[seq_name][uid] = self._idcnt
        return self._idmapping[seq_name][uid]

    def load_gts(self, gt_files):
        # maintain a dict to store original img information
        # key is image dir,value is image_height,image_width,instances
        original_gt = {}
        gts = {
            'bbox_num': Counter(),
            'gt_num': Counter(),
            'image_num': 0,
            'image_ids': list()
        }
        if not isinstance(gt_files, list):
            gt_files = [gt_files]
        for gt_file_idx, gt_file in enumerate(gt_files):
            gt_img_ids = set()
            for img in read_lines(gt_file):
                if self.label_mapping is not None:
                    img = self.set_label_mapping(img, gt_file_idx)
                image_id = img.get('virtual_filename', img['filename'])
                original_gt[image_id] = copy.deepcopy(img)
                gt_img_ids.add(image_id)
                gts['image_num'] += 1
                for idx, instance in enumerate(img.get('instances', [])):
                    instance['detected'] = False
                    # remember the original index within an image of annoated format so
                    # we can recover from distributed format into original format
                    is_ignore = instance.get('is_ignored', False)
                    instance['local_index'] = idx
                    label = instance.get('label', 0)
                    # ingore mode
                    # 0 indicates all classes share ignore region, label is set to -1
                    # 1 indicates different classes different ignore region, ignore label must be provided
                    # 2 indicates we ingore all ignore regions
                    if is_ignore and self.ignore_mode == 0:
                        label = -1
                    box_by_label = gts.setdefault(label, {})
                    box_by_img = box_by_label.setdefault(image_id, {'gts': []})
                    gt_by_img = box_by_img['gts']
                    gts['bbox_num'][label] += 1
                    if not is_ignore:
                        gt_by_img.append(instance)
                        gts['gt_num'][label] += 1
                    else:
                        ign_by_img = box_by_img.setdefault('ignores', [])
                        ign_by_img.append(instance)
            gts['image_ids'].append(gt_img_ids)
        return gts, original_gt

    def get_miss_rate(self, tp, fp, scores, image_num, gt_num, return_index=False):
        """
        input: accumulated tps & fps
        len(tp) == len(fp) == len(scores) == len(box)
        """
        N = len(self.fppi)
        maxfps = self.fppi * image_num
        mrs = np.zeros(N)
        fppi_scores = np.zeros(N)

        indices = []
        for i, f in enumerate(maxfps):
            idxs = np.where(fp > f)[0]
            if len(idxs) > 0:
                idx = idxs[0]  # the last fp@fppi
            else:
                idx = -1  # no fps, tp[-1]==gt_num
            indices.append(idx)
            mrs[i] = 1 - tp[idx] / gt_num
            fppi_scores[i] = scores[idx]
        if return_index:
            return mrs, fppi_scores, indices
        else:
            return mrs, fppi_scores

    def _np_fmt(self, x):
        s = "["
        for i in list(x):
            s += " {:.4f}".format(i)
        s += " ]"
        return s

    def export(self, output, ap, max_recall, fppi_miss, fppi_scores,
               recalls_at_fppi, precisions_at_fppi, tracking_prec):

        def compute_f1_score(recall, precision):
            return 2 * recall * precision / np.maximum(1, recall + precision)

        f1_score_at_fppi = compute_f1_score(recalls_at_fppi, precisions_at_fppi)

        fg_class_names = self.class_names
        if self.num_classes == len(self.class_names):
            fg_class_names = fg_class_names[1:]

        assert len(fg_class_names) == self.num_classes - 1

        csv_metrics = OrderedDict()
        csv_metrics['Class'] = fg_class_names
        csv_metrics['AP'] = ap[1:].tolist()
        csv_metrics['Recall'] = max_recall[1:].tolist()
        csv_metrics['TrackPrec'] = tracking_prec[1:].tolist()
        csv_metrics['DT91'] = ((1 - fppi_miss[1:, -1]) * 0.9 + tracking_prec[1:] * 0.1).tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['MR@FPPI={:.3f}'.format(fppi)] = fppi_miss[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Score@FPPI={:.3f}'.format(fppi)] = fppi_scores[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Recall@FPPI={:.3f}'.format(fppi)] = recalls_at_fppi[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['Precision@FPPI={:.3f}'.format(fppi)] = precisions_at_fppi[1:, idx].tolist()
        for idx, fppi in enumerate(self.fppi):
            csv_metrics['F1-Score@FPPI={:.3f}'.format(fppi)] = f1_score_at_fppi[1:, idx].tolist()

        # Summary
        mAP = np.mean(ap[1:]).tolist()
        m_rec = np.mean(max_recall[1:]).tolist()
        m_fppi_miss = np.mean(fppi_miss[1:], axis=0).tolist()
        m_score_at_fppi = np.mean(fppi_scores[1:], axis=0).tolist()
        m_rec_at_fppi = np.mean(recalls_at_fppi[1:], axis=0).tolist()
        m_prec_at_fppi = np.mean(precisions_at_fppi[1:], axis=0).tolist()
        m_f1_score_at_fppi = np.mean(f1_score_at_fppi[1:], axis=0).tolist()
        m_track_prec = np.mean(tracking_prec[1:], axis=0).tolist()

        csv_metrics['Class'].append('Mean')
        csv_metrics['AP'].append(mAP)
        csv_metrics['Recall'].append(m_rec)
        csv_metrics['TrackPrec'].append(m_track_prec)
        csv_metrics['DT91'].append(np.mean((1 - fppi_miss[1:, -1]) * 0.9 + tracking_prec[1:] * 0.1, axis=0).tolist())
        for fppi, mr, score, recall, precision, f1_score in zip(
                self.fppi, m_fppi_miss, m_score_at_fppi, m_rec_at_fppi, m_prec_at_fppi, m_f1_score_at_fppi):

            csv_metrics['MR@FPPI={:.3f}'.format(fppi)].append(mr)
            csv_metrics['Score@FPPI={:.3f}'.format(fppi)].append(score)
            csv_metrics['Recall@FPPI={:.3f}'.format(fppi)].append(recall)
            csv_metrics['Precision@FPPI={:.3f}'.format(fppi)].append(precision)
            csv_metrics['F1-Score@FPPI={:.3f}'.format(fppi)].append(f1_score)

        csv_metrics_table = pandas.DataFrame(csv_metrics)
        csv_metrics_table.to_csv(output, index=False, float_format='%.4f')
        return output, csv_metrics_table, csv_metrics

    def pretty_print(self, metric_table):
        columns = list(metric_table.columns)
        for col in columns:
            if col.startswith('Recall@') or col.startswith('Precision@') or col.startswith('F1-Score@'):
                del metric_table[col]

        table = PrettyTable()
        table.field_names = list(metric_table.columns)
        table.align = 'l'
        table.float_format = '.4'
        table.title = 'Evaluation Results with FPPI={}'.format(self.fppi)
        for index, row in metric_table.iterrows():
            table.add_row(list(row))
        logger.info('\n{}'.format(table))

    def get_cur_dts(self, dts_cls, gts_img_id_set):
        """ Only the detecton results of images on which class c is annotated are kept.
        This is necessary for federated datasets evaluation
        """
        cur_gt_dts = []
        for _, dt in enumerate(dts_cls):
            if dt['vimage_id'] in gts_img_id_set:
                cur_gt_dts.append(dt)
        return cur_gt_dts

    def get_cls_tp_fp(self, dts_cls, gts_cls):
        """
        Arguments:
            dts_cls (list): det results of one specific class.
            gts_cls (dict): ground truthes bboxes of all images for one specific class
        """
        fps, tps = np.zeros((len(dts_cls))), np.zeros((len(dts_cls)))
        matcheds = []
        for i, dt in enumerate(dts_cls):
            img_id = dt['vimage_id']
            seq_name, frame_id = self.parse_seq_info(img_id)
            dt_bbox = dt['bbox']
            m_iou, m_gt, m_iof = -1, -1, -1
            if img_id in gts_cls:
                gts = gts_cls[img_id]
                gt_bboxes = [g['bbox'] for g in gts['gts']]
                ign_bboxes = [g['bbox'] for g in gts['ignores']] if 'ignores' in gts else []
                m_iou, m_gt, m_iof = \
                    self.match(dt_bbox, gt_bboxes, ign_bboxes)
            if m_iou >= self.iou_thresh:
                if not gts['gts'][m_gt]['detected']:
                    tps[i] = 1
                    fps[i] = 0
                    gts['gts'][m_gt]['detected'] = True
                    gts['gts'][m_gt]['detected_score'] = dt['score']
                    gid = self.next_id(seq_name, gts['gts'][m_gt]['track_id'])
                    matcheds.append((dt['track_id'], gid))
                else:
                    fps[i] = 1
                    tps[i] = 0
                    matcheds.append((dt['track_id'], None))
            else:
                fps[i] = 1
                tps[i] = 0
                matcheds.append((dt['track_id'], None))
            if self.ignore_mode == 0:
                # if we share ignore region between classes, some images may only have ignore region
                # and the m_iof will be set to -1, this is wrong, so we need re calculate m_iof of ignore region

                igs_cls = self.gts.get(-1, {})
                if img_id in igs_cls:
                    igs = igs_cls.get(img_id, {})
                    ign_bboxes = [g['bbox'] for g in igs['ignores']] if 'ignores' in igs else []
                    m_iof = self.match_ig(dt_bbox, ign_bboxes)
            if self.ignore_mode == 2:
                m_iof = -1
            if fps[i] == 1 and m_iof >= self.ign_iou_thresh:
                fps[i] = 0
                tps[i] = 0

        return np.array(tps), np.array(fps), matcheds

    def parse_seq_info(self, filename):
        formatter = self.formatter
        if formatter == '{root}/{seq}/img1/{fr}.{ext}':
            frame_id = int(os.path.splitext(os.path.basename(filename))[0])
            seq_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))
        elif formatter == '{root}/{seq}/{fr}.{ext}':
            frame_id = int(os.path.splitext(os.path.basename(filename))[0])
            seq_name = os.path.basename(os.path.dirname(filename))
        else:
            seq_name = os.path.splitext(os.path.basename(filename))[0]
            frame_id = 0
        return seq_name, frame_id

    def get_track_prec(self, matchings):
        gt_rec = {}
        dt_rec = {}
        m = {}
        tot = 0
        for u, v in matchings:
            u = int(u)
            v = int(v) if v is not None else None
            if v is not None and v >= 0:
                gt_rec[v] = gt_rec.get(v, 0) + 1
                tot += 1
                if v not in m:
                    m[v] = {}
                m[v][u] = m[v].get(u, 0) + 1
            dt_rec[u] = dt_rec.get(u, 0) + 1
        ws = 0.
        for v in m:
            w = gt_rec[v] / tot
            cum_s = 0
            for u in m[v]:
                s = m[v][u] / (gt_rec[v] + dt_rec[u] - m[v][u])
                s = s * s
                cum_s += s
            ws += w * cum_s
        return ws

    def eval(self, res_file, res=None):
        metric_res = Metric({})
        for itype in self.iou_types:
            num_mean_cls = 0
            if not self.gt_loaded:
                self.gts, original_gt = self.load_gts(self.gt_file)
                self.gt_loaded = True
            else:
                self.reset_detected_flag()
            dts = self.load_dts(res_file, res)
            ap = np.zeros(self.num_classes)
            max_recall = np.zeros(self.num_classes)
            fppi_miss = np.zeros([self.num_classes, len(self.fppi)])
            fppi_scores = np.zeros([self.num_classes, len(self.fppi)])
            recalls_at_fppi = np.zeros([self.num_classes, len(self.fppi)])
            precisions_at_fppi = np.zeros([self.num_classes, len(self.fppi)])
            tracking_prec = np.zeros(self.num_classes)
            for class_i in self.eval_class_idxs:  # range(1, self.num_classes):
                sum_dt = self.gts['bbox_num'][class_i]
                sum_gt = self.gts['gt_num'][class_i]
                results_i = dts.get(class_i, [])
                results_i = sorted(results_i, key=lambda x: -x['score'])
                class_from = self.class_from.get(class_i, [0])
                gts_img_id_set = set()
                for data_idx in class_from:
                    gts_img_id_set = gts_img_id_set.union(self.gts['image_ids'][data_idx])
                if class_i not in self.gts:
                    self.gts[class_i] = {}
                cur_gt = self.get_cur_gt(self.gts[class_i], gts_img_id_set)
                # get the detection results from federated dataset for class_i
                results_i = self.get_cur_dts(results_i, gts_img_id_set)
                logger.info('sum_gt vs sum_dt: {} vs {}'.format(sum_gt, len(results_i)))
                if sum_gt > 0:
                    num_mean_cls += 1
                if sum_dt == 0 or len(results_i) == 0:
                    # ap[class_i] = 0.0
                    # max_recall[class_i] = 0.0
                    continue

                if itype == 'bbox':
                    tps, fps, matcheds = self.get_cls_tp_fp(results_i, cur_gt)
                drec = tps / max(1, sum_gt)
                tp = np.cumsum(tps)
                fp = np.cumsum(fps)
                rec = tp / sum_gt
                prec = tp / np.maximum(tp + fp, 1)
                for v in range(len(prec) - 2, -1, -1):
                    prec[v] = max(prec[v], prec[v + 1])
                scores = [x['score'] for x in results_i]
                image_num = len(gts_img_id_set)
                mrs, s_fppi, indices = self.get_miss_rate(tp, fp, scores, image_num, sum_gt, return_index=True)
                ap[class_i] = np.sum(drec * prec)
                max_recall[class_i] = np.max(rec)
                fppi_miss[class_i] = mrs
                fppi_scores[class_i] = s_fppi
                recalls_at_fppi[class_i] = rec[np.array(indices)]
                precisions_at_fppi[class_i] = prec[np.array(indices)]
                tracking_prec[class_i] = self.get_track_prec(matcheds)

            mAP = np.sum(ap[1:]) / num_mean_cls

            _, metric_table, csv_metrics = self.export(
                self.metrics_csv, ap, max_recall, fppi_miss, fppi_scores, recalls_at_fppi, precisions_at_fppi, tracking_prec)
            self.pretty_print(metric_table)

            mDT91 = csv_metrics['DT91'][-1]
            mTrackPrec = csv_metrics['TrackPrec'][-1]

            # metric_name = '{}_mAP:{}'.format(itype, self.iou_thresh)
            metric_name = self.cmp_key
            csv_metrics.update({'mAP': mAP})
            csv_metrics.update({'mTrackPrec': mTrackPrec})
            csv_metrics.update({metric_name: mDT91})
            metric_res.update(csv_metrics)
            metric_res.set_cmp_key(metric_name)
        return metric_res

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(name, help='subcommand for CustomDataset of mAP metric')
        subparser.add_argument('--gt_file', required=True, help='annotation file')
        subparser.add_argument('--res_file', required=True, help='results file of detection')
        subparser.add_argument('--num_classes',
                               type=int,
                               default=None,
                               help='number of classes including __background__ class')
        subparser.add_argument('--class_names',
                               type=lambda x: x.split(','),
                               default=None,
                               help='names of classes including __background__ class')
        subparser.add_argument('--iou_thresh',
                               type=float,
                               default=0.5,
                               help='iou thresh to classify true positives & false postitives')
        subparser.add_argument('--ign_iou_thresh',
                               type=float,
                               default=0.5,
                               help='ig iou thresh to ignore false postitives')
        subparser.add_argument('--metrics_csv',
                               type=str,
                               default='metrics.csv',
                               help='file to save evaluation metrics')
        subparser.add_argument('--img_root',
                               type=str,
                               default='None',
                               help='directory of images used to evaluate, only used for visualization')
        subparser.add_argument('--bad_case_analyser',
                               type=str,
                               default='manual_0.5',
                               help='choice of criterion for analysing bad case, format:{manual or fppi}_{[score]}')
        subparser.add_argument('--vis_mode',
                               type=str,
                               default=None,
                               choices=['all', 'fp', 'fn', None],
                               help='visualize fase negatives or fase positive or all')
        subparser.add_argument('--eval_class_idxs',
                               type=lambda x: list(map(int, x.split(','))),
                               default=[],
                               help='eval subset of all classes, 1,3,4'
                               )
        subparser.add_argument('--ignore_mode',
                               type=int,
                               default=0,
                               help='ignore mode, default as 0')
        subparser.add_argument('--config', type=str, default='', help='training config for eval')
        subparser.add_argument('--iou_types',
                               type=list,
                               default=['bbox'],
                               help='iou type to select eval mode')

        return subparser

    @classmethod
    def get_classes(self, gt_file):
        all_classes = set()

        # change to petrel
        with open(gt_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                labels = set([ins['label'] for ins in data['instances'] if ins['label'] > 0])
                all_classes |= labels
        all_classes = [0] + sorted(list(all_classes))
        class_names = [str(_) for _ in all_classes]
        print('class_names:{}'.format(class_names))
        return class_names

    @classmethod
    def from_args(cls, args):
        if args.config != '':   # load from training config
            cfg = yaml.load(open(args.config, 'r'), Loader=IncludeLoader)
            eval_kwargs = cfg['dataset']['test']['dataset']['kwargs']['evaluator']['kwargs']
            eval_kwargs['metrics_csv'] = args.metrics_csv
            return cls(**eval_kwargs)
        if args.num_classes is None:
            args.class_names = cls.get_classes(args.gt_file)
            args.num_classes = len(args.class_names)
        return cls(args.gt_file,
                   args.num_classes,
                   args.class_names,
                   args.iou_thresh,
                   img_root=args.img_root,
                   bad_case_analyser=args.bad_case_analyser,
                   vis_mode=args.vis_mode,
                   metrics_csv=args.metrics_csv,
                   ignore_mode=args.ignore_mode,
                   eval_class_idxs=args.eval_class_idxs,
                   ign_iou_thresh=args.ign_iou_thresh,
                   iou_types=args.iou_types,)
