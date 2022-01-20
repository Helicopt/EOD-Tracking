from .tracking_evaluator import TrackingEvaluator

import os
import numpy as np
import sys
import pandas as pd
import time
from collections import Counter, OrderedDict
import tempfile

from eod.utils.general.log_helper import default_logger as logger
from eod.utils.general.yaml_loader import IncludeLoader
from eod.utils.general.registry_factory import EVALUATOR_REGISTRY
from eod.tasks.det.data.metrics.custom_evaluator import CustomEvaluator, Metric

import configparser

try:
    import trackeval
except ImportError:
    trackeval = None

_default_trackeval_cfg = {'METRICS': ['HOTA', 'CLEAR', 'Identity'],
                          'THRESHOLD': 0.5,
                          'BENCHMARK': 'MOT17',
                          'USE_PARALLEL': True,
                          'NUM_PARALLEL_CORES': 5}


@EVALUATOR_REGISTRY.register('TrackEval')
class TrackEval(TrackingEvaluator):
    def __init__(self,
                 gt_file,
                 num_classes,
                 iou_thresh,
                 write_path='data/eval_result',
                 tracker_name='trk',
                 track_eval=_default_trackeval_cfg,
                 fast_eval=False,
                 group_by=None,
                 **kwargs):
        assert trackeval is not None, 'trackeval module not found. please install before using this evaluator'
        super(TrackEval, self).__init__(
            gt_file,
            num_classes,
            iou_thresh,
            **kwargs)
        self.tracker_name = tracker_name
        self.write_path = write_path
        self.track_eval_cfg = track_eval
        self.fast_eval = fast_eval
        self.group_by = group_by

    def build_trackeval_evaluator(self, _cfg):
        def update_value(_cfg, out_cfg):
            for key, val in _cfg.items():
                key = key.upper()
                if key in out_cfg.keys():
                    out_cfg[key] = val
            return out_cfg

        self.trackeval_eval_config = update_value(_cfg, self.get_default_trackeval_config())
        self.trackeval_eval_config['DISPLAY_LESS_PROGRESS'] = False
        self.trackeval_dataset_config = update_value(_cfg, self.get_default_track_eval_dataset_config())
        self.trackeval_metric_config = {key: value for key, value in _cfg.items() if key in ['METRICS', 'THRESHOLD']}
        self.trackeval_config = {**self.trackeval_eval_config,
                                 ** self.trackeval_dataset_config, **self.trackeval_metric_config}

        self.trackeval_evaluator = trackeval.Evaluator(self.trackeval_eval_config)
        self.trackeval_metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
            if metric.get_name() in self.trackeval_metric_config['METRICS']:
                self.trackeval_metrics_list.append(metric(self.trackeval_metric_config))
        if len(self.trackeval_metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')

    @staticmethod
    def get_default_trackeval_config():
        """Returns the default config values for evaluation"""
        code_path = os.path.abspath('.')
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': True,
        }
        return default_config

    def get_default_track_eval_dataset_config(self):
        """Default class config values"""
        code_path = os.path.abspath('.')
        default_config = {
            'GT_FOLDER': os.path.join(code_path, self.write_path, 'gt'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, self.write_path, 'trackers'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': '',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
            'SKIP_SPLIT_FOL': True,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def get_track_prec(self, dt, gt):
        # 'dump_result', gt & dt
        # process gt
        write_path = self.write_path
        os.makedirs(write_path, exist_ok=True)

        gt_write_path = os.path.join(write_path, 'gt')
        os.makedirs(gt_write_path, exist_ok=True)
        gt_writer = {}
        seq_list_path = os.path.join(gt_write_path, 'list.txt')
        seqmaps_writer = open(seq_list_path, 'w')
        seqmaps_writer.writelines('name\n')
        seq_name_list = []
        gt_sequence_names = set()
        for filename, targets in gt.items():
            seq_name, frame_id = self.parse_seq_info(filename)
            seq_path = os.path.join(gt_write_path, seq_name, 'gt')
            os.makedirs(seq_path, exist_ok=True)
            if seq_name not in gt_writer:
                gt_writer[seq_name] = open(os.path.join(seq_path, f'gt.txt'), 'w')
                seqmaps_writer.writelines(f'{seq_name}\n')
                seq_name_list.append(seq_name)
                gt_sequence_names.add(seq_name)
                try:
                    config = configparser.ConfigParser()
                    # config.read(os.path.join(gt_write_path, seq_name, 'seqinfo.ini'))
                    config.add_section("Sequence")
                    config.set("Sequence", "name", "seq_name")
                    config.set("Sequence", "imDir", 'img1')
                    config.set("Sequence", "frameRate", str(30))
                    config.set("Sequence", "seqLength", str(10000))
                    config.set("Sequence", "imWidth", str(-1))
                    config.set("Sequence", "imHeight", str(-1))
                    config.write(open(os.path.join(gt_write_path, seq_name, 'seqinfo.ini'), "w"))
                # seqLength=600
                # config.set("Sequence","imWidth",fina)
                # imHeight=1080
                except Exception as e:
                    print(e)

            gt_targets = targets['gts']
            for target in gt_targets:
                track_id = target['track_id']
                x1, y1, x2, y2 = target['bbox']
                target_line = f'{frame_id},{track_id},{x1},{y1},{x2},{y2},1,1,1\n'
                gt_writer[seq_name].writelines(target_line)
        for f in gt_writer:
            gt_writer[f].close()
        seqmaps_writer.close()

        # process dt
        logger.info('tracker_name: {}'.format(self.tracker_name))
        dt_write_path = os.path.join(write_path, 'trackers', self.tracker_name)
        os.makedirs(dt_write_path, exist_ok=True)
        dt_writer = {}

        # avoid tracker result is empty
        for seq_name in gt_sequence_names:
            logger.info('seq_name: {}'.format(seq_name))
            dt_writer[seq_name] = open(os.path.join(dt_write_path, f'{seq_name}.txt'), 'w')

        for target in dt:
            filename = target['vimage_id']
            seq_name, frame_id = self.parse_seq_info(filename)
            if seq_name not in dt_writer:
                logger.info('seq_name: {}'.format(seq_name))
                dt_writer[seq_name] = open(os.path.join(dt_write_path, f'{seq_name}.txt'), 'w')
            track_id = target['track_id']
            x1, y1, x2, y2 = target['bbox']
            target_line = f'{frame_id},{track_id},{x1},{y1},{x2},{y2},1,1,1\n'
            dt_writer[seq_name].writelines(target_line)
        for f in dt_writer:
            dt_writer[f].close()

        self.trackeval_dataset_config['TRACKERS_TO_EVAL'] = [self.tracker_name]

        if self.group_by is None or not isinstance(self.group_by, (list, tuple)) or not self.group_by:

            self.trackeval_dataset_config['SEQMAP_FILE'] = seq_list_path
            track_eval_dataset_list = [trackeval.datasets.MotChallenge2DBox(self.trackeval_dataset_config)]

            output_res, output_msg = self.trackeval_evaluator.evaluate(
                track_eval_dataset_list, self.trackeval_metrics_list)

            res = output_res['MotChallenge2DBox'][self.tracker_name]['COMBINED_SEQ']['pedestrian']

            metric = {}
            for item in self.trackeval_metric_config['METRICS']:
                if item not in res:
                    continue
                if item == 'HOTA':
                    metric['HOTA(0)'] = res[item]['HOTA(0)']
                else:
                    metric.update(res[item])
            return metric
        else:
            metric = {}
            for i, rep in enumerate(self.group_by):
                sub_list = []
                for seq_name_i in seq_name_list:
                    if seq_name_i.startswith(rep):
                        sub_list.append(seq_name_i)
                if sub_list:
                    sub_list_path = os.path.join(gt_write_path, 'list_sub%d.txt' % i)
                    with open(sub_list_path, 'w') as fd:
                        fd.write('name\n')
                        for one in sub_list:
                            fd.write('%s\n' % one)
                self.trackeval_dataset_config['SEQMAP_FILE'] = sub_list_path
                track_eval_dataset_list = [trackeval.datasets.MotChallenge2DBox(self.trackeval_dataset_config)]

                output_res, output_msg = self.trackeval_evaluator.evaluate(
                    track_eval_dataset_list, self.trackeval_metrics_list)

                res = output_res['MotChallenge2DBox'][self.tracker_name]['COMBINED_SEQ']['pedestrian']
                for item in self.trackeval_metric_config['METRICS']:
                    if item not in res:
                        continue
                    if item == 'HOTA':
                        metric['HOTA(0)@%s' % rep] = res[item]['HOTA(0)']
                    else:
                        for k in res[item]:
                            metric['%s@%s' % (k, rep)] = res[item][k]
            metric['HOTA(0)'] = metric['HOTA(0)@%s' % self.group_by[0]]
            return metric

    def export_fast(self, output, tracking_prec):

        fg_class_names = self.class_names
        if self.num_classes == len(self.class_names):
            fg_class_names = fg_class_names[1:]

        assert len(fg_class_names) == self.num_classes - 1

        csv_metrics = OrderedDict()
        csv_metrics['Class'] = fg_class_names
        for item in tracking_prec[1].keys():
            csv_metrics[item] = [cls_res[item] for cls_res in tracking_prec[1:]]

        csv_metrics['Class'].append('Mean')
        for item in tracking_prec[1].keys():
            csv_metrics[item].append(np.mean([cls_res[item] for cls_res in tracking_prec[1:]], axis=0).tolist())
        csv_metrics_table = pd.DataFrame(csv_metrics)
        csv_metrics_table.to_csv(output, index=False, float_format='%.4f')
        return output, csv_metrics_table, csv_metrics

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

        for item in tracking_prec[1].keys():
            csv_metrics[item] = [cls_res[item] for cls_res in tracking_prec[1:]]

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

        csv_metrics['Class'].append('Mean')
        csv_metrics['AP'].append(mAP)
        csv_metrics['Recall'].append(m_rec)

        for item in tracking_prec[1].keys():
            csv_metrics[item].append(np.mean([cls_res[item] for cls_res in tracking_prec[1:]], axis=0).tolist())

        for fppi, mr, score, recall, precision, f1_score in zip(
                self.fppi, m_fppi_miss, m_score_at_fppi, m_rec_at_fppi, m_prec_at_fppi, m_f1_score_at_fppi):

            csv_metrics['MR@FPPI={:.3f}'.format(fppi)].append(mr)
            csv_metrics['Score@FPPI={:.3f}'.format(fppi)].append(score)
            csv_metrics['Recall@FPPI={:.3f}'.format(fppi)].append(recall)
            csv_metrics['Precision@FPPI={:.3f}'.format(fppi)].append(precision)
            csv_metrics['F1-Score@FPPI={:.3f}'.format(fppi)].append(f1_score)

        csv_metrics_table = pd.DataFrame(csv_metrics)
        csv_metrics_table.to_csv(output, index=False, float_format='%.4f')
        return output, csv_metrics_table, csv_metrics

    def eval(self, res_file, res=None):
        tracker_name = self.tracker_name + '-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.write_path = tempfile.mkdtemp(prefix=tracker_name, suffix='trackeval', dir=self.write_path)
        self.build_trackeval_evaluator(self.track_eval_cfg)

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
            tracking_prec = [{}] * self.num_classes
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

                if itype == 'bbox' and not self.fast_eval:
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
                    tracking_prec[class_i]['track_prec'] = super().get_track_prec(matcheds)
                tracking_prec[class_i].update(self.get_track_prec(results_i, cur_gt))

            if not self.fast_eval:
                mAP = np.sum(ap[1:]) / num_mean_cls

                _, metric_table, csv_metrics = self.export(
                    self.metrics_csv, ap, max_recall, fppi_miss, fppi_scores, recalls_at_fppi, precisions_at_fppi, tracking_prec)
            else:
                _, metric_table, csv_metrics = self.export_fast(
                    self.metrics_csv, tracking_prec)
            self.pretty_print(metric_table)

            metric_name = 'HOTA(0)'
            for key in csv_metrics.keys():
                if key == 'Class':
                    continue
                metric_res[key] = csv_metrics[key][-1]
            metric_res.set_cmp_key(metric_name)

        return metric_res
