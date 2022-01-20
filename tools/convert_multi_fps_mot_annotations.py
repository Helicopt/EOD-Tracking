from typing import List
import json
import os
import sys
import configparser
import PIL.Image as Image
import argparse

from senseTk.common import TrackSet, Det

DATA_ROOT = '/mnt/lustre/share/fengweitao/MOT20'

skips = [1, 2, 4, 8, 16, 25, 36, 50, 75]


def generate_indices(n, skip=1):
    ret = []
    r = n % skip
    reverse = False
    for i in range(skip):
        tmp = [(j * skip + i) for j in range(n // skip + (i < r))]
        if reverse:
            ret.extend(tmp[::-1])
        else:
            ret.extend(tmp)
        reverse = not reverse
    assert len(set(ret)) == n
    return ret


def convert_annos(seqs, split, half=False, gap=30):
    '''
    convert mot20 annotations
    '''
    formatter = '{root}/{seq}/img1/{fr}.{ext}'
    all_data = []
    half_data_0 = []
    half_data_1 = []
    for seq in seqs:
        seq_dir = os.path.join(DATA_ROOT, split, seq)
        gt_path = os.path.join(seq_dir, 'gt', 'gt.txt')
        if os.path.exists(gt_path):
            ground_truth = TrackSet(gt_path)
        else:
            ground_truth = None
        image_probe = Image.open(os.path.join(seq_dir, 'img1', '000001.jpg'))
        image_height, image_width = image_probe.height, image_probe.width
        inifile = os.path.join(seq_dir, 'seqinfo.ini')
        iniconfig = configparser.ConfigParser()
        iniconfig.read(inifile)
        min_fr = 1
        max_fr = int(iniconfig['Sequence']['seqLength'])
        original_fps = float(iniconfig['Sequence']['frameRate'])
        suffix = '_multi_framerate'
        frames = list(range(min_fr, max_fr + 1))
        for skip in skips:
            indices = generate_indices(len(frames), skip=skip)
            virtual_seqname = 'S-%d-%s' % (skip, seq)
            half0_cnt = 0
            half1_cnt = 0
            for i, frame_id in enumerate(indices):
                frame_id = frames[frame_id]
                img = os.path.join(seq_dir, 'img1', '%06d.jpg' % frame_id)
                instances = []
                if ground_truth:
                    gtdets: List[Det] = sorted(ground_truth[frame_id], key=lambda x: x.uid)
                    for gtdet in gtdets:
                        if gtdet.status != 1:
                            continue
                        one = {
                            'is_ignored': False,
                            'bbox': [gtdet.x1, gtdet.y1, gtdet.x2, gtdet.y2],
                            'label': 1,
                            'track_id': gtdet.uid,
                            'vis_rate': gtdet.conf,
                        }
                        instances.append(one)
                data = {
                    'filename': os.path.abspath(img),
                    'instances': instances,
                    'image_width': image_width,
                    'image_height': image_height,
                    'formatter': formatter,
                    'virtual_filename': os.path.join('/', 'virtual_path', virtual_seqname, '%06d.jpg' % (i + min_fr))
                }
                all_data.append(data)
                half_ = (min_fr + max_fr) // 2
                data = dict(data)
                if frame_id <= half_:
                    data['virtual_filename'] = os.path.join(
                        '/', 'virtual_path', virtual_seqname, '%06d.jpg' % (half0_cnt + 1))
                    half0_cnt += 1
                    half_data_0.append(data)
                elif frame_id > half_ + gap:
                    data['virtual_filename'] = os.path.join(
                        '/', 'virtual_path', virtual_seqname, '%06d.jpg' % (half1_cnt + 1))
                    half1_cnt += 1
                    half_data_1.append(data)
    anno_dir = os.path.join(DATA_ROOT, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    def dump_row(x): return json.dumps(x) + '\n'
    with open(os.path.join(anno_dir, split + suffix + '.json'), 'w') as fd:
        fd.writelines(map(dump_row, all_data))
    if half:
        with open(os.path.join(anno_dir, split + suffix + '_half.json'), 'w') as fd:
            fd.writelines(map(dump_row, half_data_0))
        with open(os.path.join(anno_dir, split + suffix + '_val.json'), 'w') as fd:
            fd.writelines(map(dump_row, half_data_1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('--root', type=str)
    args = parser.parse_args()
    if args.root is not None:
        DATA_ROOT = args.root
    train_seqs = sorted(os.listdir(os.path.join(DATA_ROOT, 'train')))  # sorted to maintain the unique track ids
    test_seqs = sorted(os.listdir(os.path.join(DATA_ROOT, 'test')))
    if args.test:
        convert_annos(test_seqs, 'test')
    else:
        convert_annos(train_seqs, 'train', half=True, gap=30)
