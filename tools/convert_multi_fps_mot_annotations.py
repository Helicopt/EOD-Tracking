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


def generate_indices(frames, skip=1, separate=False):
    n = len(frames)
    ret = []
    r = n % skip
    reverse = False
    for i in range(skip):
        tmp = [(j * skip + i) for j in range(n // skip + (i < r))]
        if separate:
            ret.append([frames[j] for j in tmp])
        else:
            if reverse:
                ret.extend(tmp[::-1])
            else:
                ret.extend(tmp)
            reverse = not reverse
    if not separate:
        ret = [[frames[j] for j in ret]]
    v = set()
    for one in ret:
        v = v.union(one)
    assert len(v) == n, '%d vs %d' % (len(v), n)
    return ret


def generate(frames, ground_truth, seq_dir, image_height, image_width, virtual_seqname):
    formatter = '{root}/{seq}/img1/{fr}.{ext}'
    all_data = []
    for i, frame_id in enumerate(frames):
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
            'virtual_filename': os.path.join('/', 'virtual_path', virtual_seqname, '%06d.jpg' % (i + 1))
        }
        all_data.append(data)
    return all_data


def convert_annos(seqs, split, half=False, gap=30, separate=False):
    '''
    convert mot20 annotations
    '''
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
        if separate:
            suffix += '_sep'
        frames = list(range(min_fr, max_fr + 1))
        half_ = (min_fr + max_fr) // 2
        for skip in skips:
            if half:
                ind_list0 = generate_indices([fr for fr in frames if fr <= half_], skip=skip, separate=separate)
                ind_list1 = generate_indices([fr for fr in frames if fr > half_ + gap], skip=skip, separate=separate)
                for i, indices in enumerate(ind_list0):
                    if separate:
                        virtual_seqname_0 = 'S-%d-%d-%s' % (skip, i, seq)
                    else:
                        virtual_seqname_0 = 'S-%d-%s' % (skip, seq)
                    half_data_0 += generate(indices, ground_truth, seq_dir,
                                            image_height, image_width, virtual_seqname_0)
                for i, indices in enumerate(ind_list1):
                    if separate:
                        virtual_seqname_1 = 'S-%d-%d-%s' % (skip, i, seq)
                    else:
                        virtual_seqname_1 = 'S-%d-%s' % (skip, seq)
                    half_data_1 += generate(indices, ground_truth, seq_dir,
                                            image_height, image_width, virtual_seqname_1)
            indices_list = generate_indices(frames, skip=skip, separate=separate)
            for i, indices in enumerate(indices_list):
                if separate:
                    virtual_seqname = 'S-%d-%d-%s' % (skip, i, seq)
                else:
                    virtual_seqname = 'S-%d-%s' % (skip, seq)
                all_data += generate(indices, ground_truth, seq_dir, image_height, image_width, virtual_seqname)
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
    parser.add_argument('--separate', action='store_true')
    args = parser.parse_args()
    if args.root is not None:
        DATA_ROOT = args.root
    train_seqs = sorted(os.listdir(os.path.join(DATA_ROOT, 'train')))  # sorted to maintain the unique track ids
    test_seqs = sorted(os.listdir(os.path.join(DATA_ROOT, 'test')))
    if args.test:
        convert_annos(test_seqs, 'test', separate=args.separate)
    else:
        convert_annos(train_seqs, 'train', half=True, gap=30, separate=args.separate)
