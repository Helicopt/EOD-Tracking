from typing import List
import json
import os
import cv2

from senseTk.common import TrackSet, Det

DATA_ROOT = '/mnt/lustre/share/fengweitao/MOT20'


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
        ground_truth = TrackSet(os.path.join(seq_dir, 'gt', 'gt.txt'))
        image_probe = cv2.imread(os.path.join(seq_dir, 'img1', '000001.jpg'))
        image_height, image_width, _ = image_probe.shape
        for frame_id in ground_truth.frameRange():
            img = os.path.join(seq_dir, 'img1', '%06d.jpg' % frame_id)
            gtdets: List[Det] = ground_truth[frame_id]
            instances = []
            for gtdet in gtdets:
                if gtdet.status != 1:
                    continue
                one = {
                    'is_ignored': False,
                    'bbox': [gtdet.x1, gtdet.y1, gtdet.x2, gtdet.y2],
                    'label': 1,
                    'track_id': gtdet.uid,
                }
                instances.append(one)
            data = {
                'filename': os.path.abspath(img),
                'instances': instances,
                'image_width': image_width,
                'image_height': image_height,
                'formatter': formatter,
            }
            all_data.append(data)
            half = (ground_truth.min_fr + ground_truth.max_fr) // 2
            if frame_id <= half:
                half_data_0.append(data)
            elif frame_id > half + gap:
                half_data_1.append(data)
    anno_dir = os.path.join(DATA_ROOT, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    def dump_row(x): return json.dumps(x) + '\n'
    with open(os.path.join(anno_dir, split + '.json'), 'w') as fd:
        fd.writelines(map(dump_row, all_data))
    if half:
        with open(os.path.join(anno_dir, split + '_half.json'), 'w') as fd:
            fd.writelines(map(dump_row, half_data_0))
        with open(os.path.join(anno_dir, split + '_val.json'), 'w') as fd:
            fd.writelines(map(dump_row, half_data_1))


train_seqs = os.listdir(os.path.join(DATA_ROOT, 'train'))
convert_annos(train_seqs, 'train', half=True, gap=30)


test_seqs = os.listdir(os.path.join(DATA_ROOT, 'test'))
