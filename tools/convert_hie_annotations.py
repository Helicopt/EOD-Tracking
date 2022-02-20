from typing import List
import json
import os
import sys
import configparser
import PIL.Image as Image
import argparse

from senseTk.common import TrackSet, Det

DATA_ROOT = '/mnt/lustre/share/fengweitao/HIE/videos/train'
GT_ROOT = '/mnt/lustre/share/fengweitao/HIE/labels/train/track1'


def convert_annos(seqs, split, half=False, gap=30, fps=0):
    '''
    convert mot20 annotations
    '''
    formatter = '{root}/{seq}/{fr}.{ext}'
    all_data = []
    half_data_0 = []
    half_data_1 = []
    for seq in seqs:
        seq_dir = os.path.join(DATA_ROOT, seq)
        gt_path = os.path.join(GT_ROOT, seq.split('.')[0] + '.txt')
        if os.path.exists(gt_path):
            ground_truth = TrackSet(gt_path)
        else:
            ground_truth = None
        image_probe = Image.open(os.path.join(seq_dir, '000001.jpg'))
        image_height, image_width = image_probe.height, image_probe.width
        # inifile = os.path.join(seq_dir, 'seqinfo.ini')
        # iniconfig = configparser.ConfigParser()
        # iniconfig.read(inifile)
        min_fr = 1
        # max_fr = int(iniconfig['Sequence']['seqLength'])
        max_fr = len(os.listdir(seq_dir))
        # original_fps = float(iniconfig['Sequence']['frameRate'])
        original_fps = 25
        suffix = ''
        if fps > 0:
            skip = int(original_fps / fps + 0.9)
            real_fps = original_fps / skip
            print('generating %s in a frameRate of %.1f fps' % (seq, real_fps))
            suffix = '_f%d' % int(real_fps)
        for frame_id in range(min_fr, max_fr + 1):
            if fps > 0:
                if frame_id % skip != 1:
                    continue
            img = os.path.join(seq_dir, '%06d.jpg' % frame_id)
            instances = []
            if ground_truth:
                gtdets: List[Det] = sorted(ground_truth[frame_id - 1], key=lambda x: x.uid)
                for gtdet in gtdets:
                    # if gtdet.status != 1:
                    #     continue
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
            half_ = (min_fr + max_fr) // 2
            if frame_id <= half_:
                half_data_0.append(data)
            elif frame_id > half_ + gap:
                half_data_1.append(data)
    anno_dir = os.path.join(DATA_ROOT, '..', 'annotations')
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
    parser.add_argument('--gt', type=str)
    parser.add_argument('--fps', type=int, default=0)
    args = parser.parse_args()
    if args.root is not None:
        DATA_ROOT = args.root
    if args.gt is not None:
        GT_ROOT = args.gt
    seqs = sorted(os.listdir(DATA_ROOT))
    if args.test:
        convert_annos(eqs, 'test', fps=args.fps)
    else:
        convert_annos(seqs, 'train', half=True, gap=30, fps=args.fps)
