import json
import os
from PIL import Image


DATA_ROOT = '/mnt/lustre/share/fengweitao/crowd_human'


def convert_annos(anno, split):
    '''
    convert crowd human annotations
    '''
    img_dir = os.path.join(DATA_ROOT, 'Images')
    formatter = '{root}/Images/{seq}.{ext}'
    all_data = []
    with open(anno) as fd:
        lines = fd.readlines()
        for line in lines:
            row = json.loads(line)
            filename = os.path.join(img_dir, '%s.jpg' % row['ID'])
            image_probe = Image.open(filename)
            image_height, image_width = image_probe.height, image_probe.width
            instances = []
            gtdets = row.get('gtboxes', [])
            for i, gtdet in enumerate(gtdets):
                ignore = gtdet['extra'].pop('ignore', 0)
                if gtdet['tag'] != 'person' or ignore:
                    continue
                x1, y1, w, h = gtdet['fbox']
                x2 = x1 + w / 2.
                y2 = y1 + h / 2.
                uid = i + 1
                one = {
                    'is_ignored': False,
                    'bbox': [x1, y1, x2, y2],
                    'label': 1,
                    'track_id': -1,
                }
                instances.append(one)
            data = {
                'filename': os.path.abspath(filename),
                'instances': instances,
                'image_width': image_width,
                'image_height': image_height,
                'formatter': formatter,
            }
            all_data.append(data)
    anno_dir = os.path.join(DATA_ROOT, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    def dump_row(x): return json.dumps(x) + '\n'
    with open(os.path.join(anno_dir, split + '.json'), 'w') as fd:
        fd.writelines(map(dump_row, all_data))


if __name__ == '__main__':
    train_odgt = os.path.join(DATA_ROOT, 'annotation_train.odgt')
    val_odgt = os.path.join(DATA_ROOT, 'annotation_val.odgt')
    convert_annos(train_odgt, 'train')
    convert_annos(val_odgt, 'val')
