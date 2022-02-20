import os
import re
import json
import argparse
from senseTk.functions import autoPattern


def smart_replace(old, root):
    dataset = os.path.basename(root)
    new_loc = os.path.dirname(root)
    loc = re.split(dataset, old)[-1]
    new_path = os.path.join(new_loc, dataset + loc)
    ext = os.path.splitext(new_path)[-1]
    anno_path = new_path.replace('images', 'labels_with_ids').replace(ext, '.txt')
    return new_path, anno_path


def regenerate(jsonfile, root):
    jsonfile = os.path.abspath(jsonfile)
    root = os.path.abspath(root)
    anno = []
    paths = []
    with open(jsonfile) as fd:
        txt = fd.readlines()
        for raw in txt:
            line = json.loads(raw.strip())
            anno.append(line)
    new_anno = []
    for one in anno:
        # print(one)
        filename, gt = smart_replace(one['filename'], root)
        # print(filename, gt)
        if not os.path.exists(filename):
            continue
        elif not os.path.exists(gt):
            assert not one['instances']
            instances = []
        else:
            with open(gt) as fd:
                instances = fd.readlines()
            for i, instance in enumerate(instances):
                instance = re.split(r'\s+', instance.strip())
                uid = int(instance[1])
                cx, cy, w, h = map(float, instance[2:6])
                cx, w = map(lambda x: x * int(one['image_width']), (cx, w))
                cy, h = map(lambda x: x * int(one['image_height']), (cy, h))
                data = {
                    'is_ignored': False,
                    'bbox': [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    'label': 1,
                    'track_id': uid,
                }
                instances[i] = data
        new_one = {}
        new_one['filename'] = filename
        new_one['instances'] = instances
        new_one['image_width'] = one['image_width']
        new_one['image_height'] = one['image_height']
        seqname = ''
        frame = 0
        ext = '.jpg'
        if 'ETH' in filename:
            new_one['formatter'] = '{root}/{seq}/images/image_{fr}_{p}.{ext}'
            seqname = os.path.basename(os.path.dirname(os.path.dirname(filename)))
            frame = int(os.path.basename(filename).split('_')[1])
            ext = os.path.splitext(filename)[-1]
        if 'Cityscapes' in filename:
            new_one['formatter'] = '{root}/{seq}/{seq}_{fr}_{p}_{q}.{ext}'
            seqname = os.path.basename(os.path.dirname(filename))
            frame = int(os.path.basename(filename).split('_')[1])
            ext = os.path.splitext(filename)[-1]
        new_one['virtual_filename'] = os.path.join('/', 'virtual_path', seqname, '%06d%s' % (frame, ext))
        new_anno.append(new_one)

    def dump_row(x): return json.dumps(x) + '\n'
    with open(jsonfile + '.id.json', 'w') as fd:
        fd.writelines(map(dump_row, new_anno))
        # paths.append(filename)
    # pattern = autoPattern(paths)
    # print(pattern)
    # for one in new_anno:
    #     one['formatter'] = pattern_format
    #     new_one['virtual_filename'] = os.path.join('/', 'virtual_path', seqname, '%06d.jpg' % fr)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('root', type=str, help='data root')
    argparser.add_argument('json', type=str, help='json')
    args = argparser.parse_args()
    regenerate(args.json, root=args.root)
