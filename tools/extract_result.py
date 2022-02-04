from senseTk.common import TrackSet, Det
import os
import sys
import json


def parse(image_id):
    fr = int(os.path.basename(image_id).split('.')[0])
    seq_dir = os.path.dirname(os.path.dirname(image_id))
    seq = os.path.basename(seq_dir)
    return seq, fr


def extract(src, dest=None):
    cache = {}
    with open(src) as fd:
        rows = fd.readlines()
    for row in rows:
        detdict = json.loads(row)
        x1, y1, x2, y2 = detdict['bbox']
        uid = detdict['track_id']
        image_id = detdict.get('vimage_id', detdict['image_id'])
        seq, fr = parse(image_id)
        if seq not in cache:
            cache[seq] = TrackSet()
        d = Det(x1, y1, x2 - x1, y2 - y1, uid=uid, fr=fr)
        cache[seq].append_data(d)
    if dest is None:
        dest = os.path.dirname(src)
    for seq in cache:
        ts = cache[seq]
        nts = TrackSet()
        for fr in ts.frameRange():
            nfr = fr - ts.min_fr + 1
            for d in ts[fr]:
                d = d.copy()
                # d.fr = nfr
                nts.append_data(d)
        with open(os.path.join(dest, '%s.txt' % seq), 'w') as fd:
            nts.dump(fd)


if __name__ == '__main__':
    src = sys.argv[1]
    print(src)
    extract(src)
