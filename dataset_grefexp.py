import os
import sys
import json
import random

import redis
import numpy as np
import enchant

DATA_DIR = '/home/nealla/data'

KEY_GREFEXP_TRAIN = 'dataset_grefexp_train'
KEY_GREFEXP_VAL = 'dataset_grefexp_val'

conn = redis.Redis()
categories = {v['id']: v['name'] for v in json.load(open('coco_categories.json'))}


def example(reference_key=KEY_GREFEXP_TRAIN):
    key = conn.srandmember(reference_key)
    return get_annotation_for_key(key)


def get_all_keys(reference_key=KEY_GREFEXP_VAL, shuffle=True):
    keys = list(conn.smembers(reference_key))
    if shuffle:
        random.shuffle(keys)
    return keys


def get_annotation_for_key(key):
    grefexp = json.loads(conn.get(key))
    anno_key = 'coco2014_anno_{}'.format(grefexp['annotation_id'])
    anno = json.loads(conn.get(anno_key))
    img_key = 'coco2014_img_{}'.format(anno['image_id'])
    img_meta = json.loads(conn.get(img_key))

    img_width = float(img_meta['width'])
    img_height = float(img_meta['height'])
    x0, y0, width, height = anno['bbox']

    box = (x0 / img_width, (x0 + width) / img_width, y0 / img_height, (y0 + height) / img_height)

    texts = [g['raw'] for g in grefexp['refexps']]

    #texts = [spell(strip(t, strip_end=False)) for t in texts]

    category = categories[anno['category_id']]
    return img_meta['filename'], box, category, texts


if __name__ == '__main__':
    img_filename = sys.argv[1]
    sentence_filename = sys.argv[2]
    fp_img = open(img_filename, 'w')
    fp_sentence = open(sentence_filename, 'w')

    ref_key = KEY_GREFEXP_VAL if 'val' in sys.argv else KEY_GREFEXP_TRAIN
    for k in get_all_keys(ref_key):
        filename, box, category, texts = get_annotation_for_key(k)
        for sentence in texts:
            img_info = {'filename': filename, 'bbox': box, 'category': category}
            fp_img.write(json.dumps(img_info) + '\n')
            fp_sentence.write(sentence.encode('utf-8') + '\n')
    fp_img.close()
    fp_sentence.close()
