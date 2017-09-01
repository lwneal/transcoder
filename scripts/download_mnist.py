#!/usr/bin/env python
import sys
import tqdm
import os
import json
from keras import datasets
from PIL import Image
from tqdm import tqdm

DATA_DIR = os.path.expanduser('~/data')

def save_set(fold, x, y, suffix='png'):
    examples = []
    fp = open('mnist_{}.dataset'.format(fold), 'w')
    print("Writing MNIST dataset {}".format(fold))
    for i in tqdm(range(len(x))):
        label = y[i]
        img_filename = 'mnist/{}/{:05d}_{:d}.{}'.format(fold, i, label, suffix)
        Image.fromarray(x[i]).save(img_filename)
        entry = {
                'filename': img_filename,
                'label': str(label),
                'is_holy': label in [0, 6, 8, 9],  # numbers with holes in them
                'is_pointy': label in [1, 4, 7],  # numbers with no curvy parts
                'is_symmetric': label in [0, 1, 8],  # left-right symmetric numbers
                'is_vert_symmetric': label in [0, 3, 8],  # left-right symmetric numbers
        }
        examples.append(entry)
        fp.write(json.dumps(entry))
        fp.write('\n')
    fp.close()
    return examples
    

if __name__ == '__main__':
    os.chdir(DATA_DIR)
    try:
        os.mkdir('mnist')
        os.mkdir('mnist/train')
        os.mkdir('mnist/test')
    except OSError:
        print("MNIST dataset exists at {}/mnist".format(DATA_DIR))
        if '--force' not in sys.argv:
            exit()
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    train = save_set('train', train_x, train_y)
    test = save_set('test', test_x, test_y)
    for example in train:
        example['fold'] = 'train'
    for example in test:
        example['fold'] = 'test'
    with open('mnist.dataset', 'w') as fp:
        for example in train + test:
            fp.write(json.dumps(example) + '\n')
