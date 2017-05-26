#!/usr/bin/env python
import os
import json
from keras import datasets
from PIL import Image

DATA_DIR = os.path.expanduser('~/data')

def save_set(fold, x, y, suffix='png'):
    fp = open('mnist_{}.dataset'.format(fold), 'w')
    for i in range(len(x)):
        label = y[i]
        img_filename = 'mnist/{}/{:05d}_{:d}.{}'.format(fold, i, label, suffix)
        Image.fromarray(x[i]).save(img_filename)
        entry = {'filename': img_filename, 'label': str(label)}
        fp.write(json.dumps(entry))
        fp.write('\n')
    fp.close()
    

if __name__ == '__main__':
    os.chdir(DATA_DIR)
    try:
        os.mkdir('mnist')
        os.mkdir('mnist/train')
        os.mkdir('mnist/test')
    except:
        print("MNIST dataset exists at {}/mnist".format(DATA_DIR))
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    save_set('train', train_x, train_y)
    save_set('test', test_x, test_y)
