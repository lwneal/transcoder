#!/usr/bin/env python
import os
import json
from keras import datasets
from PIL import Image

DATA_DIR = os.path.expanduser('~/data')

def save_set(fold, x, y, suffix='png'):
    fp_txt = open('mnist_{}.txt'.format(fold), 'w')
    fp_img = open('mnist_{}.img'.format(fold), 'w')
    for i in range(len(x)):
        label = y[i]
        filename = 'mnist/{}/{:05d}_{:d}.{}'.format(fold, i, label, suffix)
        Image.fromarray(x[i]).save(filename)
        fp_txt.write('{}\n'.format(label))
        fp_img.write(json.dumps({'filename': filename}))
        fp_img.write('\n')
    fp_txt.close()
    fp_img.close()
    

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
