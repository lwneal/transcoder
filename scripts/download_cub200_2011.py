#!/usr/bin/env python
# Downloads the CUB200_2011 dataset
# (Roughly double the number of images per class of the original CUB200)
import os
import numpy as np
import json
from subprocess import check_output


DATA_DIR = os.path.expanduser('~/data')
CUB_DIR = os.path.join(DATA_DIR, 'cub200_2011')
DATASET_NAME = 'cub200_2011'


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def filesize(filename):
    return os.stat(filename).st_size


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        os.system('wget -nc {}'.format(url))
        if url.endswith('.tgz') or url.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')


def load_mat_file(filename):
    import scipy.io as sio
    mat = sio.loadmat(filename)
    return mat['seg']


def get_width_height(filename):
    from PIL import Image
    img = Image.open(os.path.join(DATA_DIR, filename))
    return (img.width, img.height)


def save_image_dataset(images, boxes, fold_name=None):
    if fold_name:
        output_filename = '{}/{}_{}.dataset'.format(DATA_DIR, DATASET_NAME, fold_name)
    else:
        output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    print("Writing {} items to {}".format(len(images), output_filename))
    fp = open(output_filename, 'w')
    for filename, box in zip(images, boxes):
        label = filename.lstrip('{}/images/'.format(DATASET_NAME)).split('/')[0].split('.')[-1]
        filename = DATASET_NAME + '/images/' + filename
        width, height = get_width_height(filename)
        left, top, box_width, box_height = box
        x0 = left / width
        x1 = (left + box_width) / width
        y0 = top / height
        y1 = (top + box_height) / height
        line = json.dumps({
            'filename': filename,
            'label': label,
            'box': (x0, x1, y0, y1),
        })
        fp.write(line + '\n')
    fp.close()


def train_test_split(filename='train_test_split.txt'):
    # Training examples end with 1, test with 0
    return [line.endswith('1\n') for line in open(filename)]


if __name__ == '__main__':
    print("CUB200_2011 dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(CUB_DIR)
    os.chdir(CUB_DIR)

    # Download and extract dataset
    print("Downloading CUB200_2011 dataset files...")
    download('images', 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz')
    download('segmentations', 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz')

    os.system('mv CUB_200_2011/* . && rmdir CUB_200_2011')

    # Generate CSV file for the full dataset
    lines = open('images.txt').readlines()
    ids = [int(line.split()[0]) for line in lines]
    images = [line.split()[1] for line in lines]

    boxes = open('bounding_boxes.txt').readlines()
    boxes = [[float(w) for w in line.split()[1:]] for line in boxes]
    save_image_dataset(images, boxes)

    # Generate train/test split CSV files
    is_training = train_test_split()
    assert len(is_training) == len(images)

    train_images = [img for (img, t) in zip(images, is_training) if t]
    test_images = [img for (img, t) in zip(images, is_training) if not t]
    train_boxes = [box for (box, t) in zip(boxes, is_training) if t]
    test_boxes = [box for (box, t) in zip(boxes, is_training) if not t]

    assert len(train_images) + len(test_images) == len(images)
    save_image_dataset(train_images, train_boxes, "train")
    save_image_dataset(test_images, test_boxes, "test")

    print("Finished building CUB200_2011 dataset")
