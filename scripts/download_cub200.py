#!/usr/bin/env python
# Downloads the CUB200 dataset
import os
import random
import json
from subprocess import check_output


DATA_DIR = os.path.expanduser('~/data')
CUB_DIR = os.path.join(DATA_DIR, 'cub200')


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


def save_image_dataset(images, fold_name=None):
    if fold_name:
        output_filename = '{}/cub200_{}.dataset'.format(DATA_DIR, fold_name)
    else:
        output_filename = '{}/cub200.dataset'.format(DATA_DIR)
    print("Writing {} items to {}".format(len(images), output_filename))
    fp = open(output_filename, 'w')
    for filename in images:
        label = filename.lstrip('cub200/images/').split('/')[0].split('.')[-1]
        line = json.dumps({
            'filename': 'cub200/' + filename,
            'label': label,
        })
        fp.write(line + '\n')
    fp.close()
    print("CUB200 dataset ready")


if __name__ == '__main__':
    print("CUB200 dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(CUB_DIR)
    os.chdir(CUB_DIR)

    # Download and extract dataset
    print("Downloading CUB200 dataset files...")
    download('images', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz')
    download('lists', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz')
    download('annotations-mat', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz')
    download('attributes', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/attributes.tgz')

    # Generate CSV files
    print("Generating Transcoder format CUB200 files")
    text = check_output('find images | grep -v "\.\_" | grep jpg$', shell=True)
    images = text.splitlines()
    save_image_dataset(images)

    # Split into a training and a test set
    TRAIN_TEST_SPLIT = .8
    split_idx = int(float(len(images)) * TRAIN_TEST_SPLIT)
    random.shuffle(images)
    train_images, test_images = images[:split_idx], images[split_idx:]

    save_image_dataset(train_images, "train")
    save_image_dataset(test_images, "test")
    print("Finished building CUB200 dataset")
