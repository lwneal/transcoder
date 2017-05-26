#!/usr/bin/env python
# Downloads the CUB200 dataset
import os

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

if __name__ == '__main__':
    mkdir(DATA_DIR)
    mkdir(CUB_DIR)
    os.chdir(CUB_DIR)
    download('images', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz')
    download('lists', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz')
    download('annotations-mat', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz')
    download('attributes', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/attributes.tgz')

