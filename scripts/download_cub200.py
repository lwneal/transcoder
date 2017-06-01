#!/usr/bin/env python
# Downloads the CUB200 dataset
import os
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

if __name__ == '__main__':
    mkdir(DATA_DIR)
    mkdir(CUB_DIR)
    os.chdir(CUB_DIR)

    # Download and extract dataset
    download('images', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz')
    download('lists', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz')
    download('annotations-mat', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz')
    download('attributes', 'http://www.vision.caltech.edu/visipedia-data/CUB-200/attributes.tgz')

    # Generate CSV files
    text = check_output('find images | grep -v "\.\_" | grep jpg$', shell=True)
    images = text.splitlines()

    print("Writing {} items to cub200.dataset".format(len(images)))
    fp = open('{}/cub200.dataset'.format(DATA_DIR), 'w')
    for filename in images:
        label = filename.lstrip('cub200/images/').split('/')[0].split('.')[-1]
        line = json.dumps({
            'filename': 'cub200/' + filename,
            'label': label,
        })
        fp.write(line + '\n')
    fp.close()
    print("CUB200 dataset ready")
