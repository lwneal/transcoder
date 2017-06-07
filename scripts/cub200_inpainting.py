#!/usr/bin/env python
# Requires that CUB200 has already been downloaded
# Generates a copy of CUB200 with random regions blacked out
# For use as training data for inpainting
import sys
import os
import json
import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = os.path.expanduser('~/data')
CUB_DIR = os.path.join(DATA_DIR, 'cub200')
IMG_DIR = os.path.join(CUB_DIR, 'images')
OUTPUT_DIR = os.path.join(CUB_DIR, 'damaged_images')


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR):
        print("{} already exists, {} exiting".format(OUTPUT_DIR, sys.argv[0]))
        exit()
    mkdir(OUTPUT_DIR)
    os.chdir(IMG_DIR)

    output_filenames = []
    np.random.seed(1)

    for dirname in os.listdir('.'):
        for filename in os.listdir(dirname):
            if filename.startswith('.'):
                #print("Skipping invalid file {}".format(filename))
                continue
            img = Image.open(os.path.join(dirname, filename))
            # Draw some black rectangles
            draw = ImageDraw.Draw(img)
            for i in range(3):
                width = np.random.randint(100)
                height = np.random.randint(100)
                x = np.random.randint(img.width - width)
                y = np.random.randint(img.height - height)
                draw.rectangle((x, y, x+width, y+width), fill=0)
            output_filename = os.path.join(OUTPUT_DIR, filename)
            img.save(output_filename)
            output_filenames.append('cub200/damaged_images/' + filename)
    os.chdir(DATA_DIR)
    with open('cub200_damaged.dataset', 'w') as fp:
        for f in output_filenames:
            fp.write(json.dumps({'filename': f}) + '\n')
