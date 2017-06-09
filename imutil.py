import math
import os
import tempfile
import subprocess
from distutils import spawn
import numpy as np
from PIL import Image
from StringIO import StringIO


# Input: Numpy array containing one or more images
# Output: JPG encoded image bytes
def encode_jpg(pixels, resize_to=None):
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels)
    # Convert to RGB to avoid "Cannot handle this data type"
    if pixels.shape[-1] < 3:
        pixels = np.repeat(pixels, 3, axis=-1)
    img = Image.fromarray(pixels.astype(np.uint8))
    if resize_to:
        img = img.resize(resize_to)
    fp = StringIO()
    img.save(fp, format='JPEG')
    return fp.getvalue()


# Input: Filename, or JPG bytes
# Output: Numpy array containing images
def decode_jpg(jpg, crop_to_box=None, resize_to=(224,224)):
    if jpg.startswith('\xFF\xD8'):
        # Input is a JPG buffer
        img = Image.open(StringIO(jpg))
    else:
        # Input is a filename
        img = Image.open(jpg)
    img = img.convert('RGB')
    if crop_to_box:
        # Crop to bounding box
        x0, x1, y0, y1 = crop_to_box
        width, height = img.size
        img = img.crop((x0 * width, y0 * height, x1 * width, y1 * height))
    if resize_to:
        img = img.resize(resize_to)
    return np.array(img).astype(float)


# Swiss-army knife for putting an image on the screen
# Accepts numpy arrays, PIL Image objects, or jpgs
# Numpy arrays can consist of multiple images, which will be collated
def show(data, filename=None, box=None, video_filename=None, resize_to=(224,224)):
    if type(data) == type(np.array([])):
        pixels = data
    elif type(data) == Image.Image:
        pixels = np.array(data)
    else:
        pixels = decode_jpg(data)
    if box:
        draw_box(pixels, box)

    if pixels.shape[-1] > 3:
        pixels = np.expand_dims(pixels, axis=-1)
    while len(pixels.shape) < 3:
        pixels = np.expand_dims(pixels, axis=-1)
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels)

    if filename is None:
        filename = tempfile.NamedTemporaryFile(suffix='.jpg').name

    with open(filename, 'w') as fp:
        print("Pixel value range {}-{}".format(pixels.min(), pixels.max()))
        pixels = (pixels - pixels.min()) * 255. / (pixels.max() - pixels.min())
        fp.write(encode_jpg(pixels, resize_to=resize_to))
        fp.flush()
        # Display image in the terminal if an appropriate program is available
        for prog in ['imgcat', 'catimg', 'feh', 'display']:
            if spawn.find_executable(prog):
                # Tmux hack
                #print('\n' * 14)
                #print('\033[14F')
                subprocess.check_call([prog, filename])
                #print('\033[14B')
                break
        else:
            print("Saved image size {} as {}".format(pixels.shape, filename))

    # Output JPG files can be collected into a video with ffmpeg -i *.jpg
    if video_filename:
        open(video_filename, 'a').write(encode_jpg(pixels, resize_to=resize_to))


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        a0, a1 = i*shape[0], (i+1)*shape[0]
        b0, b1 = j*shape[1], (j+1)*shape[1]
        image[a0:a1, b0:b1] = img
    return image


def draw_box(img, box, color=1.0):
    x0, x1, y0, y1 = (int(val) for val in box)
    height, width, channels = img.shape
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    img[y0:y1,x0] = color
    img[y0:y1,x1] = color
    img[y0,x0:x1] = color
    img[y1,x0:x1] = color
