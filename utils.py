"""
Modified version of https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py
"""
from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from glob import glob

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def get_batch_images(batch_index, data, config):

    if type(data) == list:
        is_grayscale = (config.c_dim == 1)

        batch_files = data[batch_index*config.batch_size:(batch_index+1)*config.batch_size]
        batch = [get_image(batch_file, config.image_size, is_crop=config.is_crop, resize_w=config.output_size, is_grayscale = is_grayscale) for batch_file in batch_files]

        if (is_grayscale):
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
            batch_images = np.array(batch).astype(np.float32)

        return batch_images

    elif type(data) == np.ndarray:
        return data[batch_index*config.batch_size:(batch_index+1)*config.batch_size]

def check_data_arr(config):

    is_crop = '_cropped' if config.is_crop == True else ''
    npy_path = os.path.join('./data', config.dataset+is_crop+'.npy')

    if not os.path.exists(npy_path):
        is_grayscale = (config.c_dim == 1)
        files = glob(os.path.join('./data', config.dataset, '*.jpg'))
        data = [get_image(batch_file, config.image_size, is_crop=config.is_crop, resize_w=config.output_size, is_grayscale = is_grayscale) for batch_file in files]

        if (is_grayscale):
            data = np.array(data).astype(np.float32)[:, :, :, None]
        else:
            data = np.array(data).astype(np.float32)

        np.save(npy_path, data)

    return npy_path
