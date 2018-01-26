"""
    Builds data of the right shape for the input layer and the labels of the output layer shape
"""

import tensorflow as tf
import numpy as np
import sys

from ..defaults import Config


CHARMAP = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def process_label(lex):
    """
    convert_lex is needed to convert label to an array of int
    """
    if sys.version_info >= (3,):
        lex = lex.decode('iso-8859-1')
        word = np.zeros((len(CHARMAP),), dtype=float)
        word[CHARMAP.index(lex)] =1.
        return word


def process_png(raw_image, init_height, init_width):
    """
    To be processed by a neural network, all images must be decoded and have the same shape
    """
    # Decode raw image
    img = tf.image.decode_png(raw_image, channels=Config.CHANNELS)
    # Resizes images
    img = tf.reshape(img, [init_height, init_width, Config.CHANNELS])
    img = tf.image.resize_image_with_crop_or_pad(image=img,
                                                target_height=Config.IMAGE_HEIGHT,
                                                target_width=Config.IMAGE_WIDTH)
    # Breaks down the channels and reshape the image to feed the input layer
    # /!\ : Only work for grayscale !!!! -> TODO
    img = tf.reshape(img, [Config.IMAGE_HEIGHT*Config.IMAGE_WIDTH])
    return img
