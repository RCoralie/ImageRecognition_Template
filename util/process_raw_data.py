#!/usr/bin/env python2 
# -*- coding: utf-8 -*- 
"""
    Builds data of the right shape for the input layer and the labels of the output layer shape
"""
import logging
import tensorflow as tf
import numpy as np
import sys, os

def process_label(lex, charmap):
    """
    convert_lex is needed to convert label to an array of int
    """
    lex = lex.decode('utf_8')
    word = np.zeros((len(charmap),), dtype=float)
    try:
        word[charmap.index(lex)] = 1.
    except ValueError as e :
        logging.info('\033[31mERROR - Check label file : %s \033[0m' % e)
        os._exit(-1)
    return word


def process_image(raw_image, init_height, init_width, channel, final_height, final_width):
    """
    To be processed by a neural network, all images must be decoded and have the same shape
    """
    # Decode raw image : /!\ 1: output a grayscale image !
    img = tf.image.decode_image(raw_image, channels=1)
    # Resizes images
    img = tf.reshape(img, [init_height, init_width, 1])
    img = tf.image.resize_image_with_crop_or_pad(image=img,
                                                target_height=final_height,
                                                target_width=final_width)
    # Breaks down the channels and reshape the image to feed the input layer
    # /!\ : Only work for grayscale !!!! -> TODO
    img = tf.reshape(img, [final_height*final_width])
    return img
