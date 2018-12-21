#!/usr/bin/env python2 
# -*- coding: utf-8 -*- 
"""
Generate dataset in tfrecords format(turn all the images and associated label from a folder (use folder name as unique label) into a tfRecord file).
TFRecords are TensorFlow’s default data format. A record is simply a binary file that contains serialized tf.train.Example Protobuf objects.

Basically, an Example always contains Features.
Features contains a map of strings to Feature.
And finally, a Feature contains one of a FloatList, a ByteList or a Int64List.

This Example proto contains the following fields:
image: string containing encoded image
label: string specifying the human-readable version of the label
height: integer, image height in pixels
width: integer, image width in pixels
colorspace: string, specifying the colorspace
channels: integer, specifying the number of channels
format: string, specifying the format

For more informations, see :
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto
"""

import logging
import glob
import os.path
import random
import sys
from PIL import Image
from six import b
import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def writeAnnotation(input_dir_path, output_file_path, log_step):
    """Turn all the images and associated label from a folder (use folder name as unique label) into a txt file"""
    logging.info(' - Create labeling file - ')
    logging.info('Input directory: %s', input_dir_path)

    output_file = open(output_file_path,'w+')

    idx = 0;
    for root, dirs, files in os.walk(input_dir_path):
        for d in dirs:
            for f in os.listdir(os.path.join(root,d)):
                if not f.startswith('.'):
                    output_file.write(os.path.join(os.path.join(root,d),f))
                    output_file.write("\t")
                    output_file.write(d)
                    output_file.write("\n")
                    if idx % log_step == 0:
                        logging.info('\033[0;37mProcessed %i pairs\033[0m', idx)
                    idx = idx + 1

    output_file.close
    logging.info('\033[0;32mProcessed %i pairs \033[0m', idx)
    logging.info('\033[0;32mLabeling file is ready: %s \033[0m \n ', output_file.name)


def generateDataset(annotations_path, output_path, log_step):
    """Turn all the images and associated label from a txt file into a tfRecords file"""
    logging.info(' - Generate binary dataset - ')
    logging.info('Input file: %s', annotations_path)

    writer = tf.python_io.TFRecordWriter(output_path)

    label_list = []
    greater_height = 0
    greater_width = 0
    nb_pair = 0
    with open(annotations_path,'r') as source:
        # iterate over each and construct the Example proto oject
        for idx, line in enumerate(source):
            line = line.rstrip('\n')
            try:
                (img_path, label) = line.split('\t', 1)
            except ValueError:
                logging.error('\033[0;31m missing filename or label at line %i: %s (ignored data)\033[0m', idx+1, line)
                continue

            img_pil = Image.open(img_path)
            img = np.array(img_pil)
            height = img.shape[0]
            width = img.shape[1]
            channel = 1 if len(img.shape) == 2  else img.shape[2]
            img_format = img_pil.format

            with open(img_path, 'rb') as img_file:
                img_raw = img_file.read()

            if label not in label_list:
                label_list.append(label)

            if height > greater_height:
                greater_height = height

            if width > greater_width:
                greater_width = width

            # A Feature contains a map of string to Feature proto objects which are one of either a int64_list, float_list, or bytes_list
            feature = {}
            feature['image'] = _bytes_feature(img_raw)
            feature['label'] = _bytes_feature(b(label))
            feature['height'] = _int64_feature(height)
            feature['width'] = _int64_feature(width)
            feature['channel'] = _int64_feature(channel)
            feature['format'] = _bytes_feature(tf.compat.as_bytes(img_format))

            # Example contains a Features proto object
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # use the proto object to serialize the example to a string and write the serialized object to disk
            writer.write(example.SerializeToString())

            nb_pair += 1
            if idx % log_step == 0:
                logging.info('\033[0;37mProcessed %i pairs\033[0m', idx+1)

    logging.info('\033[0;32mProcessed %i pairs\033[0m', nb_pair+1)
    logging.info('\033[0;32mGreater height : %s\033[0m', greater_height)
    logging.info('\033[0;32mGreater width : %s\033[0m', greater_width)
    logging.info('\033[0;32mLabel list : [ %s ]\033[0m', ', '.join(str(l) for l in label_list))
    logging.info('\033[0;32mDataset is ready: %s \033[0m', output_path)

    writer.close()

