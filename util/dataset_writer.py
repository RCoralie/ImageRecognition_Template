"""
Generate dataset in tfrecords format.
TFRecords are TensorFlow’s default data format. A record is simply a binary file that contains serialized tf.train.Example Protobuf objects.

Basically, an Example always contains Features.
Features contains a map of strings to Feature.
And finally, a Feature contains one of a FloatList, a ByteList or a Int64List.

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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def writeAnnotation(input_dir_path, output_file_path, log_step):
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


def shuffle(file_path) :
    data = []
    with open(file_path,'r') as source:
        data = [ (random.random(), line) for line in source ]
        data.sort()
    with open(file_path,'w+') as target:
        for _, line in data:
            target.write( line )

def generateDataset(annotations_path, output_path, log_step):
    logging.info(' - Generate binary dataset - ')
    logging.info('Input file: %s', annotations_path)

    writer = tf.python_io.TFRecordWriter(output_path)

    longest_label = ''

    with open(annotations_path,'r') as source:
        # one MUST randomly shuffle data before putting it into tfrecords formats.
        data = [ (random.random(), line) for line in source ]
        data.sort()
        # TODO : divide the data set into three train (%60), validation (%20), and test parts (%20).
        # iterate over each and construct the Example proto oject
        for idx, line in enumerate(data):
            line = line[1].rstrip('\n')
            try:
                (img_path, label) = line.split('\t', 1)
            except ValueError:
                logging.error('\033[0;31m missing filename or label at line %i: %s (ignored data)\033[0m', idx+1, line)
                continue

            with open(img_path, 'rb') as img_file:
                img = img_file.read()

            if len(label) > len(longest_label):
                longest_label = label

            # A Feature contains a map of string to Feature proto objects which are one of either a int64_list, float_list, or bytes_list
            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(b(label))

            # Example contains a Features proto object
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # use the proto object to serialize the example to a string and write the serialized object to disk
            writer.write(example.SerializeToString())

            if idx % log_step == 0:
                logging.info('\033[0;37mProcessed %i pairs\033[0m', idx+1)

    logging.info('\033[0;32mProcessed %i pairs\033[0m', idx+1)
    logging.info('\033[0;32mLongest label (%i): %s\033[0m', len(longest_label), longest_label)
    logging.info('\033[0;32mDataset is ready: %s \033[0m', output_path)

    writer.close()

