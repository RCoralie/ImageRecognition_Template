"""
    The recommended way of reading the data in TensorFlow however is through the dataset API. Indeed, if you need to read your data from file, it may be more efficient to write it in TFrecord format and use TFRecordDataset to read it. Dataset API allows you to make efficient data processing pipelines easily.

    For more informations see :
    https://www.tensorflow.org/programmers_guide/datasets
"""

import numpy as np
import tensorflow as tf

from ..util import process_raw_data


class DataRead(object):


    def __init__(self,
                 dataset_path,
                 epochs=1000):
        """
        Create the dataset that will be used to build the input data of the learning algorithm
        """

        # Prepare batch structure
        self.epochs = epochs
        self.data_list = []
        self.label_list = []

        # Create the dataset that will be used to build the input data of the learning algorithm
        # -1- Define a reader and creates a dataset that reads all of the examples from the file
        self.dataset = tf.data.TFRecordDataset([dataset_path])
        # -2- Perform preprocessing on raw records and convert the data to a usable format for the model
        self.dataset = self.dataset.map(self._parse_record)
        # -3- Create a dataset that repeats its input for many epochs
        self.dataset = self.dataset.repeat(self.epochs)
        # -4- Randomly shuffles the elements of this dataset to get batches with different sample distributions
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        # -5- Then create batches of samples by calling gen(self, batch_size) ...


    def clear(self):
        self.data_list.clear()
        self.label_list.clear()


    def _parse_record(self, example_proto):
        """
        decoder is needed to decode the record read by the reader. In case of using TFRecords files the decoder should be tf.parse_single_example. it takes a serialized Example and a dictionary which maps feature keys to FixedLenFeature or VarLenFeature values and returns a dictionary which maps feature keys to Tensor values
        """
        # Define a decoder : parses a single tf.Example into image and label tensors
        features = tf.parse_single_example(
                                           example_proto,
                                           features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'height' : tf.FixedLenFeature([], tf.int64),
                                           'width' : tf.FixedLenFeature([], tf.int64)
                                           })

        return features['image'], features['label'], features['height'], features['width']


    def gen(self, batch_size):
        """
        generator function that produces sequence of training data
            def generator():
            ...
            yield (X, y)
        """
        # Create batches of samples
        dataset = self.dataset.batch(batch_size)

        # Creates an Iterator for enumerating the elements of this dataset.
        iterator = dataset.make_one_shot_iterator()
        images, labels, heights, widths = iterator.get_next()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            while True:
                try:
                    # Advance the iterator and get the next elements
                    raw_image, raw_label, raw_height, raw_width = sess.run([images, labels, heights, widths])
                    for img, label, h, w in zip(raw_image, raw_label, raw_height, raw_width):
                        # Preprocess label
                        word = process_raw_data.process_label(label)
                        # Preprocess img data
                        init_height = tf.cast(h, tf.int32)
                        init_width = tf.cast(w, tf.int32)
                        img = process_raw_data.process_png(img, init_height, init_width)
                        img = sess.run(img)
                        bucket_size = self.bucket_append(img, word)
                        if bucket_size >= batch_size:
                            bucket = self.flush_out()
                            yield bucket

                except tf.errors.OutOfRangeError:
                    # The iterator reaches the end of the dataset
                    break

        self.clear()


    def bucket_append(self, img, label):
        """
        bucket_append is needed to add data into bucket data
        """
        self.data_list.append(img)
        self.label_list.append(label)
        
        return len(self.data_list)


    def flush_out(self):
        res = {}
        res['data'] = np.array(self.data_list)
        res['labels'] = np.array(self.label_list)
        self.data_list, self.label_list = [], []
        return res

