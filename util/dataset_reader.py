"""
    The recommended way of reading the data in TensorFlow however is through the dataset API. Indeed, if you need to read your data from file, it may be more efficient to write it in TFrecord format and use TFRecordDataset to read it. Dataset API allows you to make efficient data processing pipelines easily.

    For more informations see :
    https://www.tensorflow.org/programmers_guide/datasets
"""

import numpy as np
import tensorflow as tf
import sys

from PIL import Image
from six import BytesIO as IO


class DataRead(object):

    # Utils needed to convert label into array of int
    GO_ID = 1
    EOS_ID = 2
    CHARMAP = ['', '', ''] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')


    def __init__(self,
                 annotation_fn,
                 specs,
                 epochs=1000,
                 max_width=None):
        """
        Create the dataset that will be used to build the input data of the learning algorithm
        """

        # Prepare batch structure for learning algorithme input data
        self.data_specs = specs
        self.max_width = max_width
        self.data_list = []
        self.label_list = []
        self.label_list_plain = []
        self.epochs = epochs

        # Create the dataset that will be used to build the input data of the learning algorithm
        # -1- Define a reader and creates a dataset that reads all of the examples from the file
        dataset = tf.data.TFRecordDataset([annotation_fn])
        # -2- Perform preprocessing on raw records and convert the data to a usable format for the model
        dataset = dataset.map(self._parse_record)
        # -3- Create a dataset that repeats its input for many epochs
        self.dataset = dataset.repeat(self.epochs)
        # -4- Randomly shuffles the elements of this dataset to get batches with different sample distributions
        dataset = dataset.shuffle(buffer_size=10000)
        # -5- Then create batches of samples by calling gen(self, batch_size) ...


    def clear(self):
        self.data_list.clear()
        self.label_list.clear()
        self.label_list_plain.clear()


    @staticmethod
    def setFullAsciiCharmap():
        DataRead.CHARMAP = ['', '', ''] + [chr(i) for i in range(32, 127)]

    @staticmethod
    def _parse_record(example_proto):
        """
        decoder is needed to decode the record read by the reader. In case of using TFRecords files the decoder should be tf.parse_single_example. it takes a serialized Example and a dictionary which maps feature keys to FixedLenFeature or VarLenFeature values and returns a dictionary which maps feature keys to Tensor values
        """
        # Define a decoder : parses a single tf.Example into image and label tensors
        features = tf.parse_single_example(
                                           example_proto,
                                           features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string),
                                           })

        # Perform additional preprocessing on the parsed data (decode_jpg,, reshape, cast, etc) ...

        return features['image'], features['label']



    def gen(self, batch_size):
        """
        gen is needed to create learning algorithme input data from dataset
        """
        # Create batches of samples
        dataset = self.dataset.batch(batch_size)

        # Creates an Iterator for enumerating the elements of this dataset.
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            while True:
                try:
                    # Advance the iterator and get the next elements
                    raw_images, raw_labels = sess.run([images, labels])
                    
                    for img, lex in zip(raw_images, raw_labels):

                        if self.max_width and (Image.open(IO(img)).size[0] <= self.max_width):
                            word = self.convert_lex(lex)
                            bucket_size = self.bucket_append(img, word, lex)
                            if bucket_size >= batch_size:
                                bucket = self.flush_out(self.data_specs, go_shift=1)
                                yield bucket

                except tf.errors.OutOfRangeError:
                    # The iterator reaches the end of the dataset
                    break

        self.clear()


    def convert_lex(self, lex):
        """
        convert_lex is needed to convert label to an array of int
        """
        if sys.version_info >= (3,):
            lex = lex.decode('iso-8859-1')
        assert len(lex) < self.data_specs[-1][1]
        return np.array(
            [self.GO_ID] + [self.CHARMAP.index(char) for char in lex] + [self.EOS_ID],
            dtype=np.int32)


    def bucket_append(self, img, label, label_plain):
        """
        bucket_append is needed to add data into bucket data
        """
        self.data_list.append(img)
        self.label_list.append(label)
        self.label_list_plain.append(label_plain)
        
        return len(self.data_list)


    def flush_out(self, data_specs, valid_target_length=float('inf'), go_shift=1):
        res = {}
        decoder_input_len = data_specs[0][1]

        # ENCODER PART
        res['data'] = np.array(self.data_list)
        res['labels'] = self.label_list_plain

        # DECODER PART
        target_weights = []
        for l_idx in range(len(self.label_list)):
            label_len = len(self.label_list[l_idx])
            if label_len <= decoder_input_len:
                self.label_list[l_idx] = np.concatenate((
                                                         self.label_list[l_idx],
                                                         np.zeros(decoder_input_len - label_len,dtype=np.int32)
                                                         ))
                one_mask_len = min(label_len - go_shift, valid_target_length)
                target_weights.append(np.concatenate((
                                                      np.ones(one_mask_len, dtype=np.float32),
                                                      np.zeros(decoder_input_len - one_mask_len, dtype=np.float32))
                                                     ))
            else:
                raise NotImplementedError

        res['decoder_inputs'] = [a.astype(np.int32) for a in np.array(self.label_list).T]
        res['target_weights'] = [a.astype(np.float32) for a in np.array(target_weights).T]

        assert len(res['decoder_inputs']) == len(res['target_weights'])

        self.data_list, self.label_list, self.label_list_plain = [], [], []

        return res


    def __len__(self):
        return len(self.data_list)


    def __iadd__(self, other):
        self.data_list += other.data_list
        self.label_list += other.label_list
        self.label_list_plain += other.label_list_plain


    def __add__(self, other):
        res = BucketData()
        res.data_list = self.data_list + other.data_list
        res.label_list = self.label_list + other.label_list
        res.label_list_plain = self.label_list_plain + other.label_list_plain
        return res

