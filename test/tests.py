from ..defaults import Config
from ..util import dataset_writer
from ..util import process_raw_data
from ..util.dataset_reader import DataRead

from PIL import Image
import numpy as np
import tensorflow as tf
import random
import unittest
import os

class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.dir_path = os.path.abspath(os.path.split(__file__)[0]) + '/'


    def test_grayscale_dataset(self):

        input_dir_path = self.dir_path+'test_data/testDataset/'
        annotations_path = self.dir_path + 'test.txt'
        output_path = self.dir_path + 'test.tfrecords'

        A = np.zeros((len(process_raw_data.CHARMAP),), dtype=float)
        A[process_raw_data.CHARMAP.index('A')] = 1.

        B = np.zeros((len(process_raw_data.CHARMAP),), dtype=float)
        B[process_raw_data.CHARMAP.index('B')] = 1.

        # Writing the txt file ------------------------------------------------------
        dataset_writer.writeAnnotation(input_dir_path, annotations_path, 1)
        ## Test existence of file
        self.assertTrue(os.path.isfile(annotations_path))
        ## Test that all data are correctly written into txt file
        error = False
        init_heights = []
        init_widths = []
        init_labels = []

        with open(annotations_path,'r') as source:
            for idx, line in enumerate(source):
                line = line.rstrip('\n')
                try:
                    (img_path, label) = line.split('\t', 1)
                except ValueError:
                    error = True
                    continue
                self.assertTrue(os.path.isfile(img_path))
                img = np.array(Image.open(img_path))
                init_heights.append(img.shape[0])
                init_widths.append(img.shape[1])
                init_labels.append(label)

        self.assertEqual(init_labels.count('A'),10)
        self.assertEqual(init_labels.count('B'),11)
        self.assertEqual(len(init_heights),21)
        self.assertEqual(len(init_widths),21)
        self.assertEqual(error,False)

        # Writing the dataset file --------------------------------------------------
        dataset_writer.generateDataset(annotations_path, output_path, 1)
        ## Test existence of file
        self.assertTrue(os.path.isfile(output_path))

        # Reading the dataset -------------------------------------------------------
        BATCH_SIZE = 5
        data_generated = DataRead(output_path, epochs=1)
        with tf.Session() as sess:
            for batch in data_generated.gen(BATCH_SIZE):
                x = batch['data']
                y = batch['labels']
                self.assertEqual(x.shape[0],BATCH_SIZE)
                self.assertEqual(y.shape[0],BATCH_SIZE)
                self.assertEqual(x.shape[1],(Config.IMAGE_HEIGHT*Config.IMAGE_WIDTH))
                self.assertEqual(y.shape[1],len(process_raw_data.CHARMAP))

        # Remove tmp files ----------------------------------------------------------
        os.remove(annotations_path)
        os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
