#!/usr/bin/env python2 
# -*- coding: utf-8 -*- 
"""
Default parameters
"""
import ConfigParser

class ModelConfig(object):

    def __init__(self,
                 file):

        config = ConfigParser.ConfigParser()
        config.readfp(open(file))

        # Labels
        self.CHARMAP = config.get('label', 'list')

        # Data info
        self.IMAGE_HEIGHT = int(config.get('image', 'height'))
        self.IMAGE_WIDTH = int(config.get('image', 'width'))

        # Optimization
        self.NUM_EPOCH = int(config.get('optimization', 'num_epoch'))
        self.BATCH_SIZE = int(config.get('optimization', 'batch_size'))
        self.INITIAL_LEARNING_RATE = float(config.get('optimization', 'initial_learning_rate'))