#!/usr/bin/env python2 
# -*- coding: utf-8 -*- 
"""
Default parameters
"""


class Config:

    # I/O
    LOG_PATH = 'ocr.log'
    LOG_STEP = 100
    DATASET_PATH = 'data.tfrecords'

    # Save config
    STEPS_PER_CHECKPOINT = 25
    MODEL_DIR = 'checkpoints'
    LOAD_MODEL = True
    MAX_CHECKPOINTS = 4
