"""
Default parameters
"""


class Config:

    # I/O
    LOG_PATH = 'ocr.log'
    LOG_STEP = 100
    DATASET_PATH = 'data.tfrecords'

    # Data info
    COLOR_SPACE = 'GRAY'
    CHANNELS = 1
    FORMAT = 'PNG'
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 60
