"""
Default parameters
"""


class Config:

    # I/O
    LOG_PATH = 'ocr.log'
    LOG_STEP = 100
    DATASET_PATH = 'data.tfrecords'

    # Data info
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    # Labels
    CHARMAP = list('0123456789')

    # Optimization
    NUM_EPOCH = 1000
    BATCH_SIZE = 65
    INITIAL_LEARNING_RATE = 0.1 #1.0

    # Save config
    STEPS_PER_CHECKPOINT = 25
    MODEL_DIR = 'checkpoints'
    LOAD_MODEL = True
    MAX_CHECKPOINTS = 4
