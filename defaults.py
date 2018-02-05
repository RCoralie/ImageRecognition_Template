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

    # Optimization
    NUM_EPOCH = 1000
    BATCH_SIZE = 65
    INITIAL_LEARNING_RATE = 0.001 #1.0

    # Save config
    STEPS_PER_CHECKPOINT = 1#100
    MODEL_DIR = 'checkpoints'
    LOAD_MODEL = True
    MAX_CHECKPOINTS = 4
