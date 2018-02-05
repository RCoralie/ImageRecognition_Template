import sys
import argparse
import logging
import tensorflow as tf

from .defaults import Config
from .util import dataset_writer
from .model.mlp import MultilayerPerceptron

tf.logging.set_verbosity(tf.logging.ERROR)


def process_args(args, defaults):

    parser = argparse.ArgumentParser()
    parser.prog = 'imageRecognition'
    subparsers = parser.add_subparsers(help='Subcommands.')

    # Global arguments
    parser.add_argument('--log-path', dest='log_path',
                        type=str, default=defaults.LOG_PATH,
                        help=('Log file path, default=%s' % (defaults.LOG_PATH)))
    parser.set_defaults(load_model=defaults.LOAD_MODEL)
    parser.set_defaults(model_dir=defaults.MODEL_DIR)

    # Labeling generation
    parser_annote = subparsers.add_parser('annote', help='Create the labeling file in txt format.')
    parser_annote.set_defaults(phase='annote')
    parser_annote.add_argument('input_dir_path', nargs='?', metavar='input_dir',
                                type=str,
                                help=('Path to the input directory'))
    parser_annote.add_argument('output_file_path', metavar='output_file',
                              type=str,
                              help=('Path to the output file'))
    parser_annote.add_argument('--log-step', dest='log_step',
                                   type=int, default=defaults.LOG_STEP,
                                   help=('Print log messages every N steps, default = %s' % defaults.LOG_STEP))

    # Dataset generation
    parser_dataset = subparsers.add_parser('dataset', help='Create a dataset in the TFRecords format.')
    parser_dataset.set_defaults(phase='dataset')
    parser_dataset.add_argument('annotations_path', metavar='annotations_path',
                              type=str,
                              help=('Path to the annotation file'))
    parser_dataset.add_argument('output_path', nargs='?', metavar='output',
                              type=str, default=defaults.DATASET_PATH,
                              help=('Output path, default=%s' % (defaults.DATASET_PATH)))
    parser_dataset.add_argument('--log-step', dest='log_step',
                              type=int, default=defaults.LOG_STEP,
                              help=('Print log messages every N steps, default = %s' % defaults.LOG_STEP))

    # Train model
    parser_train = subparsers.add_parser('train', help='Train model.')
    parser_train.set_defaults(phase='train')
    parser_train.add_argument('dataset_path', metavar='dataset_path',
                                type=str,
                                help=('Path to the dataset (tfrecords) file'))
    parser_train.add_argument('--no-resume', dest='load_model', action='store_false',
                              help=('Create an empty model even if checkpoints already exist, default=%s' % (defaults.LOAD_MODEL)))
    parser_train.add_argument('--max-checkpoints', dest='max_checkpoints',
                              type=int, default=defaults.MAX_CHECKPOINTS,
                              help=('Number max of checkpoints saved, default=%s' % (defaults.MAX_CHECKPOINTS)))
    parser_train.add_argument('--num-epoch', dest='num_epoch',
                              type=int, default=defaults.NUM_EPOCH,
                              help=('Number of epochs, default = %s' % (defaults.NUM_EPOCH)))
    parser_train.add_argument('--batch-size', dest='batch_size',
                              type=int, default=defaults.BATCH_SIZE,
                              help=('Batch size, default = %s' % (defaults.BATCH_SIZE)))
    parser_train.add_argument('--initial-learning-rate', dest='initial_learning_rate',
                      type=float, default=defaults.INITIAL_LEARNING_RATE,
                      help=('Initial learning rate, default = %s' % (defaults.INITIAL_LEARNING_RATE)))
    parser_train.add_argument('--steps-per-checkpoint', dest='steps_per_checkpoint',
                            type=int, default=defaults.STEPS_PER_CHECKPOINT,
                            help=('Checkpointing (print perplexity, save model), default = %s' % (defaults.STEPS_PER_CHECKPOINT)))
    parser_train.add_argument('--log-step', dest='log_step',
                                type=int, default=defaults.LOG_STEP,
                                help=('Print log messages every N steps, default = %s' % defaults.LOG_STEP))

    parameters = parser.parse_args(args)
    return parameters


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parameters = process_args(args, Config)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    if parameters.phase == 'annote':
        dataset_writer.writeAnnotation(parameters.input_dir_path, parameters.output_file_path, parameters.log_step)
        return

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #Must allow soft placement to place on CPU an op if :
    #1. there's no GPU implementation for the OP
    #2. no GPU devices are known or registered
    #3. need to co-locate with reftype input(s) which are from CPU.

        if parameters.phase == 'dataset':
            dataset_writer.generateDataset(parameters.annotations_path, parameters.output_path, parameters.log_step)
            return
        elif parameters.phase == 'train':
            model = MultilayerPerceptron(batch_size = parameters.batch_size, initial_learning_rate=parameters.initial_learning_rate, steps_per_checkpoint = parameters.steps_per_checkpoint, max_checkpoints = parameters.max_checkpoints, model_dir=parameters.model_dir)
            model.train(parameters.dataset_path, parameters.num_epoch, parameters.load_model, parameters.log_step)
            return
        else:
            raise NotImplementedError

if __name__ == "__main__":
    main()
