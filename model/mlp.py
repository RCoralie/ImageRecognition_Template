"""
    A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
"""

import os
import time
import logging
import tensorflow as tf

from ..defaults import Config
from ..util.dataset_reader import DataRead

tf.logging.set_verbosity(tf.logging.ERROR)


class MultilayerPerceptron(object):

    def __init__(self,
                 batch_size,
                 initial_learning_rate,
                 steps_per_checkpoint,
                 max_checkpoints,
                 model_dir):

        # Parameters --------------------------------------------------------------------------------
        self.learning_rate = initial_learning_rate
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.steps_per_checkpoint = steps_per_checkpoint

        # Construct model ---------------------------------------------------------------------------
        # Network Parameters
        self.n_hidden_1 = 4800 # 1st layer number of neurons
        self.n_hidden_2 = 2400 # 2nd layer number of neurons
        self.n_input = Config.IMAGE_HEIGHT * Config.IMAGE_WIDTH # size of the data input (total pixels)
        self.n_classes = 36 # total classes (0-9 digits + A-Z chars)

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(self.X, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

        # Define loss and optimizer -----------------------------------------------------------------
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Define initializer and saver --------------------------------------------------------------
        self.checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=max_checkpoints)


    def train(self, dataset_path, num_epoch, load_model, log_step):

        logging.info('num_epoch: %d' % num_epoch)
        step_time = 0.0
        loss = 0.0
        current_step = 0

        logging.info('Starting the training process.')
        s_gen = DataRead(dataset_path, epochs=num_epoch)
        with tf.Session() as sess:
            # Initialize or load model parameters
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and load_model:
                logging.info('Reading model parameters from %s' % ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logging.info('Initialize model parameters.')
                sess.run(self.init)
            # Gradual improvement of model parameters during training steps
            for batch in s_gen.gen(self.batch_size):
                current_step += 1
                start_time = time.time()
                batch_x = batch['data']
                batch_y = batch['labels']
                feed_dict = {self.X: batch_x, self.Y: batch_y}
                _, loss_value = sess.run([self.train_op, self.loss_op], feed_dict)
                curr_step_time = (time.time() - start_time)

                # Print statistics for the previous epoch.
                logging.info('Step %04d - Time: %.3f, loss: %.4f' % (current_step, curr_step_time, loss_value))

                # Save checkpoint
                if current_step % self.steps_per_checkpoint == 0:
                    logging.info('Saving model parameters at step %d.' % current_step)
                    save_path = self.saver.save(sess, self.checkpoint_path, global_step=current_step)

        logging.info('Training Finished!')
        save_path = self.saver.save(sess, self.checkpoint_path, global_step=current_step)
        logging.info('Model stored.')

