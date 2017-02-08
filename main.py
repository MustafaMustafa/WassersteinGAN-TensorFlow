import os
import scipy.misc
import numpy as np
from glob import glob

import train
from model import DCGAN
from utils import pp, get_data_arr

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("lrD", 0.00005, "Learning rate critic/discriminator [0.00005]")
flags.DEFINE_float("lrG", 0.00005, "Learning rate generator [0.00005]")
flags.DEFINE_float("clamp", 0.01, "clamp range for Critic weights [0.01]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA]")
flags.DEFINE_boolean("preload_data", True, "Converts training dataset to a numpy array to read from disk once [True]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("tensorboard_run", "run_0", "Tensorboard run directory name [run_0]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("nc", 5, "number of critic updates [5]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(sess, 
                      dataset=FLAGS.dataset,
                      batch_size=FLAGS.batch_size,
                      output_size=FLAGS.output_size,
                      c_dim=FLAGS.c_dim)

        if FLAGS.is_train:
            if FLAGS.preload_data == True:
                data = get_data_arr(FLAGS)
            else:
                data = glob(os.path.join('./data', FLAGS.dataset, '*.jpg'))
            train.train_wasserstein(sess, dcgan, data, FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()
