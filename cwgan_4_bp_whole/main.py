# -*- coding : utf-8
# -*- coding: utf-8 -*-
import os
import scipy.misc
import numpy as np
from model import CIWGAN
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("train", "True", "train")
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]='3'
def main(_):


    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        ciwgan = CIWGAN(
                sess,
                batch_size=50,
                y_dim=45,
                checkpoint_dir=FLAGS.checkpoint_dir)


        if FLAGS.train:
            print("[INFO] Training mode")
            ciwgan.train()

        else:
            if not ciwgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

            #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()

