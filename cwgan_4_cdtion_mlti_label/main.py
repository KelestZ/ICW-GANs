# -*- coding : utf-8
# -*- coding: utf-8 -*-
import os
import scipy.misc
import numpy as np
from model import CIWGAN
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("train", "True", "train")
flags.DEFINE_string("gpu", "3", "gpu_id")
flags.DEFINE_string("cond", "11111", "conditional")
flags.DEFINE_string("concat", "1", "concate method")


FLAGS = flags.FLAGS
checkpoint_dir = '/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/test_ckpt' + '_' + FLAGS.cond + '_' + FLAGS.concat+'/ckpt'
sample_dir = '/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/test_ckpt' + '_' + FLAGS.cond + '_' + FLAGS.concat+'/samples'
test_dir = '/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/test_ckpt' + '_' + FLAGS.cond + '_' + FLAGS.concat+'/test'
cost_dir = '/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/test_ckpt' + '_' + FLAGS.cond + '_' + FLAGS.concat+'/cost'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(cost_dir):
    os.makedirs(cost_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
def main(_):

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        ciwgan = CIWGAN(
                sess,
                BATCH_SIZE=50,
                y_dim=45,
                checkpoint_dir=checkpoint_dir,
                FLAGS=FLAGS,
                sample_dir=sample_dir,
                test_dir=test_dir,
                cost_dir=cost_dir)


        if FLAGS.train:
            print("[INFO] Training mode")
            ciwgan.train()

        else:
            if not ciwgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

            #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()

