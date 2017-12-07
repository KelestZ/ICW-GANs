#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.
Usage:
python msssim.py --original_image=original.png --compared_image=distorted.png
"""
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf
# print images
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like, resample_img
import pickle as pkl
from nilearn.image import load_img
import os
from sklearn.utils import shuffle
tf.flags.DEFINE_string('original_image', None, 'Path to PNG image.')
tf.flags.DEFINE_string('compared_image', None, 'Path to PNG image.')
FLAGS = tf.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def _FSpecialGauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = np.mgrid[offset + start:stop, offset + start:stop]
  assert len(x) == size
  g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
  """Return the Structural Similarity Map between `img1` and `img2`.
  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  _, height, width, _ = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
    sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  v1 = 2.0 * sigma12 + c2
  v2 = sigma11 + sigma22 + c2
  ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  cs = np.mean(v1 / v2)
  return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
  """Return the MS-SSIM score between `img1` and `img2`.
  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
  Returns:
    MS-SSIM score between `img1` and `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
  levels = weights.size
  downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
  im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
  mssim = np.array([])
  mcs = np.array([])
  for _ in range(levels):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2)
    mssim = np.append(mssim, ssim)
    mcs = np.append(mcs, cs)
    filtered = [convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]]
    im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
  return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
          (mssim[levels-1] ** weights[levels-1]))


def pre_class_statics(data):
    data_dir = '/home/nfs/zpy/BrainPedia/pkl/'
    if data is None:
        data_pkl = 'outbraindata_4.0_p2.pkl'
        f_data = open(data_dir + data_pkl, 'rb')
        data = pkl.load(f_data)
        f_data.close()

    tag_pkl = 'multi_class_pic_tags.pkl'
    f_tags = open(data_dir + tag_pkl, 'rb')
    tags = pkl.load(f_tags)

    train_list_dir = '/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/train_data_list.pkl'
    train_list = pkl.load(open(train_list_dir, 'rb'))

    f_tags.close()
    count = {}
    train = {}

    for img_id in train_list:
        if(img_id not in data.keys()):
            continue
        if (tags[img_id] not in train.keys()):
            train[tags[img_id]] = []
            count[tags[img_id]] = 0

        if(len(train[tags[img_id]])>=30):
            continue
        brain = data[img_id].get_data()
        _max = np.max(brain)
        _min = np.min(brain)
        # normalization
        outbrain = np.array([2 * ((brain - _min) / (_max - _min)) - 1])

        train[tags[img_id]].append(outbrain.reshape([13, 15, 11]))#[13,15,11]))  # 13,15,11,1
        count[tags[img_id]] += 1

    sum_1 = 0


    for i in count.values():
        sum_1 += int(i)

    print(' train size: ', sum_1)

    return train

def load_ckpt_data():

    print('[INFO] Load Test BrainPedia dataset...')#true_real
    train_file = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/cross.pkl', 'rb')
    train_dic = pkl.load(train_file)
    train_file.close()
    return train_dic

def pre_class_statics2():
    original_data={}

    data_dir = '/data/zpy/BrainPedia/'
    brain_data = os.listdir(data_dir)
    ct=0
    for i in brain_data:
        if (i[-3:] == 'nii'):
            img_name = i.replace('.nii', '')
            print(ct)
            ct+=1
            img = load_img(data_dir+i)
            original_data.setdefault(img_name)

            original_data[img_name]=img
        if(ct == 1000):
            break

    print('Done')
    return original_data

def main(_):
  '''
  if FLAGS.original_image is None or FLAGS.compared_image is None:
    print('\nUsage: python msssim.py --original_image=original.png '
          '--compared_image=distorted.png\n\n')
    return

  if not tf.gfile.Exists(FLAGS.original_image):
    print('\nCannot find --original_image.\n')
    return

  if not tf.gfile.Exists(FLAGS.compared_image):
    print('\nCannot find --compared_image.\n')
    return

  with tf.gfile.FastGFile(FLAGS.original_image) as image_file:
    img1_str = image_file.read()
  with tf.gfile.FastGFile(FLAGS.compared_image) as image_file:
    img2_str = image_file.read()
   '''
  '''
  train = load_ckpt_data()
  gen_dicF = open('/home/nfs/zpy/BrainPedia/cwgan_2_cond_w/for_nn_classify_2.0_100_3w5_11111_1/gen_dic.pkl', 'rb')

  train = pkl.load(gen_dicF)
  gen_dicF.close()
  '''
  #data = pre_class_statics2()
  train = load_ckpt_data()#load_ckpt_data()#pre_class_statics(None)
  input_img = tf.placeholder(tf.float32, shape=[26, 31, 23])#[26, 31, 23])#tf.placeholder(tf.string)
  decoded_image = tf.expand_dims(input_img, 0)

  print('Start cal')
  with tf.Session() as sess:
      ms_score={}
      ct = 0
      for tag in train.keys():
          score = []
          train[tag] = shuffle(train[tag])

          for i in range(len(train[tag])):
              if ((i + 1) == len(train[tag])):
                  break
              for k in range(i + 1, len(train[tag])):
                  img1_ = train[tag][i].reshape([26, 31, 23])#[26, 31, 23])#[13, 15, 11])
                  img2_ = train[tag][k].reshape([26, 31, 23])#[26, 31, 23])#[13, 15, 11])
                  img1 = sess.run(decoded_image, feed_dict={input_img: img1_})
                  img2 = sess.run(decoded_image, feed_dict={input_img: img2_})
                  sc = MultiScaleSSIM(img1, img2, max_val=255)
                  score.append(sc)
          print('tag:',tag,'score: ',np.mean(np.array(score)))
          ms_score[tag] = 0.
          ms_score[tag] = np.mean(np.array(score))
          ct+=1
          if(ct == 4950):
              break
      f =open('./ms_ssim/2_r_g_ms_ssim.pkl', 'wb')
      pkl.dump(ms_score,f)
      f.close()
      print('Done')


if __name__ == '__main__':
  tf.app.run()