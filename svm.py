from sklearn import svm
import numpy as np
import nilearn.plotting as plot
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like, resample_img
import pickle as pkl
from nilearn.image import load_img
from matplotlib import pyplot as plt
import nibabel
import math
from sklearn.svm import SVC
import time
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
import os

data_dir = '/home/nfs/zpy/BrainPedia/pkl/'
data_pkl = 'outbraindata_4.0.pkl'
tag_pkl = 'multi_class_pic_tags.pkl'
smallmask_pkl = 'outmask_4.0.pkl'
tag_list = [17, 18, 20, 25]


def pre_class_statics():
    f_data = open(data_dir + data_pkl, 'rb')
    f_tags = open(data_dir + tag_pkl, 'rb')
    data = pkl.load(f_data)
    tags = pkl.load(f_tags)
    f_data.close()
    f_tags.close()
    tag_imgs_dic = {}
    count = {}

    for img_id in data.keys():

        if tags[img_id] in tag_list:
            if tags[img_id] not in tag_imgs_dic.keys():
                tag_imgs_dic[tags[img_id]] = []
            if tags[img_id] not in count.keys():
                count[tags[img_id]] = 0
            if count[tags[img_id]]<100:
                tag_imgs_dic[tags[img_id]].append(data[img_id].get_data())
                count[tags[img_id]] += 1
    print(count)
    return tag_imgs_dic

def classify(tag_imgs_dic):
    svm = SVC(C=1., kernel="linear")
    data=list(tag_imgs_dic.values())
    datas=[]
    for x in data:
        for j in x:
            datas.append(j.reshape(-1))
    targets_temp = []
    for x in [[key] * len(tag_imgs_dic[key]) for key in tag_imgs_dic.keys()]:
        targets_temp += x
    #print(targets_temp)
    #print(len(datas),len(targets_temp))
    for key in tag_imgs_dic.keys():
        target = [targets_temp[i] == key for i in range(len(targets_temp))]
        scores = cross_val_score(svm, np.array(datas), np.array(target),scoring="f1")#,
        print('key',key,'  f1_score',scores.mean(),'  f1_score_std',scores.std())


def downsample_brain(braindata, inmask, scaling=1.0):
    # initial mask

    # rescaling
    outshape = tuple([int(float(x) / scaling) for x in inmask.shape])
    realscale = float(inmask.shape[0]) / float(outshape[0])
    new_affine = inmask.get_affine().copy()
    new_affine[:3, :3] *= realscale
    '''
    # resample mask
    outmask = resample_img(inmask, target_affine=new_affine,
                           target_shape=outshape, interpolation='nearest')
    outmasker = NiftiMasker(mask_img=outmask, standardize=False)
    outmasker.fit()
    '''
    # resample image
    small_brain = resample_img(braindata, target_affine=new_affine,
                               target_shape=outshape, interpolation='continuous')

    # vec = outmasker.transform(small_brain)

    return small_brain  # inmask,outmasker,vec

def load_gen_data(tags):
    dic_ = {}
    mskFile = open(data_dir + '/msk_p2.pkl', 'rb')
    inmask = pkl.load(mskFile)
    for i in tags:
        test_dir = '/home/nfs/zpy/BrainPedia/cwgan_4_bp_whole/results/test' + str(i) + '/'
        gen_pics = os.listdir(test_dir)
        count = 0
        for img_name in gen_pics:
            brain = load_img(test_dir + img_name)
            down_img = downsample_brain(brain, inmask, 4.0)
            if i not in dic_.keys():
                dic_[i] = []
            dic_[i].append(down_img.get_data())
            count += 1

            if (count == 30):
                print(i, 'ok')
                break
    return dic_

def classify2():
    tag_imgs_dic = pre_class_statics()
    print('tag_imgs_dic.keys():',tag_imgs_dic.keys())
    for i in tag_imgs_dic.keys():
        print(i, ':', len(tag_imgs_dic[i]))
    data = list(tag_imgs_dic.values())
    datas = []
    for x in data:
        for j in x:
            datas.append(j.reshape(-1))
    targets_temp = []
    for x in [[key] * len(tag_imgs_dic[key]) for key in tag_imgs_dic.keys()]:
        targets_temp += x

    clf = svm.SVC(C=1., kernel="linear")  # class
    clf.fit(datas, targets_temp)  # training the svc model
    return clf

def test_gen_pic(tags,clf):

    dic_ = load_gen_data(tags)
    print('here', dic_.keys(), tags)

    data_ = list(dic_.values())
    datas_ = []
    for x in data_:
        for j in x:
            datas_.append(j.reshape(-1))
    targets_temp = []
    for x in [[key] * len(dic_[18]) for key in tags]:
        targets_temp += x

    result = clf.predict(datas_)  # predict the target of testing samples
    print(result)  # target

    count = 0
    for i in range(len(result)):
        if (result[i] == targets_temp[i]):
            count += 1
    print(len(datas_))
    print(float(count)/float(len(datas_)))



clf = classify2()
test_gen_pic([17,18, 20,25], clf)