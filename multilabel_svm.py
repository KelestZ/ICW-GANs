from sklearn import svm
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like, resample_img
import pickle as pkl
from nilearn.image import load_img
import nibabel
import os
import math
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

import time
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn import svm, grid_search, datasets

from sklearn.multiclass import OneVsRestClassifier

data_dir = '/home/nfs/zpy/BrainPedia/pkl/'
data_pkl = 'outbraindata_4.0_p2.pkl'
train_list_dir = '/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/shuffle_1_1111_1/cost/train_data_list.pkl'
test_list_dir = '/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/shuffle_1_1111_1/cost/test_data_list.pkl'
dev_list_dir = '/home/nfs/zpy/BrainPedia/cwgan_4_bp_conditional/shuffle_1_1111_1/cost/dev_data_list.pkl'
tag_pkl = 'multi_class_pic_tags.pkl'


def pre_class_statics():
    f_data = open(data_dir + data_pkl, 'rb')
    f_tags = open(data_dir + tag_pkl, 'rb')
    data = pkl.load(f_data)
    tags = pkl.load(f_tags)
    train_list = pkl.load(open(train_list_dir, 'rb'))
    test_list = pkl.load(open(test_list_dir, 'rb'))
    dev_list = pkl.load(open(dev_list_dir, 'rb'))

    cross = {}

    f_data.close()
    f_tags.close()
    tag_imgs_dic = {}
    count = {}
    count_test = {}
    count_dev = {}
    train = {}
    test = {}
    dev = {}

    for img_id in train_list:

        # if(tags[img_id] == 37):
        # continue

        brain = data[img_id].get_data()
        _max = np.max(brain)
        _min = np.min(brain)
        # normalization
        outbrain = np.array([2 * ((brain - _min) / (_max - _min)) - 1])
        if (tags[img_id] not in train.keys()):
            train[tags[img_id]] = []
            count[tags[img_id]] = 0
            cross[tags[img_id]] = []
        train[tags[img_id]].append(outbrain.reshape([13, 15, 11, 1]))  # 26, 31, 23, 1]))#13,15,11,1
        count[tags[img_id]] += 1

    for img_id in test_list:

        if (tags[img_id] == 37):
            continue

        brain = data[img_id].get_data()
        _max = np.max(brain)
        _min = np.min(brain)

        # normalization
        outbrain = np.array([2 * ((brain - _min) / (_max - _min)) - 1])
        if (tags[img_id] not in test.keys()):
            test[tags[img_id]] = []
            count_test[tags[img_id]] = 0

        test[tags[img_id]].append(outbrain.reshape([13, 15, 11, 1]))  # [13,15,11,1]))#[26, 31, 23, 1]))
        count_test[tags[img_id]] += 1

    for img_id in dev_list:
        if (tags[img_id] == 37):
            continue
        brain = data[img_id].get_data()
        _max = np.max(brain)
        _min = np.min(brain)

        # normalization
        outbrain = np.array([2 * ((brain - _min) / (_max - _min)) - 1])
        if (tags[img_id] not in dev.keys()):
            dev[tags[img_id]] = []
            count_dev[tags[img_id]] = 0

        dev[tags[img_id]].append(outbrain.reshape([13, 15, 11, 1]))
        count_dev[tags[img_id]] += 1

    sum_1 = 0
    sum_2 = 0
    sum_3 = 0

    for i in count.values():
        sum_1 += int(i)
    for i in count_test.values():
        sum_2 += int(i)
    print(sum_1, sum_2)

    for i in count_dev.values():
        sum_3 += int(i)
    print(' train size: ', sum_1, '\n test size: ', sum_2, '\n dev size: ', sum_3)
    print(count_test)
    return train, test, dev

train, test, dev = pre_class_statics()

def classify(train, test):
    count = {}
    max_ = 0
    # for train
    data = list(train.values())
    datas = []
    for x in data:
        for j in x:
            datas.append(j.reshape(-1))
    train_targets = []  # no_shuffle_list
    for x in [[key] * len(train[key]) for key in train.keys()]:
        train_targets += x

    for key in train.keys():
        if key not in count.keys():
            count[key] = 0
        # rememer training data size for each class
        if (len(train[key]) > max_):
            max_ = len(train[key])
        count[key] = len(train[key])

    # cal proportion
    for i in count.keys():
        count[i] = (count[i] / float(max_))

    # for test
    test_data = list(test.values())
    test_datas = []
    for x in test_data:
        for j in x:
            test_datas.append(j.reshape(-1))
    test_targets = []
    for x in [[key] * len(test[key]) for key in test.keys()]:
        test_targets += x

    # cross validation
    # split dataset
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(test_datas, test_targets)

    f1_list = []
    p_list = []
    r_list = []
    # K-fold interation
    # svr = svm.SVC()
    # clf = grid_search.GridSearchCV(svr, {'kernel':('linear','rbf'),'C':[1,10]})
    f = open('/home/nfs/zpy/BrainPedia/pkl/cvt_45_19.pkl', 'rb')
    cvt_45_19 = pkl.load(f)
    f.close()
    # cvt [0,1,2] to one-hot
    cvt_19_one_hot = np.zeros((45, 19))

    for k in cvt_45_19.keys():
        for i, label in enumerate(cvt_45_19[k]):
            cvt_19_one_hot[i, label] = 1.0

    cm = []

    ct = 0
    for train_, test_ in skf.split(test_datas, test_targets):
        ct += 1
        print('Train', len(train_), 'Test', len(test_))
        x_tr, x_te = np.array(test_datas)[train_], np.array(test_datas)[test_]
        y_tr, y_te = np.array(test_targets)[train_], np.array(test_targets)[test_]

        temp_tr_x = np.concatenate([np.array(datas), x_tr], 0)
        temp_tr_y = np.concatenate([np.array(train_targets), y_tr], 0)

        # predict
        # clf = svm.SVC(C=1.0, kernel='linear')  # class

        labels = np.zeros((len(temp_tr_y), 19))

        ctt = 0
        for m in temp_tr_y:
            labels[ctt] = (cvt_19_one_hot[m])
            ctt += 1
        classif = OneVsRestClassifier(SVC(kernel='linear'))
        classif.fit(temp_tr_x, labels)

        # print('params:',clf.get_params())

        target_names = []
        for i in range(36):
            target_names.append('class_' + str(i))
        '''
        if(ct ==3):
            m = len(x_te)%50
            results = clf.predict(x_te[:-m])
            a = confusion_matrix(y_te[:-m],results)
            print(classification_report(y_te[:-m], results, target_names=target_names))
            print(accuracy_score(y_te[:-m],results))

        else:
        '''

        results = classif.predict(x_te)
        print('score')

        '''
        clf = svm.SVC(C=1.0, kernel='linear')  # class
        clf.fit(temp_tr_x,temp_tr_y)  # training the svc model
        results_ori = clf.predict(x_te)
        print('original accuracy',accuracy_score(y_te,results_ori))
        '''
        test_labels = np.zeros((len(y_te), 19))  # 19 dimension
        ctt = 0
        for m in y_te:
            test_labels[ctt] = (cvt_19_one_hot[m])
            ctt += 1
        print('-----------------')
        print(accuracy_score(test_labels, results))
        cm.append(results)

    return cm

def downsample_brain(braindata,inmask, scaling=1.0):
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
    small_brain= resample_img(braindata, target_affine=new_affine,
                            target_shape=outshape, interpolation='continuous')

    # vec = outmasker.transform(small_brain)

    return small_brain # inmask,outmasker,vec

def load_gen_data(flag):
    # size_file = open(data_dir + 'record_size_per_tag','rb')
    # size= pkl.load(size_file)
    # size_file.close()

    not_good_list = [3, 6, 10, 16, 22, 25, 28, 30, 31, 33]
    gen = {}
    count = {}
    mskFile = open(data_dir + '/msk.pkl', 'rb')
    inmask = pkl.load(mskFile)
    if flag == 'conditional':
        test_dir = '/home/nfs/zpy/BrainPedia/cwgan_4_cdtion_mlti_label/nn_multi_label_1111_1/test/'
        gen_pics = os.listdir(test_dir)
        for i in gen_pics:

            if (i[-2:] == 'gz'):
                tag = int(i.split('.')[0].split('_')[3])
                iter_ = int(i.split('.')[0].split('_')[1])
                if tag not in gen.keys():
                    gen[tag] = []
                    count[tag] = 0

                if (count[tag] <= 19):
                    brain = load_img(test_dir + i)
                    down_img = downsample_brain(brain, inmask, 4.0)

                    brain = down_img.get_data()

                    _max = np.max(brain)
                    _min = np.min(brain)
                    outbrain = np.array([2 * ((brain - _min) / (_max - _min)) - 1])

                    gen[tag].append(outbrain.reshape([13, 15, 11, 1]))  # [26, 31, 23, 1]))
                    count[tag] += 1

        print(count)
        sum_ = 0
        for i in count.keys():
            sum_ += count[i]

        print(sum_)
    return gen

'''
gen = load_gen_data('conditional')

cross_gen_true = {}
for i in gen.keys():
    if (i not in cross_gen_true.keys()):
        cross_gen_true[i] = []
        cross_gen_true[i] = gen[i] + train[i]
'''

f = open('./pkl/multi_label_cross.pkl','rb')
cross_gen_true = pkl.load(f)
f.close()

print('done')
cm=classify(cross_gen_true, test)