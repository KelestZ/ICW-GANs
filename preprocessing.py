# -*- coding : utf-8
import csv
import numpy as np
from nilearn.image import load_img
import nilearn.masking as masking
import os
import pickle as pkl
data_path = '/data/zpy/BrainPedia/'
base_dir = '/home/nfs/zpy/BrainPedia/'
img_csv = 'images.tsv'



def tag_class(img_csv):
    '''

    :param img_csv: saved into pkl
    :return:
        Args:
            tags={'tag_name': tag_NumLabel}
            pic_dic ={'img_id':[img_tags]}
        e.g:
            tags={'visual':0,...}
            pic_dic={'36015':[0,1,2],...}
    '''

    tags = {}
    pic_dic = {}
    tag_count = 0
    tsv_reader = open(img_csv, 'r')
    # remove the head
    tsv_reader.readline()
    for row in tsv_reader:
        data = row.split('\t')
        id = data[19]
        tag = data[37].split(',')
        tag.remove(tag[-1])

        if ((str(tag) not in tags.keys()) and (tag != [])):
            tags[str(tag)] = tags.setdefault(str(tag), tag_count)
            tag_count += 1

        pic_dic.setdefault(id, -1)
        pic_dic[id] = tags[str(tag)]


    print(pic_dic)
    tagFile = open('multi_class_tag.pkl', 'wb')
    pkl.dump(tags, tagFile)
    tagFile.close()

    picFile = open('multi_class_pic_tags.pkl', 'wb')
    pkl.dump(pic_dic, picFile)
    picFile.close()

#tag_class(img_csv)

def msk(data_path):
    mskedFile = open('./pkl/msked_data.pkl', 'wb')
    collection = os.listdir(data_path)
    imgs = []
    imgs_id = []
    mean_msk = []

    count = 0
    for i in collection:
        if (str(i)[-3:] == 'nii'):
            count += 1
            imgs.append(load_img(data_path + i))
            imgs_id.append(i.replace('.nii',''))
            if (count % 10 == 0 or count == 6573): #
                mean_msk.append(masking.compute_background_mask(imgs))
                masked_datas = masking.apply_mask(imgs, mean_msk[-1])
                # { id: masked_data }
                #mskedFile.write(dict(zip(imgs_id, masked_datas)))
                #pkl.dump(dict(zip(imgs_id, masked_datas)), mskedFile )
                imgs=[]
                img_id=[]
                masked_data=[]
                print('write %d batch into msked_data.pkl' %count)
                if(count ==50):
                    break
    #
    msk = np.mean(np.array(mean_msk).astype(np.float64),0)
    print(msk)
    '''
    mskedFile.close()
    mskFile = open('msk.pkl', 'wb')
    pkl.dump(mean_msk, mskFile)
    print('write the msk list into msk.pkl')
    
    mskFile.close()
    '''
msk(data_path)



'''
tagFile = open('multi_class_tag.pkl', 'rb')
tags = pkl.load(tagFile)
print(tags)
tagFile.close()
'''

'''
mskFile = open('./pkl/msked_data.pkl', 'rb')
msks = pkl.load(mskFile)

mskFile_2 = open('./pkl/msked_data_p2.pkl', 'wb')
pkl.dump(msks,mskFile_2,protocol=2)

mskFile_2.close()
mskFile.close()
'''

'''
mskFile = open('./pkl/vec_.pkl', 'rb')
msks = pkl.load(mskFile)

mskFile_2 = open('./pkl/msked_data_p2.pkl', 'wb')
pkl.dump(msks,mskFile_2,protocol=2)

mskFile_2.close()
mskFile.close()
print("done")
'''

'''
mskFile = open('./outmask_5.pkl', 'rb')

f =  open('./outmask_5_p2.pkl', 'wb')
temp = pkl.load(mskFile)
pkl.dump(temp,f,protocol=2)
f.close()
mskFile.close()
print('done')
'''

#tag_class(img_csv)