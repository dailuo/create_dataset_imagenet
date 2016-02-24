from __future__ import division

import sys, os, time, math
import ipdb
import tensorflow as tf
import joblib
import numpy as np
from scipy import misc

from scipy.io import loadmat


log_path = './log/log_%s.txt' %time.strftime("%Y%m%d_%H%M%S")


def write_log(log_path, file_path):
    log_file = open(log_path, 'a')
    log_file.write('DAMAGED IMAGE:'+file_path+'\n')
    log_file.close()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_images_train(data_path, meta_path): # for train
    # data_path is the train set path
    # list path is the meta.txt path
    list_file = open(meta_path, 'r')
    image_list = []
    while True:
        line = list_file.readline()
        if not line:
            break
        output = line.split()
        WNID = output[1]
        label = output[0]
        root = os.path.join(data_path, WNID)
        file_list = os.listdir(root)
        for filename in file_list:
            file_path = os.path.join(root, filename)
            try:
                image = misc.imread(file_path)
            except:
                print('WARNING! UNABLE TO OPEN IMAGE:%s' %filename)
                write_log(log_path, file_path)
                continue
            image_list.append({'image':image, 'file_path':file_path, 'label':label})
    list_file.close()
    return image_list



def read_images_validation(data_path, val_label_path, val_blacklist_path):
    # for validation
    file_list = os.listdir(data_path)
    file_list.sort()

    val_label_file = open(val_label_path, 'r')
    val_label = val_label_file.readlines()
    val_label_file.close()

    #discard unqualified images
    val_blacklist_file = open(val_blacklist_path, 'r')
    val_blacklist = val_blacklist_file.readlines()
    val_blacklist_file.close()
    for index, item in enumerate(val_blacklist):
        val_blacklist[index] = int(item)

    image_list = []

    for (filename, sequence) in zip(file_list,xrange(len(file_list))):
        # discard unqualified images
        if sequence+1 in val_blacklist:
            continue

        file_path = os.path.join(data_path, filename)
        try:
            image = misc.imread(file_path)
        except:
            print('WARNING! UNABLE TO OPEN IMAGE:%s' %filename)
            write_log(log_path, file_path)
            continue
        label = string.atoi(val_label[sequence])
        image_list.append({'image':image, 'file_path':file_path,'label':label})

    return image_list


def read_images_test(data_path):
    file_list = os.listdir(data_path)

    image_list = []

    for filename in file_list:
        file_path = os.path.join(data_path, filename)
        try:
            image = misc.imread(file_path)
        except:
            print('WARNING! UNABLE TO OPEN IMAGE:%s' %filename)
            write_log(log_path, file_path)
            continue
        label = None # images in test set have no label
        image_list.append({'image':image, 'file_path':file_path, 'label':label})
    return image_list



def create_dataset():
    def save_to_records(save_path, image_list):
        writer = tf.python_io.TFRecordWriter(save_path)
        for i in xrange(len(image_list)):
            image_raw = image_list[i]['image'].tostring()
            height = image_list[i]['image'][0]
            width = image_list[i]['image'][1]
            depth = image_list[i]['image'][2]
            label = image_list[i]['label']
            path = image_list[i]['file_path']
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'depth': _int64_feature(depth),
                'label': _int64_feature(label),
                'path': _bytes_feature(path)
                'image_raw': _bytes_feature(image_raw)
                }))
            writer.write(example.SerializeToString())

    TRAIN_PATH = '/home/dailuo/Datasets/ILSVRC2013/ILSVRC2013/imagenet_train'
    TEST_PATH = '/home/dailuo/Datasets/ILSVRC2013/ILSVRC2013/imagenet_test/ILSVRC2012_img_test'
    VALIDATION_PATH = '/home/dailuo/Datasets/ILSVRC2013/ILSVRC2013/imagenet_val/'

    meta_path = 'meta.txt'
    val_label_path = 'ILSVRC2014_clsloc_validation_ground_truth.txt'
    val_blacklist_path = 'ILSVRC2014_clsloc_validation_blacklist.txt'

    train_images = read_images_train(TRAIN_PATH, meta_path)
    save_to_records('/home/dailuo/data/ILSVRC2013/train_sample.tf',train_images)

    val_images = read_images_validation(VALIDATION_PATH, val_label_path, val_blacklist_path)
    save_to_records('/home/dailuo/data/ILSVRC2013/val_sample.tf', val_images)

    test_images = read_images_test(TEST_PATH)
    save_to_records('/home/dailuo/data/ILSVRC2013/test_sample.tf',test_images)


if __name__ == '__main__':
    create_dataset()
