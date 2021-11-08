# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 22:48:44 2018

@author: abhis
"""

import numpy as np
import pickle
from lane_detection_layer import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from scipy.misc import imresize
from PIL import Image


train_path = '../datasets/lane_detection/full_CNN_train.p'
label_path = '../datasets/lane_detection/full_CNN_labels.p'
weights_path = '../models/full_CNN_model_30.h5'
output_path = '../datasets/lane_detection/result/'

train_images = pickle.load(open(train_path,'rb'))
labels = pickle.load(open(label_path,'rb'))

#make into one array
train_images = np.array(train_images)
labels = np.array(labels)

#shuffle
train_images, labels = shuffle(train_images, labels)
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)
print (X_train.shape,y_train.shape)
#small dataset
#X_train = train_images[:50]
#y_train = y_train[:50]


input_shape = X_train.shape[1:]
#train model
#model = train_model(X_train,y_train,X_val,y_val)

#load trained model
model = load_trained_model(input_shape,weights_path)
print (model.summary())


result = lanes(X_val[0:25],model)
for i in range(25):
    im = Image.fromarray(result[i])
    im.save(output_path+str(i)+'.jpg')




