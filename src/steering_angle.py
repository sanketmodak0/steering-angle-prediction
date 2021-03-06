import numpy as np
import pickle
import lane_detection_layer
#import steering_vgg_model
import steering_nvidia_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from scipy.misc import imresize
from PIL import Image


train_path = 'X_center_new.npy'
label_path = 'Y_center_new.npy'
test_path = 'X_test.npy'
test_label_path = 'Y_test.npy'
lane_weights_path = 'full_CNN_model_30.h5'
steering_weights_path = 'steering_nvidia_model.h5'
output_path = '../datasets/steering angle/'

print ('reading training dataset..')
train_images = np.load(train_path)
labels = np.load(label_path)
print ('train images',train_images.shape)
print ('labels',labels.shape)
#test_images = np.load(test_path)
#test_labels = np.load(test_label_path)


#shuffle
print ('shuffling images..')
train_images, labels = shuffle(train_images, labels)

#small dataset
# train_images = train_images[:1000]
# labels = labels[:1000]
#pre-processing


#lane detect
#resizing images for lane detector
def resize(train_images):
    original_size = train_images.shape[1:]
    print ('image original size',original_size)
    small_images = np.array(list(map(lambda image: imresize(image,(80,160,3)), train_images[:])))
    input_shape = small_images.shape[1:]
    print ('resize',input_shape)
    return small_images

small_images = resize(train_images)
#predict lanes
print ('loading lane detection model..')
input_shape = small_images.shape[1:]
lane_model = lane_detection_layer.load_trained_model(input_shape,lane_weights_path)
print ('predicting lanes..')
lane_images = lane_detection_layer.lanes(small_images,lane_model)
print (lane_images.shape)

#see sample preformance of lane detector on steering angle images
#sample_image = small_images[80]
#sample_image = Image.fromarray(sample_image)
#sample_image
#sample_image = lane_images[80]
#sample_image = Image.fromarray(sample_image)
#sample_image

#concatonate
X_train = np.concatenate((small_images,lane_images),axis=2)

#shuffle
X_train, X_val, y_train, y_val = train_test_split(X_train, labels, test_size=0.1)
print (X_train.shape,y_train.shape)

#train steering angle CNN
input_shape = X_train.shape[1:]
print ('starting training..')
angle_model = steering_nvidia_model.train_model(X_train,y_train,X_val,y_val,'steering_nvidia_model.h5')
#print ('loading steering angle model')
#angle_model = steering_nvidia_model.load_trained_model(input_shape,steering_weights_path)
print (angle_model.summary())

#bins
bins = np.linspace(-1,1,20)

#testing
print ('training accuracy..')
train_result = angle_model.predict(X_train[:])
train_result= train_result.reshape(train_result.shape[0])
train_score  = np.mean(np.square(train_result- y_train))
print (train_score)
train_result_digitize = np.digitize(train_result,bins)
y_train_digitize = np.digitize(y_train,bins)
train_accuracy = np.mean(train_result_digitize==y_train_digitize)
print (train_accuracy)
#print (y_val[0:1],result)


print ('validation accuracy..')
val_result = angle_model.predict(X_val[:])
val_result= val_result.reshape(val_result.shape[0])
val_score  = np.mean(np.square(val_result- y_val))
print (val_score)
val_result_digitize = np.digitize(val_result,bins)
y_val_digitize = np.digitize(y_val,bins)
val_accuracy = np.mean(val_result_digitize==y_val_digitize)
print (val_accuracy)