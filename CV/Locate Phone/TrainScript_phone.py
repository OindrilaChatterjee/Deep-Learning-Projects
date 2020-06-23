#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 00:41:51 2020

@author: oindrilac
"""

import os
import sys
import glob

import numpy as np
import random
import matplotlib.pyplot as plt

import cv2
from scipy import ndimage

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
K.set_image_data_format("channels_last")


# Get the prediction function to find accuracy
from find_phone_copied import predict_phone_position

crop_size = 44
half_crop_size = 22

    
    
def main():
    path = os.path.abspath(os.getcwd())
    path = os.path.join(path, 'find_phone')
    print(path)
    train_data, test_data, train_labels, test_labels = preprocess(path, os.path.join(path, 'labels.txt'))
    model = create_model(train_data, test_data, train_labels, test_labels)
    print("Model trained and saved.")
    

def preprocess(image_dir, label_dir):
    
    ## Read labels into a dictionary
    f = open(label_dir)
    iter_f = iter(f)
    label_dir = {}
    for line in iter_f:
        line_arr = line.strip('\n').split(" ")
        label_dir[line_arr[0]] = np.array([round(float(line_arr[1]),4), round(float(line_arr[2]),4)])
        
     ## Process images
    phone_images = []
    background_images = []
    for fname in os.listdir(image_dir):
        if fname != 'labels.txt':
            image = cv2.imread(image_dir + '/' + fname)
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            phone_images_fname, background_images_fname = preprocess_image(image_gray, label_dir[fname])
            phone_images.extend(phone_images_fname)
            background_images.extend(background_images_fname)
    phone_images = np.array(phone_images)
    background_images = np.array(background_images)
    data = np.vstack((phone_images, background_images))
    labels = np.hstack((np.ones(len(phone_images)), np.zeros(len(background_images))))
    data, labels = shuffle(data, labels, random_state = 42)
   
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42) 
   
    # Reshape data to match input format of CNN
    train_data = train_data.reshape(train_data.shape[0], 1, crop_size, crop_size).astype('float32') / 255.0
    test_data = test_data.reshape(test_data.shape[0], 1, crop_size, crop_size).astype('float32') / 255.0

        
    return train_data, test_data, train_labels, test_labels
    
    
    
def preprocess_image(img, pos, num_sample_phones=50, num_sample_background=50):
    
    height, width = img.shape
    phone_images = []
    background_images = []
    pos_pixel = np.array((int(pos[0] * width), int(pos[1] * height)))
    # left boundary of box
    box_lb = pos_pixel[0] - half_crop_size
    # right boundary of box
    box_rb = pos_pixel[0] + half_crop_size
    # upper boundary of box
    box_ub = pos_pixel[1] - half_crop_size
    # bottom boundary of box
    box_bb = pos_pixel[1] + half_crop_size
    # crop the phone from the image
    phone_crop = img[box_ub:box_bb, box_lb:box_rb]
    # randomly rotate 90 degree of cropped phone
    for i in range(num_sample_phones):
        random.seed(i)
        pi = random.random()
        if pi > 0.75:
            t = random.choice([1, 2, 3, 4])
            phone_images.append(np.rot90(phone_crop, t))
        else:
            phone_images.append(phone_crop)

    # randomly crop background images
    for i in range(num_sample_background):
        # coordinate of the left up corner of cropped background
        random.seed(i)
        start_x = box_lb - 60 if (box_lb > 60) else 0
        start_y = box_ub - 60 if (box_ub > 60) else 0
        b_x = random.randint(start_x, width - crop_size)
        b_y = random.randint(start_y, height - crop_size)
        # in case there would be overlap between the background crop and phone crop
        while b_x in range(start_x, box_rb) and b_y in range(start_y, box_bb):
            b_x = random.randint(0, width - crop_size)
            b_y = random.randint(0, height - crop_size)
        back_crop = img[b_y: b_y + crop_size, b_x: b_x + crop_size]
        background_images.append(back_crop)

    return phone_images, background_images


def create_model(X_train, X_test, y_train, y_test):
    # to get reproducible results
    np.random.seed(0)
    tf.random.set_seed(0)

    # create model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(1, 44, 44), activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    sgd = optimizers.SGD(lr=0.1, decay=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # Earlystopping
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
    callbacks_list = [earlystop]
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks_list, epochs=50,
                        batch_size=128)
    # save model in HDF5 format
    model.save("model.h5")
    return model

def accuracy(image_dir, label_dir):
    f = open(label_dir)
    iter_f = iter(f)
    list_f = []
    for line in iter_f:
        line = line.strip('\n')
        list_f.append(line.split(" "))
    # convert list to dict
    dict_f = {x[0]: np.array([round(float(x[1]), 4), round(float(x[2]), 4)]) for x in list_f}

    model = load_model('model.h5')
    accuracy = 0
    total = 0
    for filename in os.listdir(image_dir):
        total = total + 1
        image = image_dir + '/' + filename
        pos = predict_phone_position(image, model)
        res = np.sqrt(np.sum(np.power(pos - dict_f[filename], 2)))
        if res <= 0.05:
            accuracy = accuracy + 1
        else:
            print(filename, " ", pos, " ", dict_f[filename])
    accuracy = accuracy / total
    print(accuracy)
    return accuracy



if __name__ == "__main__":
    main()