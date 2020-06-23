#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:39:12 2020

@author: oindrilac
"""

import os
import sys
import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt


crop_size = 44

def main():
    path = os.path.abspath(os.getcwd())
    image_dir = os.path.join(path, 'find_phone', '51.jpg')
    model = load_model('model.h5')
    pos = predict_phone_position(image_dir, model)
    print(pos[0], " ", pos[1])
    
def predict_phone_position(image_dir, model):
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    windows = sliding_windows(image) # create sliding windows of the same size as the training image patches

    windows_positive = positive_windows(image, windows, model) # detect windows that give a positive prediction

    if len(windows_positive) == 0: # if no phone is present
        return [0, 0]
    
    detection_map = np.zeros_like(image[:, :]).astype(np.float)

    for w in windows_positive:
        detection_map[w[0][1]: w[1][1], w[0][0]:w[1][0]] += 1
    detection_map = detection_map.astype('uint8')

    im2, contours, hierarchy = cv2.findContours(detection_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # extract contours in detection_map
    # find the biggiest area
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area > 10000: # remove the false positives (black stripes)
        desc_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cont in desc_contours:
            if cv2.contourArea(cont) < 10000:
                c = cont
                break
    x, y, w, h = cv2.boundingRect(c)

    bbox = [(x, y), (x + w, y + h)]
    
    title_p1 = (bbox[0][0],bbox[0][1])
    title_p2 = (bbox[0][0]+70,bbox[0][1]-30)
    cv2.rectangle(image,bbox[0],bbox[1],(0,0,255),2)
    cv2.rectangle(image,title_p1, title_p2,(255,51,157),-1)
    cv2.putText(image, "phone", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    #show the image
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    return [round((float(bbox[0][0] + bbox[1][0]) / 2) / image.shape[1], 4),
            round((float(bbox[0][1] + bbox[1][1]) / 2) / image.shape[0], 4)]


def sliding_windows(image):
    height, width = image.shape

    step_x = 4
    step_y = 4

    # number of windows in x/y
    nx_windows = int((width - crop_size) / step_x)
    ny_windows = int((height - crop_size) / step_y)

    windows = []
    for i in range(ny_windows):
        for j in range(nx_windows):
            # calculate window position
            start_x = j * step_x
            end_x = start_x + crop_size
            start_y = i * step_y
            end_y = start_y + crop_size
            # append window to the list of windows
            windows.append(((start_x, start_y), (end_x, end_y)))
    return windows


def positive_windows(img, windows, model):
    # create empty list to receive positive detection windows
    windows_positive = []
    # iterate over all windows
    for window in windows:
        t_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        t_img = t_img.reshape(1, 1, 44, 44).astype('float32')
        t_img = t_img / 255.0
        pred_probability = model.predict(t_img, batch_size=1)

        if pred_probability[0][0] > 0.7:
            windows_positive.append(window)
    return windows_positive

if __name__ == '__main__':
    main()
