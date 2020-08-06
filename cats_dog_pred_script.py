#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 23:31:16 2020

@author: teddy
"""

import numpy as np
import pandas as pd
from keras import optimizers 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import cv2


# read train data into dataframe
def read_image(path):
    filenames = os.listdir(path)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    
    return df
        

# split into train and valid
def train_test_split(df,
                     test_size = 0.2):
    
    df_train, df_valid = train_test_split(df, test_size=test_size, random_state=42)
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    return df_train, df_valid


    

# datagen for producing image with 15 different angles

def datagen(rotation_range=15,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1):
    
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        rescale=rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range
    )

    
    return datagen



# generate data from datagen

def data_generator(df, 
                   path,
                   datagen,
                   target_size=(224,224),
                   class_mode='categorical',
                   batch_size=100,
                   x_col='filename',
                   y_col='category',
                   shuffle=True):  # shuffle train data, no shuffle with test and valid data
    
    data_generator = datagen.flow_from_dataframe(
    df, 
    path, 
    x_col=x_col,
    y_col=y_col,
    target_size=target_size,
    class_mode=class_mode,
    batch_size=batch_size,
    shuffle=shuffle)
    
    return data_generator




# train model
def train(model, 
          train_generator,
          validation_generator,
          valid_size,
          train_size,
          batch_size=100,
          epoch=10):
    
    model.fit_generator(
    train_generator, 
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=valid_size//batch_size,
    steps_per_epoch=train_size//batch_size)
    
    return model




# read test data
def test_data(test_path):
    test_filenames = os.listdir(test_path)
    df_test = pd.DataFrame({
    'filename': test_filenames})

    return df_test



# predict (0:cat, 1:dog)
    
def predict(model, test_generator, test_size, batch_szie=100):
    
    predict = model.predict_generator(test_generator, steps=np.ceil(test_size/batch_size))
    
    return predict



# main
def main(path, test_path):

    df = read_image(path)
    
    df_train = train_test_split(df)[0]
    df_valid = train_test_split(df)[1]
    

    datagen = datagen()# default
    
    train_generator = data_generator(df_train, path, datagen)
    validation_generator = data_generator(df_valid, path, ImageDataGenerator(rescale=1./255))
    
    test_generator = data_generator(df_test, 
                                    test_path, 
                                    ImageDataGenerator(rescale=1./255), 
                                    y_col=None, 
                                    class_mode=None,
                                    shuffle=False)
    
    df_test = test_data(test_path)
    


    model=VGG16(include_top=True, weights='imagenet')
    
    
    model = train(model, train_generator, validation_generator, 
                  df_valid.shape[0], df_train.shape[0])
    
    res = predict(model, test_generator, df_test.shape[0])
    
    return predict




# train data path
path = "../input/dogs-vs-cats/train/train/"


# test data path
test_path = "../input/dogs-vs-cats/test1/test1"


if __name__ == "__main__":
    main(path, test_path)
    

















