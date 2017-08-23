# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:43:34 2017

@author: lxf96
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import load_data

Imgs, labels = load_data.getData()
trainX = load_data.preprocess(Imgs)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation, Reshape
from keras.optimizers import SGD, Adadelta, Adagrad,Adam, rmsprop
from keras import objectives
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU

#encoder:
input_img = Input(shape=(64,64,3))

conv1_1 = Conv2D(32, (5, 5), activation='relu', padding='valid',kernel_initializer='normal')(input_img)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='valid',kernel_initializer='normal')(conv1_1)
maxpool_1 = MaxPooling2D((2, 2))(conv1_2)

conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_1)
conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv2_1)
conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv2_2)
maxpool_2 = MaxPooling2D((2, 2))(conv2_3)

conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_2)
conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv3_1)
conv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv3_2)
maxpool_3 = MaxPooling2D((2, 2))(conv3_3)

fc_1 = Flatten()(maxpool_2)
fc_2 = Dense(2048, activation='relu')(fc_1)
fc_3 = Dense(2048, activation='relu')(fc_2)
predictions = Dense(1, activation='sigmoid')(fc_3)

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

cnn = Model(inputs=input_img, outputs=predictions)
cnn.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(train_Img[:19000], labels[:19000],
        shuffle=True,
        epochs=100,
        batch_size=100,
        validation_data=(train_Img[19000:20000], labels[19000:20000]), callbacks=[EarlyStopping]
       )