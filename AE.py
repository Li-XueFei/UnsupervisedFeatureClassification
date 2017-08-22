# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:48:19 2017

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
input_img = Input(shape=(64,64,1))

conv1_1 = Conv2D(32, (5, 5), activation='relu', padding='same',kernel_initializer='normal')(input_img)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv1_1)
maxpool_1 = MaxPooling2D((2, 2))(conv1_2)

conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_1)
conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv2_1)
conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv2_2)
maxpool_2 = MaxPooling2D((2, 2))(conv2_3)

conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_2)
conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv3_1)
conv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv3_2)
maxpool_3 = MaxPooling2D((2, 2))(conv3_3)

fc_1 = Flatten()(maxpool_3)
fc_2 = Dense(2048, activation='relu')(fc_1)
fc_3 = Dense(8*8*128, activation='relu')(fc_2)
fc_4 = Reshape((8, 8, 128))(fc_3)

upsamp_4 = UpSampling2D((2, 2))(fc_4)
conv4_1 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsamp_4)
conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv4_1)
conv4_3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv4_2)

upsamp_5 = UpSampling2D((2, 2))(conv4_3)
conv5_1 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsamp_5)
conv5_2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv5_1)
conv5_3 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv5_2)

upsamp_6 = UpSampling2D((2, 2))(conv5_3)
conv6_1 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsamp_6)
conv6_2 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv6_1)
conv6_3 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv6_2)

decode = Conv2D(1, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(conv6_3)

#EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

cnn = Model(inputs=input_img, outputs=decode)
cnn.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(train_Img[:10000], train_Img[:10000],
        shuffle=True,
        epochs=25,
        batch_size=100,
        #validation_data=(train_Img[10000:10100],range(10000,10100),callbacks=[EarlyStopping]
       )