# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 07:26:13 2017

@author: lxf96
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import load_data

Imgs, labels = load_data.getData()
train_Img = load_data.preprocess(Imgs)
labels = np.zeros((50000))
for i in range(500):
    labels[i*100:(i+1)*100]=np.array(range(100))

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, concatenate, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD, Adadelta, Adagrad,Adam, rmsprop
from keras import objectives
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping

input_img2 = Input(shape=(64,64,1))
 
conv1_1 = Conv2D(32, (5, 5), activation='relu', padding='valid',kernel_initializer='normal')(input_img2)
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

fc_1 = Flatten()(maxpool_3)
fc_2 = Dense(2048, activation='relu')(fc_1)
fc_3 = Dense(2048, activation='relu')(fc_2)
predictions = Dense(1, activation='sigmoid')(fc_3)

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

cnn = Model(inputs=input_img2, outputs=predictions)
cnn.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(trainX[:45000], labels[:45000],
        shuffle=True,
        epochs=100,
        batch_size=100,
        validation_data=(trainX[45000:50000], labels[45000:]), callbacks=[EarlyStopping]
        )
