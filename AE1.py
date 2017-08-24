# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 07:04:02 2017

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

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation, Reshape
from keras.optimizers import SGD, Adadelta, Adagrad,Adam, rmsprop
from keras import objectives
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU

inter_unit = 2048
hid_unit = 50

# encoder:
input_img = Input(shape=(64,64,1))

conv1_1 = Conv2D(16, (5, 5), activation='relu', padding='valid',kernel_initializer='normal')(input_img)
conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='valid',kernel_initializer='normal')(conv1_1)
#maxpool_1 = MaxPooling2D((2, 2))(conv1_2)
maxpool1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same',kernel_initializer='normal')(conv1_2)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool1)
#maxpool_2 = MaxPooling2D((2, 2))(conv2_3)
maxpool2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same',kernel_initializer='normal')(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool2)
maxpool3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same',kernel_initializer='normal')(conv3)

fc_1 = Flatten()(maxpool3)
fc_2 = Dense(inter_unit, activation='relu')(fc_1)
fc_3 = Dense(hid_unit, activation='relu')(fc_2)

# decoder:
fc_4 = Dense(inter_unit, activation='relu')(fc_3)
fc_5 = Reshape((8, 8, 32))(fc_4)

upsamp1 = UpSampling2D((2, 2))(fc_5)
conv4 = Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsamp1)
                 
upsamp2 = UpSampling2D((2, 2))(conv4)
conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsamp2)
            
upsamp3 = UpSampling2D((2, 2))(conv5)
conv6 = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsamp3)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv6)
                 
ae = Model(inputs=input_img, outputs=decoded) 
ae.compile(optimizer='rmsprop', loss='binary_crossentropy')

#EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    
ae.fit(train_Img[:5000], train_Img[:5000], 
        shuffle=True,
        verbose=1,
        batch_size = 100,
        epochs = 50,
        #validation_data = (train_Img[5000:5100], train_Img[5000:5100]),
        #callbacks=[EarlyStopping]
      )