# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:26:42 2017

@author: lxf96
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import sqrt

from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))
print(x_train.shape)
print(x_test.shape)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, concatenate
from keras.optimizers import SGD, Adadelta, Adagrad,Adam, rmsprop
from keras import objectives
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping


input_img = Input(shape=(28,28,1))

#encoder
conv1 = Conv2D(16, (3, 3), activation='relu',padding='same')(input_img)
maxp1 = MaxPooling2D((2,2), padding='same')(conv1)
conv2 = Conv2D(8, (3, 3), activation='relu',padding='same')(maxp1)
maxp2 = MaxPooling2D((2,2), padding='same')(conv2)
conv3 = Conv2D(8, (3, 3), activation='relu',padding='same')(maxp2)
encoded = MaxPooling2D((2,2), padding='same')(conv3)

#decoder
conv4 = Conv2D(8, (3, 3), activation='relu',padding='same')(encoded)
upsp4 = UpSampling2D((2,2))(conv4)
conv5 = Conv2D(8, (3, 3), activation='relu',padding='same')(upsp4)
upsp5 = UpSampling2D((2,2))(conv5)
conv6 = Conv2D(8, (3, 3), activation='relu',padding='valid')(upsp5)
upsp6 = UpSampling2D((2,2))(conv6)

decoded = Conv2D(1, (3, 3), activation='sigmoid',padding='same')(upsp6)

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

vae = Model(inputs=input_img, outputs=decoded)
vae.compile(optimizer='adadelta', loss='binary_crossentropy')

vae.fit(x_train,x_train,
        shuffle=True,
        epochs=100,
        batch_size=100,
        validation_data=(x_test,x_test),
        callbacks=[EarlyStopping]
        )