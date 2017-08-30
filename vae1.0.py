# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:23:46 2017

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

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, concatenate
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD, Adadelta, Adagrad,Adam, rmsprop
from keras import objectives
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping

batch_size =100
latent_dim = 30
nb_epoch = 50  
epsilon_std =5.0
intermediate_dim_1 = 600
#intermediate_dim_2 = 300
original_dim = 64*64

input_img = Input(shape=(64,64,1))

conv_1 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(input_img)
conv_1 = PReLU()(conv_1)
#maxpool_1 = Conv2D(80, (3, 3), strides=(2, 2), activation='sigmoid', padding='same',kernel_initializer='normal')(conv_1)
maxpool_1 = MaxPooling2D((2, 2),  padding='same')(conv_1)

conv_2 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(maxpool_1)
conv_2 = PReLU()(conv_2)
#maxpool_2 = Conv2D(80, (3, 3), strides=(2, 2), activation='sigmoid', padding='same',kernel_initializer='normal')(conv_2)
maxpool_2 = MaxPooling2D((2, 2),  padding='same')(conv_2)

conv_3 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(maxpool_2)
conv_3 = PReLU()(conv_3)
#maxpool_3 = Conv2D(80, (3, 3), strides=(2, 2), activation='sigmoid', padding='same',kernel_initializer='normal')(conv_3)
maxpool_3 = MaxPooling2D((2, 2),  padding='same')(conv_3)

conv_4 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(maxpool_3)
conv_4= PReLU()(conv_4)
#maxpool_4 = MaxPooling2D((2, 2),  padding='same')(conv_4)

#conv_5 = Conv2D(80, (3, 3), activation='tanh', padding='same',kernel_initializer='normal')(maxpool_4)
#maxpool_5 = MaxPooling2D((2, 2),  padding='same')(conv_5)

#x = Conv2D(5, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(x)
#x = MaxPooling2D((2, 2),  padding='same')(x)

visual = Flatten()(conv_3)
h = Dense(intermediate_dim_1)(visual)
h = PReLU()(h)
#h_2 = Dense(intermediate_dim_2, activation='tanh')(h_1)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):   
    z_mean, z_log_var = args  
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2)* epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#h_3 = Dense(intermediate_dim_2,activation='tanh')(z)
h_1 = Dense(intermediate_dim_1,activation='tanh')(z)
h_2 = Dense(128*8*8)(h_1)
h_3= PReLU()(h_2)
h_4 = Reshape((8,8,128))(h_3)

#conv_6 = Conv2D(80, (3, 3), activation='tanh', padding='same',kernel_initializer='normal')(h_6)
#upsample_6 = UpSampling2D((2, 2))(conv_6)
h_5 = concatenate([h_4,conv_4])
conv_5 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(h_5)
conv_5 = PReLU()(conv_5)
#upsample_7 = UpSampling2D((2, 2))(conv_7)

conv_6 = concatenate([conv_5,conv_4])
conv_7 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(conv_6)
conv_7 = PReLU()(conv_7)
upsamp_7 = UpSampling2D((2, 2))(conv_7)

upsamp_8 = concatenate([upsamp_7,conv_3])
conv_9 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(upsamp_8)
conv_9 = PReLU()(conv_9)
upsamp_9 = UpSampling2D((2, 2))(conv_9)

upsamp_9 = concatenate([upsamp_9,conv_2])
conv_10 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(upsamp_9)
conv_10 = PReLU()(conv_10)
upsamp_10 = UpSampling2D((2, 2))(conv_10)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsamp_10)

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')


def vae_loss(x, decoded):  
    xent_loss = K.sum(K.sum(objectives.binary_crossentropy(x ,decoded),axis=-1),axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
    return xent_loss  


vae = Model(inputs=input_img, outputs=decoded)
vae.compile(optimizer='rmsprop', loss=vae_loss)