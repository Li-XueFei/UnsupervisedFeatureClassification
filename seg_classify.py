# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:42:04 2017

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

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, concatenate
from keras.optimizers import SGD, Adadelta, Adagrad,Adam, rmsprop
from keras import objectives

Imgs, labels = load_data.getData()

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, concatenate
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

input_img = Input(shape=(64,64,1))

conv_1 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(input_img)
maxpool_1 = Conv2D(80, (3, 3), strides=(2, 2), activation='tanh', padding='same',kernel_initializer='normal')(conv_1)

conv_2 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_1)
maxpool_2 = Conv2D(80, (3, 3), strides=(2, 2), activation='tanh', padding='same',kernel_initializer='normal')(conv_2)

conv_3 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_2)
maxpool_3 = Conv2D(80, (3, 3), strides=(2, 2), activation='tanh', padding='same',kernel_initializer='normal')(conv_3)

encoded = Conv2D(1, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_3)

conv_5 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(concat_5)
upsample_5 = UpSampling2D((2, 2))(conv_5)

conv_6 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(encoded)
upsample_6 = UpSampling2D((2, 2))(conv_6)

concat_7 = concatenate([upsample_6,conv_3])
conv_7 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(concat_7)
upsample_7 = UpSampling2D((2, 2))(conv_7)

concat_8 = concatenate([upsample_7,conv_2])
conv_8 = Conv2D(80,  (3, 3), activation='relu',padding='same',kernel_initializer='normal')(concat_8)
upsample_8 = UpSampling2D((2, 2))(conv_8)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsample_8)

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto') 

ae1 = Model(inputs=input_img, outputs=decoded)
ae1.compile(optimizer='rmsprop', loss='binary_crossentropy')

ae1.fit((train_Img[:10000], train_Img[:10000]),
        shuffle=True,
        epochs=50,
        batch_size=batch_size,
        )

inp = vae1.input    # input placeholder
outputs = [layer.output for layer in vae1.layers]       # all layer outputs
functor = K.function([inp] + [K.learning_phase()], outputs ) # evaluation function
decode1 = np.zeros((10000, 64, 64, 1))

for i in range(100):
    layer_outs = functor([train_Img[i*100:(i+1)*100], 1.])
    decode1[i*100:(i+1)*100]=layer_outs[15]
    
h1 = train_Img[:100000].reshape((10000, 64*64))
h2 = decode1.reshape((10000, 64*64))
hpcol = np.dstack(h1, h2))

from skimage.transform.pyramids import pyramid_expand
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import whiten

kmeans = MiniBatchKMeans(n_clusters=2, compute_labels=False)

for img_idx in range(10000):
    sample = whiten(hpcol[img_idx,:,:])
    kmeans.fit(sample)
    pred = kmeans.predict(sample)
    pred = pred.reshape((64,64))
    c = pred[31,31]
    decode[img_idx][pred==1-c]=0
    
h1 = train_Img[:10000].reshape((10000,64*64))
h2 = decode1[:10000].reshape((10000,64*64))
#h3 = decode2[5000:5100, :,:,:].reshape((100, 64*64))
hypercol = np.dstack((h1, h2))

new_Img = np.zeros((10000, 64, 64, 1))
clus = 10
aggregate_hpcol = whiten(allimg)
kmeans.fit(aggregate_hpcol)
for img_idx in range(10000):
    sample = hypercol[img_idx].reshape((4096,2))
        
    aggregate_hpcol = whiten(sample)
    kmeans.fit(sample)
        
    pred = kmeans.predict(sample)
    pred = pred.reshape((64,64))
    c = pred[31, 31]
    
    for row in range(64):
        col=0
        while col<32:
            edge1 = [0]
            edge2 = [-10]
            if pred[(31+row)%64, 31+col]==c and pred[(31+row)%64, 32+col]!=c:
                if col-edge2[-1]>clus:
                    edge1.append(col)
                    pred[(31+row)%64, edge1[-1]+31:31+col]=c
            if pred[(31+row)%64, 31+col]!=c and pred[(31+row)%64, 32+col]==c:
                if col-edge1[-1]>clus:
                    edge2.append(col)
                    pred[(31+row)%64, edge1[-1]+31:31+col]=1-c
            col += 1
        if len(edge2)==1:
            pred[:(30+row)%64, :]=1-c
            break
                
        while col<31:
            edge1 = [0]
            edge2 = [10]
            if pred[(31+row)%64, 31-col]==c and pred[(31+row)%64, 32-col]!=c:
                if edge2[-1]-col>clus:
                    edge1.append(col)
                    pred[(31+row)%64, 31+col:31+edge1[-1]]=c
            if pred[31+row, 31+col]!=c and pred[31+row, 32+col]==c:
                if edge1[-1]-col>clus:
                    edge2.append(col)
                    pred[(31+row)%64, 31+col:31+edge1[-1]]=1-c
            col -=1
        if len(edge2)==1:
            pred[(32+row)%64:, :]=1-c
            break
        
        new_Img[img_idx] = pred

ae2 = Model(inputs=input_img, outputs=decoded)
ae2.compile(optimizer='rmsprop', loss='binary_crossentropy')

ae2.fit((new_img[:9000], new_Img[:9000]),
        shuffle=True,
        epochs=50,
        batch_size=batch_size,
        validation_data=(new_Img[9000:],new_Img[9000:]),callbacks=[EarlyStopping])

inp = ae2.input    # input placeholder
outputs = [layer.output for layer in ae2.layers]       # all layer outputs
functor = K.function([inp] + [K.learning_phase()], outputs ) # evaluation function

layer_outs = functor([train_Img[i*100:(i+1)*100], 1.])
attr1=layer_outs[7].reshape((64))


from keras import regularizers

intermediate_dim = 64
latent_dim = 2

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

input_img = Input(shape=(64,))
# add a Dense layer with a L1 activity regularizer
h = Dense(intermediate_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(64, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(x, x_decoded_mean)

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(attr1[:9000], attr1[:9000],
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(attr1[9000:], attr1[9000:]))

np.save('attr.npy', attr[:100])
