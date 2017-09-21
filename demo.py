# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:13:48 2017

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

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, concatenate, Activation, BatchNormalization
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
intermediate_dim_1 = 1024
#intermediate_dim_2 = 300
original_dim = 64*64

input_img1 = Input(shape=(64,64,1))

conv_1 = Conv2D(80, (3, 3), padding='same',kernel_initializer='normal')(input_img1)
conv_1 =PReLU()(conv_1)
conv_1 = BatchNormalization()(conv_1)
maxpool_1 = MaxPooling2D((2, 2),padding='same')(conv_1)

conv_2 = Conv2D(40, (3, 3), padding='same',kernel_initializer='normal')(maxpool_1)
conv_2 = PReLU()(conv_2)
conv_2 = BatchNormalization()(conv_2)
maxpool_2 = MaxPooling2D((2, 2),  padding='same')(conv_2)

conv_3 = Conv2D(20, (3, 3),padding='same',kernel_initializer='normal')(maxpool_2)
conv_3 = PReLU()(conv_3)
conv_3 = BatchNormalization()(conv_3)
maxpool_3 = MaxPooling2D((2, 2),  padding='same')(conv_3)

f = Flatten()(maxpool_3)
encoded = Dense(latent_dim)(f)
#maxpool_4 = MaxPooling2D((2, 2),  padding='same')(conv_4)

h_1 = Dense(80*8*8,activation='relu')(encoded)
h_2 = Reshape((8,8,80))(h_1)

upsample_6 = UpSampling2D((2, 2))(h_2)
conv_6 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsample_6)

concat_7 = concatenate([upsample_6,conv_3])
upsample_7 = UpSampling2D((2, 2))(conv_6)
conv_7 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsample_7)

concat_8 = concatenate([upsample_7,conv_2])
upsample_8 = UpSampling2D((2, 2))(conv_7)
conv_8 = Conv2D(80,  (3, 3), activation='relu',padding='same',kernel_initializer='normal')(upsample_8)

decoded = Conv2D(1, (3, 3), padding='same')(conv_8)
decoded = PReLU()(decoded)

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')

def ae_loss(x, decoded):  
    xent_loss = original_dim * objectives.mean_squared_error(x,decoded)
    return xent_loss


def vae_loss(x, decoded):  
    xent_loss = K.sum(K.sum(objectives.binary_crossentropy(x ,decoded),axis=-1),axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
    return xent_loss + 10*kl_loss  


ae1 = Model(inputs=input_img1, outputs=decoded)
ae1.compile(optimizer='adam', loss=ae_loss)

ae1.fit(train_Img, train_Img,
        shuffle=True,
        epochs=2,
        batch_size=batch_size,
        validation_dsplit=0.2, 
        callbacks=[EarlyStopping])

def getFun(model):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp] + [K.learning_phase()], outputs ) # evaluation function
    return functor

functor = getFun(ae1)
decode = np.zeros((train_Img.shape[0], 64, 64, 1))
for i in range(train_Img.shape[0]/100):
    layer_outs = functor([train_Img[i*100:(i+1)*100], 1.])
    decode[i*100:(i+1)*100]=layer_outs[27]

import math

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class pointMap:
    attr = []
    
    def __init__(self, value):
        self.value = value
        self.used = np.ones_like(value)
        self.row = value.shape[0]
        self.col = value.shape[1]
    
    def getValue(self, p1):
        x = int(getattr(p1,'x'))
        y = int(getattr(p1,'y'))
        return self.value[x, y]
    
    def visit(self,x, y):
        if self.used[x, y]:
            self.used[x, y]=0
            return 1
        else:
            return 0
    
    def getCluster(self):
        mark = []
        pend = []
        x = int(self.row/2-1)
        y = int(self.col/2-1)
        p = point(x, y)
        self.used[x,y]=0
        mark.append((x,y))
        pend.append(p)
        for p1 in pend:
            x = int(getattr(p1,'x'))
            y = int(getattr(p1,'y'))
            #print(x,y)
            if x>0 and pointMap.visit(self, x-1, y):
                p2 = point(x-1, y)
                if pointMap.getValue(self, p2) == 1:                        
                    mark.append((x-1, y))
                    pend.append(p2)
            if x<self.row-1 and pointMap.visit(self, x+1, y):
                p2 = point(x+1, y)
                if pointMap.getValue(self, p2) == 1:
                    mark.append((x+1, y))
                    pend.append(p2)
            if y>0 and pointMap.visit(self, x, y-1):
                p2 = point(x, y-1)
                if pointMap.getValue(self, p2) == 1:
                    mark.append((x, y-1))
                    pend.append(p2)
            if y<self.col-1 and pointMap.visit(self, x, y+1):
                p2 = point(x, y+1)
                if pointMap.getValue(self, p2) == 1:
                    mark.append((x, y+1))
                    pend.append(p2)
        return mark
    
    def getMusk(self, mark):
        musk = np.zeros_like(self.value)
        for p in mark:
            musk[p[0], p[1]] = 1
        return musk
    
    
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import whiten

kmeans = MiniBatchKMeans(n_clusters=2, compute_labels=False)
#h1 = train_Img[:14000].reshape((14000,64*64))
h = deocode.reshape((10000, 64*64, 1))
#h3 = decode2[5000:5100, :,:,:].reshape((100, 64*64))
#hypercol = np.dstack((h1, h2))

new_Img = np.empty((10000, 64, 64, 1))

add = 0
skip = []
for img_idx in range(10000):
    if img_idx+add==9999:
        break
    sample = h[img_idx+add]
        
    sample = whiten(sample)
    kmeans.fit(sample)
        
    pred = kmeans.predict(sample)
    pred = pred.reshape((64,64))
        
    if pred.sum()>64*32:
        if pred[32, 32] == 0:
            pred = 1-pred
        else:
            add+=1
            skip.append(img_idx)
            continue

    pm = pointMap(pred)
    mark = pm.getCluster()
    musk = pm.getMusk(mark)
    
    new_img = np.ones((64, 64, 1)) * np.mean(train_Img1[img_idx+add])
    new_img[musk==1] = train_Img1[img_idx+add, musk==1]
    new_Img[img_idx] = np.random.normal(new_img,0.01, size=(64, 64, 1))
    
ae2 = Model(inputs=input_img1, outputs=decoded)
ae2.compile(optimizer='adam', loss=ae_loss)

ae2.fit(new_Img, new_Img,
        shuffle=True,
        epochs=2,
        batch_size=batch_size,
        validation_dsplit=0.2, 
        callbacks=[EarlyStopping])

functor = getFun(ae2)
attr1 = np.zeros((new_Img.shape[0], latent_dim))
for i in range(new_Img.shape[0]/100):
    layer_outs = functor([new_Img[i*100:(i+1)*100], 1.])
    attr1[i*100:(i+1)*100]=layer_outs[14]
    
print(attr1[:10])


net_3_input= Input(shape=(30,))
h_1 = Dense(39, activation='tanh')(net_3_input)
z_mean = Dense(2,activation='tanh')(h_1)
z_log_var = Dense(2)(h_1)

def sampling(args):   
    z_mean, z_log_var = args  
    epsilon = K.random_normal(shape=(100, 2), mean=0.,stddev=0.1)
    return z_mean + K.exp(z_log_var / 2)* epsilon

z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])
decoded = Dense(30, activation='tanh')(z)


def vae_loss(x, decoded):
    xent_loss = K.sum((objectives.mse(x ,decoded)),axis=-1)
    #kl_loss_d1 = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
    m = K.constant(1)
    s = K.constant(1)
    #kl_loss_d1 = K.sum(K.log(2/K.exp(z_log_var/2))+(K.square(z_mean)+(K.exp(z_log_var/2)-K.constant(1))*(K.exp(z_log_var/2)+K.constant(1)))/(K.constant(2)),axis = -1)
    kl_loss_d1 = K.sum(K.log(2*s/K.exp(z_log_var/2))+(K.constant(2)*m*(-K.exp(-(K.square(z_mean))/((K.constant(2))*K.exp(z_log_var)))*K.exp(z_log_var/2) + K.sqrt(K.constant(np.pi/2))*z_mean*(K.constant(1)-K.tanh(K.constant(1.19)*z_mean/K.constant(np.sqrt(2))/K.exp(z_log_var/2)))) )/(K.square(s))+(K.square(m-z_mean)+(K.exp(z_log_var/2)-s)*(K.exp(z_log_var/2)+s))/(K.constant(2)*K.square(s)),axis = -1)
    return 1*xent_loss + 0.1*kl_loss_d1 

vae = Model(input=net_3_input, output=decoded)
vae.compile(optimizer='adam', loss=vae_loss)

vae.fit(attr1, attr1,
        shuffle=True,
        epochs=100,
        batch_size=batch_size,
        validation_split=0.2,callbacks=[EarlyStopping])

functor = getFun(vae)
attr2 = np.zeros((new_Img.shape[0], 2))
for i in range(new_Img.shape[0]/100):
    layer_outs = functor([attr1[i*100:(i+1)*100], 1.])
    attr1[i*100:(i+1)*100]=layer_outs[2]
    
print(attr2[:10])