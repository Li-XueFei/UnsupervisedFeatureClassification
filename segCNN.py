# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 00:47:51 2017

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

conv_1 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(input_img1)
conv_1 = BatchNormalization()(conv_1)
maxpool_1 = Conv2D(80, (3, 3), strides=(2, 2), padding='same',kernel_initializer='normal')(conv_1)
#maxpool_1 = MaxPooling2D((2, 2),  padding='same')(conv_1)

conv_2 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_1)
conv_2 = BatchNormalization()(conv_2)
maxpool_2 = Conv2D(80, (3, 3), strides=(2, 2), padding='same',kernel_initializer='normal')(conv_2)
#maxpool_2 = MaxPooling2D((2, 2),  padding='same')(conv_2)

conv_3 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_2)
conv_3 = BatchNormalization()(conv_3)
maxpool_3 = Conv2D(80, (3, 3), strides=(2, 2), padding='same',kernel_initializer='normal')(conv_3)
#maxpool_3 = MaxPooling2D((2, 2),  padding='same')(conv_3)

f = Flatten()(maxpool_3)
encoded = Dense(50)(f)
#maxpool_4 = MaxPooling2D((2, 2),  padding='same')(conv_4)

h_1 = Dense(80*8*8,activation='relu')(encoded)
h_2 = Reshape((8,8,80))(h_1)

upsample_6 = UpSampling2D((2, 2))(h_2)
conv_6 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsample_6)

#concat_7 = concatenate([upsample_6,conv_3])
upsample_7 = UpSampling2D((2, 2))(conv_6)
conv_7 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(upsample_7)

#concat_8 = concatenate([upsample_7,conv_2])
upsample_8 = UpSampling2D((2, 2))(conv_7)
conv_8 = Conv2D(80,  (3, 3), activation='relu',padding='same',kernel_initializer='normal')(upsample_8)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv_8)

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')

def ae_loss(x, decoded):  
    xent_loss = original_dim * objectives.mean_squared_error(x,decoded)
    return xent_loss


def vae_loss(x, decoded):  
    xent_loss = K.sum(K.sum(objectives.binary_crossentropy(x ,decoded),axis=-1),axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
    return xent_loss + 10*kl_loss  

ae1 = Model(inputs=input_img1, outputs=decoded)
ae1.compile(optimizer='rmsprop', loss=ae_loss)

ae1.fit(trainX[:50000], trainX[:50000],
        shuffle=True,
        epochs=50,
        batch_size=batch_size,
        validation_data=(trainX[50000:],trainX[50000:]),callbacks=[EarlyStopping])

class point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


class pointMap:
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
                    mark.append(p2)
                    pend.append(p2)
            if x<self.row-1 and pointMap.visit(self, x+1, y):
                p2 = point(x+1, y)
                if pointMap.getValue(self, p2) == 1:
                    mark.append(p2)
                    pend.append(p2)
            if y>0 and pointMap.visit(self, x, y-1):
                p2 = point(x, y-1)
                if pointMap.getValue(self, p2) == 1:
                    mark.append(p2)
                    pend.append(p2)
            if y<self.col-1 and pointMap.visit(self, x, y+1):
                p2 = point(x, y+1)
                if pointMap.getValue(self, p2) == 1:
                    mark.append(p2)
                    pend.append(p2)
        return mark
    
    def getMusk(self):
        musk = self.value
        cluster = pointMap.getCluster()
        for p in cluster:
            musk[int(getattr(p,'x')), int(getattr(p,'y'))] = 0
        return musk
    
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import whiten

kmeans = MiniBatchKMeans(n_clusters=2, compute_labels=False)

h2 = decode1[:10000].reshape((10000,64*64))

new_Img = np.zeros((10000, 64, 64, 1))

add = 0
for img_idx in range(10000):
    if img_idx+add==9999:
        break
    sample = h2[img_idx+add].reshape((4096,1))
        
    aggregate_hpcol = whiten(sample)
    kmeans.fit(aggregate_hpcol)
        
    pred = kmeans.predict(aggregate_hpcol)
    pred = pred.reshape((64,64))
        
    if pred.sum()>64*32:
        if pred[32, 32] == 0:
            pred = 1-pred
        else:
            add += 1
            continue

    pm = pointMap(pred)
    musk = pm.getMusk()
        
    bg = np.mean(train_Img[img_idx+add, pic0==1])
    new_Img[img_idx, musk==1] = bg
    
    


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

cnn.fit(trainX[:19000], labels[:19000],
        shuffle=True,
        epochs=100,
        batch_size=100,
        validation_data=(trainX[19000:20000], labels[19000:20000]), callbacks=[EarlyStopping]
        )
