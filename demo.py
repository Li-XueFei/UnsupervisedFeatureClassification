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

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, concatenate
from keras.optimizers import SGD, Adadelta, Adagrad,Adam, rmsprop
from keras import objectives

Imgs, labels = load_data.getData()
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Flatten, Reshape, Lambda, concatenate
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
original_dim = 64*64

input_img = Input(shape=(64,64,1))

conv_1 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(input_img)
conv_1 = BatchNormalization()(conv_1)
maxpool_1 = Conv2D(80, (3, 3), strides=(2, 2), activation='tanh', padding='same',kernel_initializer='normal')(conv_1)

conv_2 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_1)
conv_2 = BatchNormalization()(conv_2)
maxpool_2 = Conv2D(80, (3, 3), strides=(2, 2), activation='tanh', padding='same',kernel_initializer='normal')(conv_2)

conv_3 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(maxpool_2)
conv_3 = BatchNormalization()(conv_3)
maxpool_3 = Conv2D(80, (3, 3), strides=(2, 2), activation='tanh', padding='same',kernel_initializer='normal')(conv_3)

f = Flatten()(maxpool_3)
encoded = Dense(50)(f)

h_1 = Dense(80*8*8,activation='relu')(encoded)
h_1 = Dropout(0.2)(h_1)
h_2 = Reshape((8,8,80))(h_1)
h_2 = Dropout(0.1)(h_2)

conv_6 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(h_2)
upsample_6 = UpSampling2D((2, 2))(conv_6)

concat_7 = concatenate([upsample_6,conv_3])
conv_7 = Conv2D(80, (3, 3), activation='relu', padding='same',kernel_initializer='normal')(concat_7)
upsample_7 = UpSampling2D((2, 2))(conv_7)

concat_8 = concatenate([upsample_7,conv_2])
conv_8 = Conv2D(80,  (3, 3), activation='relu',padding='same',kernel_initializer='normal')(concat_8)
upsample_8 = UpSampling2D((2, 2))(conv_8)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsample_8)

def ae_loss(x, decoded):
    xent_loss = original_dim * objectives.mean_squared_error(x,decoded)
    return xent_loss

EarlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')

ae1 = Model(inputs=input_img, outputs=decoded)
ae1.compile(optimizer='adam', loss=ae_loss)
ae1.fit(train_Img[:45000], train_Img[:45000],
        shuffle=True,
        epochs=5,
        batch_size=batch_size,
        validation_data=(train_Img[45000:50000],train_Img[45000:50000]),
        callbacks=[EarlyStopping])

inp = ae1.input    # input placeholder
outputs = [layer.output for layer in ae1.layers]       # all layer outputs
functor = K.function([inp] + [K.learning_phase()], outputs ) # evaluation function
decode1 = np.zeros((50000, 64, 64, 1))
for i in range(500):
    layer_outs = functor([train_Img[i*100:(i+1)*100], 1.])
    decode1[i*100:(i+1)*100]=layer_outs[24]
    
    h2 = decode1.reshape((50000, 64*64))
    
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
    def getMask(self, mark):
        mask = np.zeros_like(self.value)
        for p in mark:
            mask[p[0], p[1]] = 1
            return mask
    
    def getCluster(self):
        mark = []
        pend = []
        x = int(self.row/2-1)
        y = int(self.col/2-1)
        p = point(x, y)
        for p1 in pend:
            x = int(getattr(p1,'x'))
            y = int(getattr(p1,'y'))
            #print(x,y)
            if x>0 and pointMap.visit(self, x-1, y):
                p2 = point(x-1, y)
                if pointMap.getValue(self, p2) == 1:
                    mark.append((x-1,y))
                    pend.append(p2)
            if x<self.row-1 and pointMap.visit(self, x+1, y):
                p2 = point(x+1, y)
                if pointMap.getValue(self, p2) == 1:
                    mark.append((x+1,y))
                    pend.append(p2)
            if y>0 and pointMap.visit(self, x, y-1):
                p2 = point(x, y-1)
                if pointMap.getValue(self, p2) == 1:
                    mark.append((x,y-1))
                    pend.append(p2)
            if y<self.col-1 and pointMap.visit(self, x, y+1):
                p2 = point(x, y+1)
                if pointMap.getValue(self, p2) == 1:
                    mark.append((x,y+1))
                    pend.append(p2)
        return mark

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import whiten

kmeans = MiniBatchKMeans(n_clusters=2, compute_labels=False)
new_Img = []
add = 0
for img_idx in range(50000):
    if img_idx == 49999:
        break
    sample = h2[img_idx+add].reshape((4096, 1))
    sample = whiten(sample)
    kmeans.fit(sample)
    pred = kmeans.predict(sample)
    pred = pred.reshape((64,64))
    if pred.sum()>64*32:
        if pred[32, 32]==0:
            pred = 1-pred
        else:
            add+=1
    
    pm = pointMap(pred)
    mark = pm.getCluster()
    #mask = pm.getMask(mark)
    
    new_img = np.random.normal(train_Img[img_idx],0.05, size=(64, 64, 1)) 
    #new_Img[img_idx][mask==1] = train_Img[img_idx][mask==1]
    for p in mark:
		new_img[p[0], p[1], :] = train_Img[img_idx, p[0], p[1], :]

	new_Img.append(new_img)



new_Img = np.array(new_Img)

ae2 = Model(inputs=input_img, outputs=decoded)
ae2.compile(optimizer='adam', loss=ae_loss)
ae2.fit(new_Img[:45000], new_Img[:45000],
		shufle=True,
		epochs=22,
		batch_size=batch_size,
		validation_data=(new_Img[45000:],new_Img[45000:]),callbacks=[EarlyStopping])
    
inp = ae2.input    # input placeholder
outputs = [layer.output for layer in ae2.layers]       # all layer outputs
functor = K.function([inp] + [K.learning_phase()], outputs ) # evaluation function
attr = np.zeros((50000, 30))
for i in range(500):
    layer_outs = functor([train_Img[i*100:(i+1)*100], 1.])
    attr1[i*100:(i+1)*100]=layer_outs[10]
    
from keras import regularizers

intermediate_dim = 64
latent_dim = 2

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

input_img = Input(shape=(30,))
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
vae.fit(attr[:40000], attr[:40000],
        shuffle=True,
        nb_epoch=nb_epoch,
        validation_data=(attr[40000:50000], attr[40000:50000]))

model = vae
imp = model.input
outputs = [layer.output for layer in model.layers]
functor = K.function([inp] + [K.learning_phase()], outputs)
attr2 = np.zeros((50000, 2))
for i in ranger(500):
    layerouts = functor([attr[i*100:(i+1)*100], 1.])
    attr2[i*100:(i+1)*100] = layer_outs[2]
    
np.save('attr2.npy', attr2)
np.save('class2.npy', labels)
