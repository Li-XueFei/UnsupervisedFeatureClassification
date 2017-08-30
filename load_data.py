from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


def scale(x):
	result = x - x.mean()
	result = (x - x.min()) / (x.max() - x.min())
	return result

def norm(x):
	result = x - x.mean()
	result = result / x.min()
	return result

def getData():
	path = os.listdir('../images/')
	
	Img = []
	Class = []
	for i in range(len(path)):
		path_now = path[0]
		a = np.load('../images/'+str(path_now))
		for j in range(a.shape[0]):
			Img.append(a[j]['image'])
			Class.append(a[j]['class'])
	
	Img = np.asarray(Img)

	Class = np.asarray(Class)
	Class[Class=='STAR'] = 0
	Class[Class=='GALAXY'] = 1
	Class[Class=='QSO'] = 0
	return Img, Class

def preprocess(data):
	data[np.isnan(data)==True] = 1000
	for c in range(5):
		img = data[:, c, :, :]
		img_mean = np.mean(img)
		img[img==1000] = img_mean
		data[:, c, :, :] = norm(img)

	train_img = np.zeros((data.shape[0], 64, 64, 3))
	for i in range(data.shape[0]):
		train_img[i] = np.stack((data[i][1], data[i][2], data[i][3]), axis=-1)
		#train_img[i] = data[i][3].reshape((64, 64, 1))
	return train_img
	
Image, label = getData()
train_img = preprocess(Image)
print(train_img.shape)



