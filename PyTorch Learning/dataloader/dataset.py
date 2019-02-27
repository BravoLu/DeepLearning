# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-01-14 16:38:39
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-01-14 17:02:01

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io,transform
import numpy as numpy
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

def show_landmarks(image,landmarks):
	plt.imshow(image)
	plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
	plt.pause(0.001)



plt.ion()



landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n,0]
landmarks = landmarks_frame.iloc[n,1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1,2)


print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))



plt.figure()
show_landmarks(io.imread(os.path.join('faces/', img_name)), landmarks)

plt.show()
