# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 13:17:32 2022

@author: Neel
"""

import os, time

import numpy as np
from tqdm import tqdm
from IPython import display
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as transforms
from math import ceil, sqrt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, ZeroPadding2D

def reverse_tanh(x):
    return np.rint(x * 127.5 +127.5).astype("int")

model = load_model('character_generator_50.h5', compile = False)

def load():
    model.load_weights("face_generator_50.h5")
    num = 16
    j = 0
    while True:
        noise = np.random.normal(0, 1, (num, 100))
        gen_img = model.predict(noise)
        fig = plt.figure()
        for i in range(0, num):
            fig.add_subplot(ceil(sqrt(num)), ceil(sqrt(num)), i+1)
            plt.imshow(reverse_tanh(gen_img[i]))
            plt.axis("off")
            
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.savefig('gen_data/'+str(j)+'.png')
        plt.show()
        j += 1
