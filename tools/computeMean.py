import os
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread 
from tqdm import tqdm
 
filepath = '/home/tanghm/Documents/YFF/R2CNN_FPN_Tensorflow-master/data/VOCdevkit_train/JPEGImages/' 
pathDir = os.listdir(filepath)
 
R_channel = 0
G_channel = 0
B_channel = 0
num = 0
for idx in tqdm(range(len(pathDir))):
    filename = pathDir[idx]
    img = imread(filepath + filename)
    w,h,__ = img.shape
    num += w*h
    R_channel = R_channel + np.sum(img[:,:,0])
    G_channel = G_channel + np.sum(img[:,:,1])
    B_channel = B_channel + np.sum(img[:,:,2])


R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num
    
print("R_mean is %f, G_mean is %f, B_mean is %f" %(R_mean, G_mean, B_mean))