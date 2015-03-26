from pylab import *
import numpy as np
import random
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

matplotlib
gray()

os.chdir('C:/Users/Ramaneek/SkyDrive/Documents/Github/CSC320-Winter-2014/project 4/')


## MAIN
b1 = imread('images/flowers-backA.jpg')/255.0
c1 = imread('images/flowers-compA.jpg')/255.0
b2 = imread('images/flowers-backB.jpg')/255.0
c2 = imread('images/flowers-compB.jpg')/255.0

alpha = np.zeros((b1.shape[0], b1.shape[1]))
f = np.zeros(b1.shape)

for i in range(b1.shape[0]):
    for j in range(b2.shape[1]):
        A = np.matrix([ [1,0,0,b1[i,j,0]], [0,1,0,b1[i,j,1]], [0,0,1,b1[i,j,2]], 
                        [1,0,0,b2[i,j,0]], [0,1,0,b2[i,j,1]], [0,0,1,b2[i,j,2]] ])
        A_pinv = linalg.pinv(A)
        b = np.matrix([ [c1[i,j,0]-b1[i,j,0]], [c1[i,j,1]-b1[i,j,1]], [c1[i,j,2]-b1[i,j,2]],
                        [c2[i,j,0]-b2[i,j,0]], [c2[i,j,1]-b2[i,j,1]], [c2[i,j,2]-b2[i,j,2]] ])
        x = np.clip(A_pinv*b, 0, 1)
        
        for k in range(3):
            f[i,j,k] = x[k]
        
        #if sum(b)/6 <= 0:
        if sum(f[i,j,:])/3 < 0.06:
            alpha[i,j] = 0
        else:
            alpha[i,j] = 1

imshow(alpha)
show()

new_b = imread('images/window.jpg')