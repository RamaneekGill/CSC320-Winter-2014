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

# c = f + (1-alpha)b

## MAIN
b1 = imread('images/flowers-backA.jpg')/255.0
c1 = imread('images/flowers-compA.jpg')/255.0
b2 = imread('images/flowers-backB.jpg')/255.0
c2 = imread('images/flowers-compB.jpg')/255.0
new_b = imread('images/window.jpg')/255.0

b1 = imresize(imread('my-images/b1.jpg')/255.0, 0.1)
c1 = imresize(imread('my-images/c1.jpg')/255.0, 0.1)
b2 = imresize(imread('my-images/b2.jpg')/255.0, 0.1)
c2 = imresize(imread('my-images/c2.jpg')/255.0, 0.1)
new_b = imresize(imread('my-images/new-b.jpg')/255.0, 0.1)

print "Read the input images"

new_c = np.zeros(new_b.shape)
alpha = np.zeros((b1.shape[0], b1.shape[1]))
f = np.zeros(b1.shape)

for i in range(b1.shape[0]):
    for j in range(b2.shape[1]):
        #set A
        A = np.matrix([ [1,0,0,-b1[i,j,0]], 
                        [0,1,0,-b1[i,j,1]], 
                        [0,0,1,-b1[i,j,2]], 
                        [1,0,0,-b2[i,j,0]], 
                        [0,1,0,-b2[i,j,1]], 
                        [0,0,1,-b2[i,j,2]] ])
                        
        #get the pseudo inverse of A
        A_pinv = linalg.pinv(A)
        
        #set b
        b = np.matrix([ [c1[i,j,0]-b1[i,j,0]], 
                        [c1[i,j,1]-b1[i,j,1]], 
                        [c1[i,j,2]-b1[i,j,2]],
                        [c2[i,j,0]-b2[i,j,0]], 
                        [c2[i,j,1]-b2[i,j,1]], 
                        [c2[i,j,2]-b2[i,j,2]] ])
                        
        #solve for x by A^-1*b
        x = np.clip(dot(A_pinv,b), 0, 1)
        
        #set foreground
        for k in range(3):
            f[i,j,k] = x[k]
        
        #set alpha channel
        alpha[i,j] = x[3]
        
        #set new composite image
        for k in range(3):
            new_c[i,j,k] = f[i,j,k] + np.dot((1-alpha[i,j]), new_b[i,j,k])

print "Computed alpha channel and isolated the foreground"
figure(1)
imshow(alpha)
figure(2)
imshow(f)
figure(3)
imshow(new_c)

show()

            