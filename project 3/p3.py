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

os.chdir('C:/Users/Ramaneek/SkyDrive/Documents/Github/CSC320-Winter-2014/project 3/')

act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']

#creates an array of flattened arrays in rows of the training data
#returns a tuple of 3 arrays; one for training, validation, testing (all rows of the array is flattened)
def getData():
    training_set = np.zeros((len(act)*100, 32*32)) - 1  #need to use zeros()!!
    validation_set = np.zeros((len(act)*10, 32*32)) - 1 #32*32 since a 32x32 matrix is flattened
    test_set = np.zeros((len(act)*10, 32*32)) - 1

    count_tr = 0 #training
    count_va = 0 #validation
    count_te = 0 #testing
    k = 140 #every actor has at least 140 pics

    
    for a in act:
        name = a.split()[1].lower()
        for i in range(k):
            
            if os.path.isfile("cropped/"+name+str(i)+".jpg"):
                #print "JPG"
                img = imread("cropped/"+name+str(i)+".jpg")
            elif os.path.isfile("cropped/"+name+str(i)+".png"):
                #print "PNG"
                img = imread("cropped/"+name+str(i)+".png")
            else: #couldn't open this image
                print "trying next image"
                continue
            
            #need to convert img to gray scale
            gray_img = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
            if training_set[-1,-1] == -1: #get 100 training images for this actor
                training_set[count_tr][:] = gray_img.flatten()
                count_tr += 1
            elif validation_set[-1,-1] == -1: #get 10 validation images for this actor
                validation_set[count_va][:] = gray_img.flatten()
                count_va += 1
            elif test_set[-1,-1] == -1: #get 10 test images for this actor
                test_set[count_te][:] = gray_img.flatten()
                count_te += 1
            else: #got all the pictures needed for this actor, move on to next actor
                break
    
    return (training_set, validation_set, test_set)


def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        From: Jan Erik Solem, Programming Computer Vision with Python
        #http://programmingcomputervision.com/
    """
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X
    
def get_reconstruction(V, im, mean_im, num_eigenfaces):
    coefs = [np.dot(V[i,:], (im-mean_im)) for i in range(num_eigenfaces)]
    new_im = mean_im.copy()
    for i in range(len(coefs)):
        new_im = new_im + coefs[i]*V[i, :]
    return new_im
    
###############################################################################
data = getData()
training_set = data[0]
validation_set = data[1]
test_set = data[2]

projection_M, variance, mean_img = pca(training_set)

average_face = np.reshape(mean_img, (32,32)) #keep this to show later on in the report
top_25_eigenfaces = np.zeros((25, 32, 32)) #keep this too
for i in range(25):
    top_25_eigenfaces[i, :, :] = np.reshape(projection_M[i, :], (32,32))

validation_settings = [2, 5, 10, 20, 50, 80, 100, 150, 200] # the top k eigenfaces
#store results in here
#results[ith num faces used from validation_settings][accuracy of ith validation image's reconstruction]
results = np.zeros((len(validation_settings), 10))

for i in range(validation_set.shape[0]):
    for setting in validation_settings:
        #get a reconstructed face
        reconstructed_face = get_reconstruction(projection_M, validation_set[i], mean_img, setting)
        #compute how close that face is
        















