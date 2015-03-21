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

    #these store the names of the actor which validation_set[i] and test_set[i] and training_set[i] belong to
    training_names = []
    validation_names = []
    test_names = []

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
                #print "trying next image"
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
    
def ssd(x, y):
    return sum((x-y)**2)    
    
################################################################################################################
data = getData()
training_set = data[0]
validation_set = data[1]
test_set = data[2]

projection_M, variance, mean_img = pca(training_set)

average_face = np.reshape(mean_img, (32,32)) #keep this to show later on in the report
top_25_eigenfaces = np.zeros((25, 32, 32)) #keep this too
for i in range(25):
    top_25_eigenfaces[i] = np.reshape(projection_M[i], (32,32))

validation_settings = [2, 5, 10, 20, 50, 80, 100, 150, 200] # the top k eigenfaces

#store results in here
#results[ith num faces used from validation_settings][accuracy of ith validation image's reconstruction]
results = np.zeros((len(validation_settings), validation_set.shape[0]+1))
# set the first values of the result table with the validation setting
for i in range(len(validation_settings)):
    results[i][0] = validation_settings[i]

#Validation set
max = 0 #for determining the most correct setting for how many eigenfaces to use
best_setting = 0
num_incorrect = 0 #for displaying incorrect matches for reporting purposes
num_correct = 0

for setting in validation_settings:
    for i in range(validation_set.shape[0]):
        #project the validation image on to the eigenface space
        val_proj_img = np.dot(projection_M[:setting], validation_set[i] - mean_img)
         
        #compute the closest projected training face to identify validation_set[i]
        min = Infinity
        for j in range(training_set.shape[0]):
            train_proj_img = np.dot(projection_M[:setting], training_set[j] - mean_img)
            ssd_value = ssd(val_proj_img, train_proj_img)
            if ssd_value < min:
                min = ssd
                #the index of the actor's name
                validation_name = int(i/10)
                training_name = int(j/100)
        
        if validation_name == training_name:
            num_correct += 1
            #print "found match"
        elif num_incorrect < 5:
            num_incorrect += 1
            print "This match was incorrect, validation image", i, "incorrectly matched actor", validation_name, "with ", training_name
        
    performance = num_correct * 100.0 / validation_set.shape[0]
    print "Performance on the validation set using", setting, "eigenfaces was: ", performance
    if max < performance:
        max = performance
        best_setting = setting
    num_correct = 0  
        
#test set
num_correct = 0
for i in range(test_set.shape[0]):
    test_proj_img = np.dot(projection_M[:best_setting], test_set[i] - mean_img)
    
    #compute the closest projected training face to identify validation_set[i]
    min = Infinity
    for j in range(training_set.shape[0]):
        train_proj_img = np.dot(projection_M[:best_setting], training_set[j] - mean_img)
        ssd_value = ssd(test_proj_img, train_proj_img)
        if ssd_value < min:
            min = ssd
            #the index of the actor's name
            validation_name = int(i/10)
            training_name = int(j/100)

    if validation_name == training_name:
        num_correct += 1
        
performance = num_correct / test_set.shape[0] * 100.0
print "Performance on the test set using", best_setting, "eigenfaces was: ", performance

# Part 3 is done


## Part 4 begin

#Validation set
max = 0 #for determining the most correct setting for how many eigenfaces to use
best_setting = 0
num_incorrect = 0 #for displaying incorrect matches for reporting purposes
num_correct = 0

for setting in validation_settings:
    for i in range(validation_set.shape[0]):
        #project the validation image on to the eigenface space
        val_proj_img = np.dot(projection_M[:setting], validation_set[i] - mean_img)
         
        #compute the closest projected training face to identify validation_set[i]
        min = Infinity
        for j in range(training_set.shape[0]):
            train_proj_img = np.dot(projection_M[:setting], training_set[j] - mean_img)
            ssd_value = ssd(val_proj_img, train_proj_img)
            if ssd_value < min:
                min = ssd
                #the index of the actor's name
                validation_name = int(i/10)
                training_name = int(j/100)
        
        if validation_name >= 3 and training_name >= 3: #both female
            num_correct += 1
            #print "found match"
        elif validation_name < 3 and training_name <3: #both male
            num_correct += 1
        elif num_incorrect < 5:
            num_incorrect += 1
            print "This match was incorrect, validation image", i, "incorrectly matched actor", validation_name, "with ", training_name
        
    performance = num_correct * 100.0 / validation_set.shape[0]
    print "Performance on the validation set using", setting, "eigenfaces was: ", performance
    if max < performance:
        max = performance
        best_setting = setting
    num_correct = 0  
        
#test set
num_correct = 0
for i in range(test_set.shape[0]):
    test_proj_img = np.dot(projection_M[:best_setting], test_set[i] - mean_img)
    
    #compute the closest projected training face to identify validation_set[i]
    min = Infinity
    for j in range(training_set.shape[0]):
        train_proj_img = np.dot(projection_M[:best_setting], training_set[j] - mean_img)
        ssd_value = ssd(test_proj_img, train_proj_img)
        if ssd_value < min:
            min = ssd
            #the index of the actor's name
            validation_name = int(i/10)
            training_name = int(j/100)

    if validation_name >= 3 and training_name >= 3: #both female
        num_correct += 1
        #print "found match"
    elif validation_name < 3 and training_name <3: #both male
        num_correct += 1
        
performance = num_correct / test_set.shape[0] * 100.0
print "Performance on the test set using", best_setting, "eigenfaces was: ", performance

# End part 4




