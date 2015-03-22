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

#global variables
act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']
training_set = np.zeros((len(act)*100, 32*32)) - 1  #need to use zeros()!!
validation_set = np.zeros((len(act)*10, 32*32)) - 1 #32*32 since a 32x32 matrix is flattened
test_set = np.zeros((len(act)*10, 32*32)) - 1
    
#fills in the global variable data
def getData():
    count_tr = 0 #training
    count_va = 0 #validation
    count_te = 0 #testing
    k = 135 #every actor has at least 135 pics

    for a in act:
        name = a.split()[1].lower()
        for i in range(k):
            if i == k:
                print "You might want to get more images or lower the training set amount"
                exit()
                
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
            #if training_set[(act.index(a)+1)*100,-1] == -1: #get 100 training images for this actor
            if count_tr < (act.index(a)+1) * 100:
                training_set[count_tr][:] = gray_img.flatten()
                count_tr += 1
            elif count_va < (act.index(a)+1) * 10: #get 10 validation images for this actor
                validation_set[count_va][:] = gray_img.flatten()
                count_va += 1
            elif count_te < (act.index(a)+1) * 10: #get 10 test images for this actor
                test_set[count_te][:] = gray_img.flatten()
                count_te += 1
            else: #got all the pictures needed for this actor, move on to next actor
                break
    print "Training, validation, and test sets created!"
    
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
    print "Projection matrix and mean eigenface computed!"
    return V,S,mean_X
    
def ssd(x, y):
    return sum((x.flatten().astype(float)-y.flatten().astype(float))**2)    
    
def display_save_25_comps(V, im_shape):
    '''Display 25 components in V'''
    figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        gray()
        imshow(V[i,:].reshape(im_shape))
    savefig('report/display_save_25_comps.jpg')  
    show()        

##########################################################################################################################################

getData() #get the data from the cropped images and fill in some of the global variables

projection_M, variance, mean_img = pca(training_set)  #we actually don't even need the variance in this project
display_save_25_comps(projection_M, (32,32))
average_face = np.reshape(mean_img, (32,32)) #keep this to show later on in the report
imsave("report/mean_face.jpg", average_face)

validation_settings = [2, 5, 10, 20, 50, 80, 100, 150, 200] # the top k eigenfaces

#Validation set
max_performance = 0 #for determining the most correct setting for how many eigenfaces to use
best_setting = 0
num_incorrect = 0 #for displaying incorrect matches for reporting purposes

for setting in validation_settings:
    num_correct = 0
    for i in range(validation_set.shape[0]):
        
        #project the validation image on to the eigenface space
        val_proj_img = np.dot(projection_M[:setting], validation_set[i] - mean_img)
        validation_name_index = int(i/10)
        
        #compute the closest projected training face to identify validation_set[i]
        min = Infinity
        for j in range(training_set.shape[0]):
            train_proj_img = np.dot(projection_M[:setting], training_set[j] - mean_img)
            ssd_value = ssd(val_proj_img, train_proj_img)
            if ssd_value < min:
                #print "found new min:", int(j/100)
                min = ssd_value
                #the index of the actor's name
                training_name_index = int(j/100)
                raw_training_name_index = j
                
        #print ssd_value
        #print validation_name, training_name
        if validation_name_index == training_name_index:
            num_correct += 1
            #print "\t found match"
        elif num_incorrect < 5:
            num_incorrect += 1
            print "This match was incorrect, validation image", i, "incorrectly matched actor", validation_name_index, "with ", training_name_index
            imsave("report/"+str(num_incorrect)+"_"+act[validation_name_index]+"_incorrectly_matched.jpg", np.reshape(validation_set[i], (32,32)))
            imsave("report/"+str(num_incorrect)+"_"+act[training_name_index]+"_with.jpg", np.reshape(training_set[raw_training_name_index], (32,32)))
            

    performance = num_correct * 100.0 / validation_set.shape[0]
    print "Performance on the validation set using", setting, "eigenfaces was: ", performance
    if max_performance < performance:
        max_performance = performance
        best_setting = setting
    print "Best setting for face recognition is to use", best_setting, "eigenfaces"
    
#test set
num_correct = 0
for i in range(test_set.shape[0]):
    test_proj_img = np.dot(projection_M[:best_setting], test_set[i] - mean_img)
    test_name_index = int(i/10)
    
    #compute the closest projected training face to identify validation_set[i]
    min = Infinity
    for j in range(training_set.shape[0]):
        train_proj_img = np.dot(projection_M[:best_setting], training_set[j] - mean_img)
        ssd_value = ssd(test_proj_img, train_proj_img)
        if ssd_value < min:
            min = ssd_value
            #the index of the actor's name
            training_name_index = int(j/100)

    if test_name_index == training_name_index:
        num_correct += 1
        
performance = num_correct * 100.0 / test_set.shape[0] 
print "Performance on the test set using", best_setting, "eigenfaces was: ", performance

# Part 3 is done


## Part 4 begin
print "\n\n\nNOW TESTING GENDER RECOGNITION\n\n\n"

#Validation set
max_performance = 0 #for determining the most correct setting for how many eigenfaces to use
best_setting = 0
num_incorrect = 0 #for displaying incorrect matches for reporting purposes

for setting in validation_settings:
    num_correct = 0
    for i in range(validation_set.shape[0]):
        #project the validation image on to the eigenface space
        val_proj_img = np.dot(projection_M[:setting], validation_set[i] - mean_img)
        validation_name_index = int(i/10)
         
        #compute the closest projected training face to identify validation_set[i]
        min = Infinity
        for j in range(training_set.shape[0]):
            train_proj_img = np.dot(projection_M[:setting], training_set[j] - mean_img)
            ssd_value = ssd(val_proj_img, train_proj_img)
            if ssd_value < min:
                min = ssd_value
                training_name_index = int(j/100)
        
        if validation_name_index >= 3 and training_name_index >= 3: #both female
            num_correct += 1
            #print "found match"
        elif validation_name_index < 3 and training_name_index < 3: #both male
            num_correct += 1
        elif num_incorrect < 5:
            num_incorrect += 1
            #print "This match was incorrect, validation image", i, "incorrectly matched actor", validation_name_index, "with ", training_name_index
            #imsave(str(num_incorrect)+act[validation_name_index]+"gender_incorrectly_matched_with"+act[training_name_index]+".jpg")

        
    performance = num_correct * 100.0 / validation_set.shape[0]
    print "Performance on the validation set using", setting, "eigenfaces was: ", performance
    if max_performance < performance:
        max_performance = performance
        best_setting = setting
        
#test set
num_correct = 0
for i in range(test_set.shape[0]):
    test_proj_img = np.dot(projection_M[:best_setting], test_set[i] - mean_img)
    validation_name_index = int(i/10)
    #compute the closest projected training face to identify validation_set[i]
    min = Infinity
    for j in range(training_set.shape[0]):
        train_proj_img = np.dot(projection_M[:best_setting], training_set[j] - mean_img)
        ssd_value = ssd(test_proj_img, train_proj_img)
        if ssd_value < min:
            min = ssd_value
            training_name = int(j/100)

    if validation_name_index >= 3 and training_name_index >= 3: #both female
        num_correct += 1
        #print "found match"
    elif validation_name_index < 3 and training_name_index < 3: #both male
        num_correct += 1
        
performance = num_correct * 100.0 / test_set.shape[0] 
print "Performance on the test set using", best_setting, "eigenfaces was: ", performance

# End part 4




