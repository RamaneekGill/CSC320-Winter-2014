from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os

os.chdir('C:/Users/Ramaneek/SkyDrive/Documents/University/Third Year/CSC320/project 1/')

matplotlib
gray()

#######################################################################

#function to align img1 to img2, returns shifted img1, if use_SSD is > 0 then
#img1 is aligned using SSD, NCC if use_SSD < 0
def align(img1, img2, use_SSD):
    print 'aligning ', image_name

    #so we don't overwrite the original images
    new_img1 = img1 
    new_img2 = img2
    
    if use_SSD > 0:
        x_offset, y_offset = ssd_offset(new_img1, new_img2)
    else:
        x_offset, y_offset = ncc_offset(new_img1, new_img2)
        
    #circurally shifts the matrix, this makes last elements wrap around
    #since only affects a very small edge of the image just leave it
    print 'total offset is (',x_offset,',',y_offset,')'
    new_img1 = roll(new_img1, x_offset, 1)
    new_img1 = roll(new_img1, y_offset, 0)
    
    return new_img1

#aligning images for png files (the big images)
#first match is done on 1/8th of img, then another time on 1/4th
def align_png(img1, img2, use_SSD):
    print 'aligning', image_name
    new_img1 = img1 #refers to resized images that are passed to ncc_offset
    new_img2 = img2 
    aligned_img1 = new_img1 #refers to the most up to date full size aligned img
    totalx = 0; totaly = 0
    
    for ratio in reversed(range(0,4)):
        #aligning gaussian pyramid at level 3->2->1->original img size
        new_img1 = imresize(aligned_img1, 0.5**ratio)
        new_img2 = imresize(img2, 0.5**ratio)
        
        if use_SSD > 0:
            x_offset, y_offset = ssd_offset(new_img1, new_img2)
        else:
            x_offset, y_offset = ncc_offset(new_img1, new_img2)

        #resize offsets to coordinate with the original image size
        #I do 2**ratio because the image was reduced 0.5**ratio
        x_offset = int(x_offset*(2**ratio))
        y_offset = int(y_offset*(2**ratio))
        totalx += x_offset
        totaly += y_offset
        
        print ratio,' level of gaussian pyramid offset is (',x_offset,',',y_offset,')'
        #now make the new_img1 aligned and update aligned_img1 to point to latest update
        aligned_img1 = roll(aligned_img1, x_offset, 1)
        aligned_img1 = roll(aligned_img1, y_offset, 0)
        
        img1=img1

    print 'total offset is (',totalx,',',totaly,')'
    
    return aligned_img1


#function to compute SSD offset of img1 onto img2, returns tuple (x,y)
def ssd_offset(img1, img2):
    print 'using ssd'
    w_height, w_width = img1.shape
    x_final = 0; y_final = 0
    min = Infinity
    for x in range(-10,11): #goes from -10 to 10
        for y in range(-10, 11):
            img1_copy = img1 #so we can reset the image displacement every iteration
            
            #translate images
            img1_copy = roll(img1_copy, x, 1)
            img1_copy = roll(img1_copy, y, 0)
  
            #calculate the ssd with respect to the overlapping area (prof said this was good)
            overlapping_area = (img1_copy.shape[0] - abs(x))*(img1_copy.shape[1] - abs(y))
            ssd = sum((img2 - img1_copy)**2) - overlapping_area
            
            if ssd < min:
                min = ssd
                x_final = x
                y_final = y
    print 'finished ssd'
    return (x_final, y_final)


def ncc_offset(img1, img2):
    print 'using ncc'
    w_height, w_width = img1.shape
    x_final = 0; y_final = 0
    max = -Infinity
    for x in range(-10,11): #goes from -10 to 10
        for y in range(-10, 11):
            img1_copy = img1
            
            img1_copy = roll(img1_copy, x, 1)
            img1_copy = roll(img1_copy, y, 0)
            
            a_prime = img2 - mean(img2)
            b_prime = img1_copy - mean(img1_copy)
            
            dot = np.dot(a_prime.flatten(), b_prime.flatten())
            ncc = dot/(norm(a_prime.flatten())*norm(b_prime.flatten()))
            
            if ncc > max:
                
                max = ncc
                x_final = x
                y_final = y
    print 'finished ncc'
    return (x_final, y_final)

########################################################################

#TO USE THIS PROGRAM INPUT THE FILE NAME BELOW
#TO USE SSD CHANGE THE 'use_SSD' variable to > 0
#TO USE NCC CHANGE THE 'use_SSD' variable to <= 0

start_time = time.time()  #for displaying time the program takes to run

image_name = '01657v.jpg'
use_SSD = 1    # >0 to use SSD, <= 0 to use NCC

i = imread(image_name)

if image_name[-4:] == '.png':
    i = i/float(max(i.flatten())) #change values to 0to1
else:
    i = i.astype(uint8) #convert the image to uint8 just in case

height, width = i.shape
height = height/3 # each r,g,b image's height

# #get rid of the common left and right borders of i, 5% should be enough
# i = i[:, width*0.05:-width*0.05]

#get the rgb channels from the photo, only need to manipulate rows
b_i = i[:height]
g_i = i[height:2*height]
r_i = i[2*height:3*height]

#update the height and width value to correspond with individual rgb imgs
height, width = b_i.shape

#now lets 'crop' each rgb image a little more, 10% should be enough
b_i = b_i[width*0.1:-width*0.1, height*0.1:-height*0.1]
g_i = g_i[width*0.1:-width*0.1, height*0.1:-height*0.1]
r_i = r_i[width*0.1:-width*0.1, height*0.1:-height*0.1]

# figure(2); imshow(b_i)
# figure(3); imshow(g_i)
# figure(4); imshow(r_i)
# show()

#aligned image
if image_name[-3:] == 'png':
    new_g_i = align_png(g_i, b_i, use_SSD)
    new_r_i = align_png(r_i, b_i, use_SSD)
else:
    new_g_i = align(g_i, b_i, use_SSD)
    new_r_i = align(r_i, b_i, use_SSD)

# figure(2); imshow(b_i)
# figure(3); imshow(g_i)
# figure(4); imshow(r_i)
# figure(5); imshow(b_i)
# figure(6); imshow(new_g_i)
# figure(7); imshow(new_r_i)
# show()


#combine the rgb images into one, the 3 is for adding the 3rd dimension
size = b_i.shape

if image_name[-4:] == '.png':
    result = zeros((size[0], size[1], 3))  #because png will operate on floats
else:
    result = zeros((size[0], size[1], 3)).astype(uint8)

result[:, :, 0] = new_r_i
result[:, :, 1] = new_g_i
result[:, :, 2] = b_i

#save the result and then show it
if use_SSD > 0:
    if image_name[-4:] == '.png':
        image_name = image_name[:-4] + '_SSD.png'
    else:
        image_name = image_name[:-4] + '_SSD.jpg'
else:
    if image_name[-4:] == '.png':
        image_name = image_name[:-4] + '_NCC.png'
    else:
        image_name = image_name[:-4] + '_NCC.jpg'
        
imsave(image_name, result)

print 'Algorithm took', -1*(start_time - time.time()),'to run'

figure(1); imshow(result)
show()