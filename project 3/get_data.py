#for downloading the image data
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
from scipy.ndimage import filters
import urllib


os.chdir('C:/Users/Ramaneek/SkyDrive/Documents/Github/CSC320-Winter-2014/project 3/')

act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']





def timeout(func, args=(), kwargs={}, timeout_duration=0.5, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("faces_subset.txt"):
        if a in line:
            # line.split()[-2] gives the x1,y1,x2,y2 coordinates for specific image in this line
            coordinates = line.split()[-2].split(',')  #an array of x1,y1,x2,y2 coordinates
            
            #filename = actorname + number of pics of this actor + filename
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            print "saved original file:" + filename
            
            if os.path.isfile("uncropped/"+filename): #if the image has been saved
                #need to open this image and crop it
                #no point converting it to grayscale here since imsave saves it as a 3D array
                
                #check to see if image can be read
                try:
                    image_cropped = imread("uncropped/"+filename)
                    image_cropped = image_cropped[int(coordinates[1]):int(coordinates[3]) , int(coordinates[0]):int(coordinates[2])]
                    #now lets resize image to 32x32
                    image_cropped = imresize(image_cropped, (32,32))
                    imsave("cropped/"+filename, image_cropped)
                    print "saved cropped file: " + filename
                except Exception:
                    print "cant open file: " + filename
                    continue 
    
            else:
                print "CONTINUING!!!!!!!!!!!!!!!!!"
                continue

            
            i += 1
    
    