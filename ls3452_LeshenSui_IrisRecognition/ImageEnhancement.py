
# coding: utf-8

# In[1]:

import os
import numpy as np
from scipy import misc, signal, ndimage
import matplotlib.pyplot as plt
from PIL import Image 
from skimage import io, color, exposure
get_ipython().magic('matplotlib inline')
import warnings
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
warnings.filterwarnings("ignore")
import skimage
from skimage.draw import circle_perimeter
from astropy.modeling.models import Gaussian2D
import IrisLocalization
from skimage.filters import gabor_kernel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
from scipy.spatial import distance


# In[2]:

def background(normalized_image, delta):
    M,N = normalized_image.shape
    background = np.array([[0 for i in range(int(N/delta))] for j in range(int(M/delta))])
    for i in range(int(N/delta)):
        for j in range(int(M/delta)):
            background[j][i] = np.mean(normalized_image[j*delta:j*delta+delta,i*delta:i*delta+delta])
    background = misc.imresize(background,(64,512),'bicubic')
    return background


# In[3]:

def background_subtraction(normalized_image,background):
    return cv2.subtract(normalized_image, background)


# In[4]:

#delta = 32
def enhancement(image, delta):
    M,N = image.shape
    enhanced = image.copy()
    for i in range(int(N/delta)):
        for j in range(int(M/delta)):
            local = enhanced[j*delta:j*delta+delta,i*delta:i*delta+delta]
            local = exposure.equalize_hist(local)
            enhanced[j*delta:j*delta+delta,i*delta:i*delta+delta] = local*255
    return enhanced


# In[ ]:



