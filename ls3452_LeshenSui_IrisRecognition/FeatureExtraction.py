
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

def gobar_filter(x,y,sigma_x,sigma_y):
    G = (1/(2*np.pi*sigma_x*sigma_y)) * np.exp(-0.5*(x**2/sigma_x**2 + y**2/sigma_y**2))
    return G

def M1(x,y,f):
    return np.cos(2*np.pi*f*(np.sqrt(x**2 + y**2)))

def spatial_filter(enhanced, sigma_x,sigma_y,height,width):
    kernel = np.array([[0.0 for i in range(9)] for j in range(9)])
    for y in range(0 - height,height+1):
        for x in range(0 - width,width+1):
            kernel[height+y,width+x] = gobar_filter(x,y,sigma_x,sigma_y) * M1(x,y,1/sigma_y);
    return signal.convolve2d(enhanced,kernel,'same')


# In[ ]:

def feature_extraction(delta, roi):
    h,w = roi.shape
    features = []
    for i in range(int(w/delta)):
        for j in range(int(h/delta)):
            mean = np.mean(roi[j*delta:j*delta+delta, i*delta:i*delta+delta])
            std = np.std(roi[j*delta:j*delta+delta, i*delta:i*delta+delta])
            features+=[mean,std]
    return features

