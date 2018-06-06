
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

def iris_normalization(image):
    pupil, iris = IrisLocalization.iris_localization(image)
    iris_x, iris_y, iris_r = iris[0]
    pupil_x, pupil_y, pupil_r = pupil[0]
    M = 64
    N = 512
    new = [[0 for i in range(N)] for j in range(M)]
    for i in range(M):
        for j in range(N):
            theta = 2*np.pi*j/N
            x = pupil_r*np.cos(theta) + (iris_r*np.cos(theta) - pupil_r*np.cos(theta))*(i/M)
            y = pupil_r*np.sin(theta) + (iris_r*np.sin(theta) - pupil_r*np.sin(theta))*(i/M)
            new[i][j] = image[min(279,int(y) + pupil_x)][min(319,int(x) + pupil_y)]
    return np.array(new)


# In[ ]:



