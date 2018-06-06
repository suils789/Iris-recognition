
# coding: utf-8

# In[9]:

import os
import numpy as np
from scipy import misc, signal, ndimage
import matplotlib.pyplot as plt
from PIL import Image 
from skimage import io, color
get_ipython().magic('matplotlib inline')
import warnings
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
warnings.filterwarnings("ignore")
import skimage
from skimage.draw import circle_perimeter
from astropy.modeling.models import Gaussian2D


# imhist = plt.hist(image_1.ravel(), bins = 256)
# plt.show(imhist)


# In[190]:

def edge_detection(image):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    gaussian = 1/159 * np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])
    image = signal.convolve2d(image, gaussian)
    edge_x = signal.convolve2d(image, sobel_x)
    edge_y = signal.convolve2d(image, sobel_y)
    edge_image = np.sqrt(np.square(edge_x) + np.square(edge_y)).astype(int)
    return edge_image


# In[191]:

def threshold(image, num):
    image_bw = image.copy()
    image_bw[image_bw <= num] = 0 
    image_bw[image_bw > num] = 255
    return image_bw


# In[206]:

def find_circles(edge_img, min_radius, max_radius):
    hough_radii = np.arange(min_radius, max_radius, 2)
    hough_res = hough_circle(edge_img, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)
    return list(zip(cy, cx, radii))

def canny_edge_detection(img, sigma):
    gaussian = 1/159 * np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])
    img = signal.convolve2d(img, gaussian)
    edge_image = canny(img, sigma)
    return edge_image

def draw_circle(image, circle):
    for center_y, center_x, radius in circle:
        circy, circx = circle_perimeter(center_y, center_x, radius)
        image[circy, circx] = 0
    io.imshow(image, cmap='gray')


# In[223]:

# def iris_localization(image):
#     def find_pupil(image):
#         image = canny_edge_detection(threshold(image, 50), 10)
#         circle = find_circles(image, 30, 50)
#         return circle
#     def find_iris(image):
#         image = canny_edge_detection(image, 10)
#         circle = find_circles(image, 100, 150)
#         return circle
#     inner_circle = find_pupil(image)
#     outer_circle = find_iris(image)
#     return inner_circle, outer_circle

def iris_localization(image):
    def find_pupil(image):
        image = canny_edge_detection(threshold(image, 70), 10)
        circle = find_circles(image, 30, 50)
        return circle
    def find_iris(image, inner_circle):
        cy, cx, r = inner_circle[0]
        image2 = canny_edge_detection(image, 10)
        circle = find_circles(image2, 102, 120)
        return circle
    inner_circle = find_pupil(image)
    outer_circle = find_iris(image,inner_circle)
    if (outer_circle[0][0] - inner_circle[0][0])**2 + (outer_circle[0][1] - inner_circle[0][1])**2 > 100:
        outer_circle = [(inner_circle[0][0], inner_circle[0][1] ,outer_circle[0][2])]
    return inner_circle, outer_circle


# # In[225]:

# inner_circle = find_pupil(image_3)
# outer_circle = find_iris(image_3)
# image = image_3.copy()
# draw_circle(image, inner_circle)
# draw_circle(image, outer_circle)

