import IrisLocalization
import IrisNormalization
import FeatureExtraction
import ImageEnhancement
import IrisMatching
import os
from skimage import io
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations,product
# coding: utf-8

# In[ ]:

def compute_CRR(features_train, features_test, labels_train,labels_test, metric = 'cosine'):
    if metric == 'L1':
        prediction = IrisMatching.nearest_neighbor(features_train, features_test, labels_train,1)
    elif metric == 'L2':
        prediction = IrisMatching.nearest_neighbor(features_train, features_test, labels_train,2)
    else:
        prediction = IrisMatching.nearest_neighbor_cosine(features_train, features_test, labels_train)
    return (list(np.array(prediction) - np.array(labels_test)).count(0))/len(features_test)


# In[ ]:

def verification(x,y,threshold):
    if distance.cosine(x,y) < threshold:
        return 1
    else:
        return 0

def compute_FMR(predicted_label, matched_pair, labels):
    return 1 - (list(np.array(predicted_label[:len(matched_pair)]) - np.array(labels[:len(matched_pair)])).count(0))/sum(predicted_label)
def compute_FNMR(predicted_label, matched_pair, labels):
    return 1 - (list(np.array(predicted_label[:len(matched_pair)]) - np.array(labels[:len(matched_pair)])).count(0))/len(matched_pair)

