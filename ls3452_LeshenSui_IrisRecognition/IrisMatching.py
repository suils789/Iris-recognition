
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[ ]:

def nearest_neighbor(features_train, features_test, labels_train, n):
    neighbors = []
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.fit_transform(features_test)
    for i in range(len(features_test)):
        neighbor = 0
        dis = np.inf
        for j in range(len(features_train)):
            d = distance.minkowski(features_test[i],features_train[j],p = n)
            if d < dis:
                dis = d
                neighbor = j
        neighbors.append(labels_train[neighbor])
    return neighbors

def nearest_neighbor_cosine(features_train, features_test, labels_train):
    neighbors = []
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.fit_transform(features_test)
    for i in range(len(features_test)):
        neighbor = 0
        dis = np.inf
        for j in range(len(features_train)):
            d = distance.cosine(features_test[i],features_train[j])
            if d < dis:
                dis = d
                neighbor = j
        neighbors.append(labels_train[neighbor])
    return neighbors


# In[ ]:

def dimentionality_reduction(features_train, features_test,components = 324):
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.fit_transform(features_test)
    pca = PCA(n_components = components)
    f_train = pca.fit_transform(features_train)
    f_test = pca.transform(features_test)
    f_train = scaler.fit_transform(f_train)
    f_test = scaler.fit_transform(f_test)
    return f_train, f_test

