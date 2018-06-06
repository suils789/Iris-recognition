
# coding: utf-8

# In[1]:

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
get_ipython().magic('matplotlib inline')
import PerformanceEvaluation


# In[2]:

base_dir = os.getcwd() + '/CASIA Iris Image Database (version 1.0)'
def read_data(base_dir):
    train_image = []
    test_image = []
    for filename in os.listdir(base_dir)[1:]:
        for file in os.listdir(base_dir + '/' + filename):
            if str(file) == '1':
                for image in os.listdir(base_dir + '/' + filename + '/' + str(file)):
                    try:
                        img = io.imread(base_dir + '/' + filename + '/' + '1' + '/' + str(image))
                        train_image.append(img)
                    except:
                        pass
            elif str(file) == '2':
                for image in os.listdir(base_dir + '/' + filename + '/' + str(file)):
                    try:
                        img = io.imread(base_dir + '/' + filename + '/' + '2' + '/' + str(image))
                        test_image.append(img)
                    except:
                        pass
            else:
                pass
    return train_image, test_image


# In[3]:

train_image, test_image = read_data(base_dir)


# In[4]:

def image_process(image_list):
    features = []
    for image in image_list:
        normalized_image = IrisNormalization.iris_normalization(image)
        enhanced = ImageEnhancement.enhancement(normalized_image, 32)
        sigma_x1,sigma_y1 = 3.0,1.5
        sigma_x2,sigma_y2 = 4.5,1.5
        roi_1 = FeatureExtraction.spatial_filter(enhanced,sigma_x1,sigma_y1,4,4)[0:48, :]
        roi_2 = FeatureExtraction.spatial_filter(enhanced,sigma_x2,sigma_y2,4,4)[0:48, :]
        features.append(FeatureExtraction.feature_extraction(8, roi_1) + FeatureExtraction.feature_extraction(8, roi_2))
    return np.array(features)


# In[5]:

features_train = image_process(train_image)


# In[6]:

features_test = image_process(test_image)


# In[10]:

labels_train = []
for i in range(int(len(features_train)/3)):
    for j in range(3):
        labels_train.append(i + 1)
labels_test = []
for i in range(int(len(features_test)/4)):
    for j in range(4):
        labels_test.append(i + 1)


# In[11]:

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)


# In[12]:

L1_full = PerformanceEvaluation.compute_CRR(features_train, features_test, labels_train,labels_test, metric = 'L1')
L2_full = PerformanceEvaluation.compute_CRR(features_train, features_test, labels_train,labels_test, metric = 'L2')
cosine_full = PerformanceEvaluation.compute_CRR(features_train, features_test, labels_train,labels_test, metric = 'cosine')


# In[13]:

f_train, f_test = IrisMatching.dimentionality_reduction(features_train, features_test,180)


# In[14]:

L1_reduced = PerformanceEvaluation.compute_CRR(f_train, f_test, labels_train,labels_test, metric = 'L1')
L2_reduced = PerformanceEvaluation.compute_CRR(f_train, f_test, labels_train,labels_test, metric = 'L2')
cosine_reduced = PerformanceEvaluation.compute_CRR(f_train, f_test, labels_train,labels_test, metric = 'cosine')


# In[15]:

CRR = {"L1 distance measure":[L1_full,L1_reduced], 
          "L2 distance measure":[L2_full,L2_reduced],
          "cosine distance measure":[cosine_full,cosine_reduced]}
CRR_table = pd.DataFrame.from_dict(CRR,orient='index')
CRR_table.columns = ['Original Feature Set','Reduced Feature Set']
CRR_table


# In[16]:

score_cosine = []
score_L1 = []
score_L2 = []
for num in range(40, 320, 20):
    f_train, f_test = IrisMatching.dimentionality_reduction(features_train, features_test,num)
    L1_reduced = PerformanceEvaluation.compute_CRR(f_train, f_test, labels_train,labels_test, metric = 'L1')
    L2_reduced = PerformanceEvaluation.compute_CRR(f_train, f_test, labels_train,labels_test, metric = 'L2')
    cosine_reduced = PerformanceEvaluation.compute_CRR(f_train, f_test, labels_train,labels_test, metric = 'cosine')
    score_cosine.append((num,cosine_reduced))
    score_L1.append((num,L1_reduced))
    score_L2.append((num,L2_reduced))


# In[17]:

plt.plot(*zip(*score_cosine), '-o')
plt.xlabel("Number of features")
plt.ylabel("CRR")
plt.title("Cosine distance vs. CRR")
plt.show()


# In[18]:

plt.plot(*zip(*score_L1), '-o')
plt.xlabel("Number of features")
plt.ylabel("CRR")
plt.title("L1 distance vs. CRR")
plt.show()


# In[19]:

plt.plot(*zip(*score_L2), '-o')
plt.xlabel("Number of features")
plt.ylabel("CRR")
plt.title("L2 distance vs. CRR")
plt.show()


# In[20]:

f_train, f_test = IrisMatching.dimentionality_reduction(features_train, features_test,80)
d_train = {}
for k,v in zip(list(labels_train),list(f_train)):
    if k not in d_train:
        d_train[k] = []
        d_train[k].append(tuple(v))
    else:
        d_train[k].append(tuple(v))


# In[21]:

d_test = {}
for k,v in zip(list(labels_test),list(f_test)):
    if k not in d_test:
        d_test[k] = []
        d_test[k].append(tuple(v))
    else:
        d_test[k].append(tuple(v))


# In[22]:

matched_pair = []
for k in d_train:
    matched_pair += list(product(d_train[k], d_test[k]))


# In[23]:

unmatched_pair = []
for i in range(1,1 + len(d_train.keys())):
    for j in range(1, 1 + len(d_train.keys())):
        if i != j:
            unmatched_pair += list(product(d_train[i], d_test[j]))


# In[24]:

pairs = matched_pair + unmatched_pair
labels = [1 for x in matched_pair] + [0 for x in unmatched_pair]


# In[25]:

predicted_labels = []
thresholds = [0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8]
for threshold in thresholds:
    temp = []
    for x,y in pairs:
        temp.append(PerformanceEvaluation.verification(x,y,threshold))
    predicted_labels.append(temp)


# In[26]:

FMR = []
FNMR = []
for predicted_label in predicted_labels:
    FMR.append(PerformanceEvaluation.compute_FMR(predicted_label, matched_pair, labels))
    FNMR.append(PerformanceEvaluation.compute_FNMR(predicted_label, matched_pair, labels))


# In[27]:

table_verification = pd.DataFrame(np.array([thresholds,FMR,FNMR]).T, columns = ["Threshold","FMR","FNMR"])
table_verification = table_verification.set_index('Threshold')
table_verification


# In[28]:

plt.plot(FMR,FNMR,'-o')
plt.xlabel("FMR")
plt.ylabel("FNMR")
plt.title("ROC Curve")
plt.show()

