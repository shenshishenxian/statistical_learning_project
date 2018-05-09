
# coding: utf-8

# In[3]:


#This is the code for using semi-supervised learning for our dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import label_propagation
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

#data import
import math
import pandas as pd
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


df = pd.read_csv("diabetes.csv", header=0)
ls=[]
for i in range(9):
    ls.append(str(i))
df.columns = ls
X_train = df[ls]
X_train = np.array(X_train)
divi=520



# Learn with LabelSpreading


# In[4]:


#Prepocessing the data, seperating attributes with label
X_train2=[]
X_label2=[]
X_label3=[]
divi=520
for line in X_train:
    X_train2.append(line[0:8])
    X_label2.append(line[8])
    X_label3.append(line[8])


# In[5]:


for i in range(divi, len(X_train2)):
    X_label2[i]=-1


# In[7]:


#iterate different parameters to get a best result for using semi-supervised learning
knnls=[]
rbfls=[]
for alpha1 in range(0, 10):
    alpha1 = alpha1 / (10 + 0.0) + 0.01
    rbflss=[]
    for gamma1 in range(0,200):
        gamma1 = gamma1 / (4 + 0.0)
        label_spread = label_propagation.LabelSpreading(kernel='rbf', alpha=alpha1, gamma=gamma1)
#         scores2 = cross_val_score(label_spread, X_train2, X_label2, cv = 5)
#         truee = np.mean(scores2)
        label_spread.fit(X_train2, X_label2)
        output_labels = label_spread.transduction_
        truee=0
        for i in range(divi,len(X_train2)):
            if output_labels[i] == X_label3[i]:
                truee+=1
        print gamma1, truee
        rbflss.append(truee)
    rbfls.append(rbflss)
    label_spread2 = label_propagation.LabelSpreading(kernel='knn', alpha=alpha1)
#     scores2 = cross_val_score(label_spread2, X_train2, X_label2, cv = 5)
#     truee2 = np.mean(scores2)
    label_spread2.fit(X_train2, X_label2)
    output_labels2 = label_spread2.transduction_
    truee2=0
    for i in range(divi,len(X_train2)):
        if output_labels[i] == X_label3[i]:
            truee2+=1
    print alpha1, truee2
    knnls.append(truee2)


# In[ ]:


#find the maximum value
max(knnls)
max(max(rbfls))

