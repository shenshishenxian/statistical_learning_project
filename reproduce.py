
# coding: utf-8

# In[235]:


#This is the code for reproducing Rule based Classification for Diabetic Patients using Cascaded K-Means and Decision Tree C4.5
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from sklearn import preprocessing

#importing the data
df = pd.read_csv("diabetes.csv", header=0)
ls=[]
for i in range(9):
    ls.append(str(i))
df.columns = ls
X_train = df[ls]
X_train = np.array(X_train)


# In[236]:


#The original paper use Weka for several steps including k-means. We are using Weka interface in R to process the data and use that data with Python
df = pd.read_csv("MyData.csv", header=0)
ls=[]
for i in range(2):
    ls.append(str(i))
df.columns = ls
mydata = df[ls]
mydata = np.array(mydata)


# In[237]:


#Checking if correct number of samples were left after deleting missing zeros
summ=0
for i in range(392):
    if mydata[i][1] == X_label[i]:
        summ+=1
print summ


# In[238]:


#Saving the data after processing
X_train2=[]
total=0
X_label=[]
for i in range(len(X_train)):
    line = X_train[i]
    val = True
    for j in range(1,8):
        if line[j] == 0:
            val = False
    if val:
        total+=1
        X_train2.append(line[0:8])
        X_label.append(line[8])
X_train2=np.array(X_train2)
np.savetxt("afterprocessfirst.csv", X_train2, delimiter=",")


# In[240]:


X_train3=[]
X_label3=[]
kmeans.labels_[0]
for i in range(len(X_train2)):
    if mydata[i][1] == X_label[i]:
        X_label3.append(X_label[i])
        X_train3.append(X_train2[i]) 


# In[242]:


#Changing the continuous data into categorical
X_train4=[]
X_label4=[]
for i in range(len(X_train3)):
    for j in range(len(X_train3[i])):
        if j == 0:
            if X_train3[i][j]<3:
                X_train3[i][j]=0
            elif X_train3[i][j]<=5:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2
        if j == 1:
            if X_train3[i][j]<95:
                X_train3[i][j]=0
            elif X_train3[i][j]<=150:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2
        if j == 2:
            if X_train3[i][j]<70:
                X_train3[i][j]=0
            elif X_train3[i][j]<=100:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2
        if j == 3:
            if X_train3[i][j]<21:
                X_train3[i][j]=0
            elif X_train3[i][j]<=40:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2
        if j == 4:
            if X_train3[i][j]<140:
                X_train3[i][j]=0
            elif X_train3[i][j]<=200:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2
        if j == 5:
            if X_train3[i][j]<23:
                X_train3[i][j]=0
            elif X_train3[i][j]<=29:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2
        if j == 6:
            if X_train3[i][j]<0.4:
                X_train3[i][j]=0
            elif X_train3[i][j]<=0.8:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2
        if j == 7:
            if X_train3[i][j]<34:
                X_train3[i][j]=0
            elif X_train3[i][j]<=46:
                X_train3[i][j]=1
            else :
                X_train3[i][j]=2


# In[243]:


#Saving the categorical data for classification in R
for i in range(len(X_train3)):
    line = np.append(X_train3[i], X_label3[i])
    X_train4.append(line)


# In[246]:


np.savetxt("afterprocess.csv", X_train4, delimiter=",")

