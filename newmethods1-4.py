import pandas
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# # Without Preprocessing

df = pandas.read_csv('./data/diabetes.csv', delimiter=',')
df_matrix = df.as_matrix()

data = df_matrix[:,0:8]
label = df_matrix[:,8]

clf = XGBClassifier(max_depth=2, min_child_weight=1, gamma=0.2, subsample=0.6, colsample_bytree=0.6, reg_alpha=1e-05)
scores = cross_val_score(clf, data, label, cv=5)
print "XGBoost Accuracy: %.2f %% "%(np.mean(scores) * 100)

ada = AdaBoostClassifier()
scores = cross_val_score(ada, data, label, cv=5)
print "Adaboost Accuracy: %.2f %% "%(np.mean(scores) * 100)

nn = MLPClassifier(alpha = 1e-06, activation = 'relu', solver = 'lbfgs', learning_rate = 'invscaling',
               hidden_layer_sizes = (7, ))
scores = cross_val_score(nn, data, label, cv = 5)
print "Neural Network Accuracy: %.2f %% "%(np.mean(scores) * 100)

knn = KNeighborsClassifier(n_neighbors = 22, weights = 'distance', algorithm = 'brute')
scores =  cross_val_score(knn, data, label, cv = 5)
print "kNN Accuracy: %.2f %% "%(np.mean(scores) * 100)

nb = GaussianNB()
scores = cross_val_score(nb, data, label, cv = 5)
print "Gaussian Naive Bayes: %.2f %% "%(np.mean(scores) * 100)


# # Preprocessing1 (Normalization)

df = pandas.read_csv('./data/diabetes.csv', delimiter=',')
df_matrix = df.as_matrix()

data = df_matrix[:,0:8]
label = df_matrix[:,8]

col_max = np.max(data, axis=0)
col_min = np.min(data, axis = 0)
for i in range(len(col_max)):
    data[:,i] = (data[:,i] - col_min[i]) / (col_max[i] - col_min[i])

clf = XGBClassifier(max_depth=2, min_child_weight=1, gamma=0.2, subsample=0.6, colsample_bytree=0.6, reg_alpha=1e-05)
scores = cross_val_score(clf, data, label, cv=5)
print "XGBoost Accuracy: %.2f %% "%(np.mean(scores) * 100)

ada = AdaBoostClassifier()
scores = cross_val_score(ada, data, label, cv=5)
print "Adaboost Accuracy: %.2f %% "%(np.mean(scores) * 100)

nn = MLPClassifier(alpha = 1e-06, activation = 'relu', solver = 'lbfgs', learning_rate = 'invscaling',
               hidden_layer_sizes = (7, ))
scores = cross_val_score(nn, data, label, cv = 5)
print "Neural Network Accuracy: %.2f %% "%(np.mean(scores) * 100)

knn = KNeighborsClassifier(n_neighbors = 22, weights = 'distance', algorithm = 'brute')
scores =  cross_val_score(knn, data, label, cv = 5)
print "kNN Accuracy: %.2f %% "%(np.mean(scores) * 100)

nb = GaussianNB()
scores = cross_val_score(nb, data, label, cv = 5)
print "Gaussian Naive Bayes: %.2f %% "%(np.mean(scores) * 100)


# # Preprocessing2 (Discretization)

df = pandas.read_csv('./data/diabetes.csv', delimiter=',')
df_matrix = df.as_matrix()

bins_preg = np.array([0,2,5])
bins_glu = np.array([0,95,140])
bins_bp = np.array([0,80,90])
bins_bmi = np.array([0,18.5,25,30,35])
bins_dpf = np.array([0,0.42,0.82])
bins_age = np.array([0,41,61])

data[:,0] = np.digitize(df_matrix[:,0], bins_preg)
data[:,1] = np.digitize(df_matrix[:,1], bins_glu)
data[:,2] = np.digitize(df_matrix[:,2], bins_bp)
data[:,3] = df_matrix[:,3]
data[:,4] = df_matrix[:,4]
data[:,5] = np.digitize(df_matrix[:,5], bins_bmi)
data[:,6] = np.digitize(df_matrix[:,6], bins_dpf)
data[:,7] = np.digitize(df_matrix[:,7], bins_age)
label = df_matrix[:,8]

col_max = np.max(data, axis=0)
col_min = np.min(data, axis = 0)
for i in range(len(col_max)):
    data[:,i] = (data[:,i] - col_min[i]) / (col_max[i] - col_min[i])
    
clf = XGBClassifier(max_depth=2, min_child_weight=1, gamma=0.2, subsample=0.6, colsample_bytree=0.6, reg_alpha=1e-05)
scores = cross_val_score(clf, data, label, cv=5)
print "XGBoost Accuracy: %.2f %% "%(np.mean(scores) * 100)

ada = AdaBoostClassifier()
scores = cross_val_score(ada, data, label, cv=5)
print "Adaboost Accuracy: %.2f %% "%(np.mean(scores) * 100)

nn = MLPClassifier(alpha = 1e-06, activation = 'relu', solver = 'lbfgs', learning_rate = 'invscaling',
               hidden_layer_sizes = (7, ))
scores = cross_val_score(nn, data, label, cv = 5)
print "Neural Network Accuracy: %.2f %% "%(np.mean(scores) * 100)

knn = KNeighborsClassifier(n_neighbors = 22, weights = 'distance', algorithm = 'brute')
scores =  cross_val_score(knn, data, label, cv = 5)
print "kNN Accuracy: %.2f %% "%(np.mean(scores) * 100)

nb = GaussianNB()
scores = cross_val_score(nb, data, label, cv = 5)
print "Gaussian Naive Bayes: %.2f %% "%(np.mean(scores) * 100)


# # Preprocessing3 (Discretization + Normalization)

df = pandas.read_csv('./data/diabetes.csv', delimiter=',')
df_matrix = df.as_matrix()

bins_preg = np.array([0,2,5])
bins_glu = np.array([0,95,140])
bins_bp = np.array([0,80,90])
bins_bmi = np.array([0,18.5,25,30,35])
bins_dpf = np.array([0,0.42,0.82])
bins_age = np.array([0,41,61])

data[:,0] = np.digitize(df_matrix[:,0], bins_preg)
data[:,1] = np.digitize(df_matrix[:,1], bins_glu)
data[:,2] = np.digitize(df_matrix[:,2], bins_bp)
data[:,3] = df_matrix[:,3]
data[:,4] = df_matrix[:,4]
data[:,5] = np.digitize(df_matrix[:,5], bins_bmi)
data[:,6] = np.digitize(df_matrix[:,6], bins_dpf)
data[:,7] = np.digitize(df_matrix[:,7], bins_age)
label = df_matrix[:,8]


clf = XGBClassifier(max_depth=2, min_child_weight=1, gamma=0.2, subsample=0.6, colsample_bytree=0.6, reg_alpha=1e-05)
scores = cross_val_score(clf, data, label, cv=5)
print "XGBoost Accuracy: %.2f %% "%(np.mean(scores) * 100)

ada = AdaBoostClassifier()
scores = cross_val_score(ada, data, label, cv=5)
print "Adaboost Accuracy: %.2f %% "%(np.mean(scores) * 100)

nn = MLPClassifier(alpha = 1e-06, activation = 'relu', solver = 'lbfgs', learning_rate = 'invscaling',
               hidden_layer_sizes = (7, ))
scores = cross_val_score(nn, data, label, cv = 5)
print "Neural Network Accuracy: %.2f %% "%(np.mean(scores) * 100)

knn = KNeighborsClassifier(n_neighbors = 22, weights = 'distance', algorithm = 'brute')
scores =  cross_val_score(knn, data, label, cv = 5)
print "kNN Accuracy: %.2f %% "%(np.mean(scores) * 100)

nb = GaussianNB()
scores = cross_val_score(nb, data, label, cv = 5)
print "Gaussian Naive Bayes: %.2f %% "%(np.mean(scores) * 100)

