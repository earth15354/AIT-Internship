import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd
from sklearn import neighbors,metrics
from sklearn.preprocessing import LabelEncoder

from sklearn import svm

from sklearn import linear_model

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

# # PLOTTING GRAPHS
# x = [i for i in range(10)]
# y = [2*i for i in range(10)]

# plt.plot(x,y)

# plt.xlabel("x axis")
# plt.ylabel("y axis")

# plt.scatter(x,y)

#----------

# TRAIN TEST SPLIT
iris = datasets.load_iris()

features = iris.data
labels = iris.target

# 20% of our data will be used for testing purposes
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(features, labels, test_size=0.2)

#------------

# K NEAREST NEIGHBORS
data = pd.read_csv('Datasets/car.data')
#print(data.head)

# Using only buying, maintenance, and safety
x = data[['buying','maint','safety']].values
y = data[['class']]

# Converting x (to numerical features)
le = LabelEncoder()
for i in range(len(x[0])):
    x[:,i] = le.fit_transform(x[:,i])

# Converting y (to numerical labels)
label_map = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_map)
y = np.array(y)

# Create Model

# Note: n_neighbors is k
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform') 

# Train test split, 20% of data is for testing
x_train_car, x_test_car, y_train_car, y_test_car = train_test_split(x, y, test_size=0.2)

# Trains model
knn.fit(x_train_car,y_train_car)

prediction_car = knn.predict(x_test_car)

accuracy_car = metrics.accuracy_score(y_test_car, prediction_car)

print("knn accuracy: " + str(accuracy_car))

# a = 17
# print("actual: ", y[a])
# print("predicted: ", knn.predict(x)[a])

#——————————————————

# SUPPORT VECTOR MACHINE

# note: using iris database (flowers)
classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

model = svm.SVC()

model.fit(x_train_iris, y_train_iris)

prediction_iris = model.predict(x_test_iris)

accuracy_iris = metrics.accuracy_score(y_test_iris, prediction_iris)

print("svm accuracy: " + str(accuracy_iris))

#——————————————————————

# LINEAR REGRESSION AND LOGARITHMIC REGRESSION

boston = datasets.load_diabetes()

X_bos = boston.data
y_bos = boston.target

# algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X_bos.T[5], y_bos) # <--- Can look at all features!
plt.show()

X_train_bos, X_test_bos, y_train_bos, y_test_bos = train_test_split(X_bos, y_bos, test_size=0.2)

model_lin = l_reg.fit(X_train_bos, y_train_bos)
prediction_bos = model_lin.predict(X_test_bos)

#print("boston predictions: " + str(prediction_bos)) 
print("R^2 value: " + str(l_reg.score(X_bos, y_bos)))
print("Coefficients: " + str(l_reg.coef_))
print("Intersect: " + str(l_reg.intercept_))

# ————————————————————

# K MEANS CLUSTERING

bc = datasets.load_breast_cancer()
x_bc = scale(bc.data)
y_bc = bc.target

x_train_bc, x_test_bc, y_train_bc, y_test_bc = train_test_split(x_bc, y_bc, test_size=0.2)

km_model = KMeans(n_clusters = 2, random_state = 0)
km_model.fit(x_train_bc)

predictions = km_model.predict(x_test_bc)
accuracy_km = metrics.accuracy_score(y_test_bc, predictions)

print("km accuracy: ", str(accuracy_km))

# Note: We might have switched 0 and 1 because k means doesn't know 
# what the labels are called. So accuracy score might be inaccurate.
#———————————————————————

# MULTILAYER PERCEPTRON - NEURAL NETWORK
per = Perceptron()
nn = MLPClassifier(activation="logistic", solver='sgd', hidden_layer_sizes=(45,50), random_state=1)

# More hidden layers usually means more accuracy
nn.fit(x_train_bc, y_train_bc)

predictions = nn.predict(x_test_bc)
accuracy = metrics.accuracy_score(y_test_bc, predictions)

print("nn accuracy: ", str(accuracy))

#————————————————————————

# DECISION TREE
tree = DecisionTreeClassifier()
tree.fit(x_train_bc, y_train_bc)

predictions = tree.predict(x_test_bc)
accuracy = metrics.accuracy_score(y_test_bc, predictions)

print("tree accuracy: ", str(accuracy))