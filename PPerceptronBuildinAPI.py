#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 00:27:38 2017

@author: hust
"""

from __future__ import division
import random
import matplotlib.pyplot as plt  

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report,confusion_matrix

iris = load_iris()
# 特征和标签
X= iris.data[0:100, :2]
y = iris.target[0:100]
zero_Index = np.where(y == 0)
y[zero_Index] = -1
# 统计样本书和类别数
n_samples, n_features = X.shape
labelNum = np.unique(y).shape[0]
print('样本数:',n_samples, '特征数:',n_features, '类别数:',labelNum)

# 特征归一化
X = StandardScaler().fit_transform(X)
#训练集和测试机分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

myPerceptron = Perceptron(eta0=1.0, n_iter=200)
myPerceptron.fit(X_train,y_train)
w = myPerceptron.coef_
b = myPerceptron.intercept_
# 预测
labelPredict = myPerceptron.predict(X_test)

zero_IndexNeg = np.where(y == -1)
zero_IndexPos = np.where(y == 1)
plt.scatter(X[zero_IndexNeg, 0],X[zero_IndexNeg, 1], marker='x')  
plt.scatter(X[zero_IndexPos, 0],X[zero_IndexPos, 1], marker='o')  

# 画超平面分布图
x1 = np.linspace(-2, 2, 100)
x2 = (b - w[0,0] * x1) / (w[0,1] +1e-10)
plt.plot(x1, x2, color='r')
plt.show()
        
        
# 混淆矩阵
print(confusion_matrix(y_test,labelPredict))
print(classification_report(y_test,labelPredict))