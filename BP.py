#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 09:07:31 2017

@author: hust
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
# iris数据集
iris = load_iris()
n_samples, n_features = iris.data.shape
labelNum = np.unique(iris.target).shape[0]
print('样本数:',n_samples, '特征数:',n_features, '类别数:',labelNum)
# 特征和标签
X= iris.data
y = iris.target
# 特征归一化
X = StandardScaler().fit_transform(X)
#训练集和测试机分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 训练模型
MLP = MLPClassifier(hidden_layer_sizes=(n_features,200,labelNum),max_iter=500)
MLP.fit(X_train,y_train)
# 预测
labelPredict = MLP.predict(X_test)
# 混淆矩阵
print(confusion_matrix(y_test,labelPredict))
print(classification_report(y_test,labelPredict))