#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:26:33 2017

@author: hust
"""

from __future__ import division
import random
import matplotlib.pyplot as plt  

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report,confusion_matrix


class Perceptron:
    
    def __init__(self):  
        self.w = 0
        self.b = 0
        pass
    
    def train(self, X, y, max_iter=1000, learning_rate=0.1):
        """
        X: [n_sample, features]
        y: [n_sample, 1]
        max_iter: The maximum number of epochs， Defaults to 1000
        learning_rate：Defaults to 1
        """
        
        X = np.array(X)
        w = 0.0
        b = 0.0  # 初始化w和b
        datas_len = len(y)  # 数据个数
        alpha = [0 for i in range(datas_len)] #初始化alpha为0
        gram = np.matmul( X , X.T) # 求Gram矩阵
        for idx in range(max_iter):  # 循环迭代次数
            for i in range(datas_len):  # 循环数据点，知道找到一个误分类点
                yi = y[i]
                tmp = 0
                for j in range(datas_len):  # 判断当前点是否分类正确
                    tmp += alpha[j]*y[j]*gram[i,j]
                tmp += b
                if yi*tmp <=0 :  # 找到第一个分类错误的点
                    alpha[i] += learning_rate
                    b += learning_rate * yi
                    break
        for i in range(datas_len):
            w += alpha[i] * X[i,:] * y[i]
        self.w = w
        self.b = b
    
    def predict(self,X_test):
        self.w.reshape(2,1)
        y_predict = np.where( np.dot(self.w.T, X_test.T) + self.b >0, 1, -1 )
        
        return y_predict
         
    def plot_points(self,X):
        plt.figure()
        # 画原数据分布图
        zero_IndexNeg = np.where(y == -1)
        zero_IndexPos = np.where(y == 1)
        plt.scatter(X[zero_IndexNeg, 0],X[zero_IndexNeg, 1], marker='x')  
        plt.scatter(X[zero_IndexPos, 0],X[zero_IndexPos, 1], marker='o')  
        
        # 画超平面分布图
        x1 = np.linspace(-2, 2, 100)
        x2 = (-self.b - self.w[0] * x1) / (self.w[1] +1e-10)
        plt.plot(x1, x2, color='r')
        plt.show()


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

# 训练加预测
myPerceptron = Perceptron()
myPerceptron.train(X_train, y_train, max_iter=200)
myPerceptron.plot_points(X)
labelPredict = myPerceptron.predict(X_test)

# 混淆矩阵
print(confusion_matrix(y_test,labelPredict))
print(classification_report(y_test,labelPredict))
