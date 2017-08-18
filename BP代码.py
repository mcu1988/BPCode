#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:19:40 2017

@author: hust
"""

import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# 载入数据
removeZeroopensmile7200 = scipy.io.loadmat('removeZeroopensmile7200.mat')
X = removeZeroopensmile7200['Data']
y = removeZeroopensmile7200['Label']
# 归一化
X = MinMaxScaler((-1,1)).fit_transform(X)
# 4个人的数据
X1 = X[:514, :]; y1 = y[:514, :]
X2 = X[514:(514+504), :]; y2 = y[514:(514+504), :]
X3 = X[(514+504):(514+504+714), :]; y3 = y[(514+504):(514+504+714), :]
X4 = X[(514+504+714):(514+504+714+455), :]; y4 = y[(514+504+714):(514+504+714+455), :]
# 3个人训练，一个人测试
X_train = np.vstack( (X1, X2, X3) )
y_train= np.vstack( (y1, y2, y3) )
X_test = X4
y_test = y4
# 训练随机森林模型
RF = RandomForestClassifier(n_estimators=500)
RF.fit(X_train, y_train)
# 重要度排序
rfSortedFeatureImportance = sorted(enumerate(RF.feature_importances_), key=lambda x:x[1],  reverse=True)
# 设定阈值，截取要选择的特征
SumImportances = 0
X_trainS = []
y_trainS = y_train
X_testS = []
y_testS = y_test
for i in range(len(rfSortedFeatureImportance)):    #总共10个特征 ！！！！！！！！！！！！！！！！！！！！
    SumImportances = SumImportances + rfSortedFeatureImportance[i][1]
    X_trainS.append ( X_train [ :, rfSortedFeatureImportance[i][0] ])
    X_testS.append ( X_test [ :, rfSortedFeatureImportance[i][0] ])
    if SumImportances > 0.85:   # 阈值0.85
        break
X_trainS = np.transpose( np.array(X_trainS) )
X_testS = np.transpose( np.array(X_testS) )
# SVM多分类器
RFSVM1 = svm.SVC(decision_function_shape='ovo')
RFSVM1.fit(X_trainS, y_trainS)
RFPredict1= RFSVM1.predict(X_testS)
RFAccuracy1 = recall_score(y_testS, RFPredict1, average='micro')
