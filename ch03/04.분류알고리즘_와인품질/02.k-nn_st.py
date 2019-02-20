# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:44:32 2018
http://scikit-learn.org
@author: kim
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import math

df = pd.read_csv('./preprocessed_redwine_quality.csv', engine='python', encoding='cp949')

train, test = train_test_split(df, test_size=0.3) 

train_x = train.drop('품질수준', 1)    # 1: col
train_y = train['품질수준']
test_x = test.drop('품질수준', 1)
test_y = test['품질수준']

M = int(math.sqrt(len(train_x)))
for k in range(1, M, 2):
     #n_neighbors=5(default), algorithm='auto'(default), metric='minkowski’(default)
    model = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree').fit(train_x, train_y)
    predited = model.predict(test_x)
    print(k, accuracy_score(test_y, predited))

























