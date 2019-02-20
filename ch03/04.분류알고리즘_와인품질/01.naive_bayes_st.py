# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:44:32 2018
http://scikit-learn.org/
@author: kim
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.metrics import *

df = pd.read_csv('./preprocessed_redwine_quality.csv', engine='python', encoding='cp949')

train, test = train_test_split(df, test_size=0.3)

train_x = train.drop('품질수준', 1)    # 1: col
train_y = train['품질수준']
test_x = test.drop('품질수준', 1)
test_y = test['품질수준']

model1 = BernoulliNB().fit(train_x, train_y)
model2 = MultinomialNB().fit(train_x, train_y)
model3 = GaussianNB().fit(train_x, train_y)

predict1 = model1.predict(test_x)
predict2 = model2.predict(test_x)
predict3 = model3.predict(test_x)

print(accuracy_score(test_y, predict1))
print(accuracy_score(test_y, predict2))
print(accuracy_score(test_y, predict3))


















