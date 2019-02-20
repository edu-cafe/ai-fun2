# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:18:00 2018

@author: kim
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *

df = pd.read_csv('./preprocessed_redwine_quality.csv', engine='python', encoding='cp949')

train, test = train_test_split(df, test_size=0.3) 

train_x = train.drop('품질수준', 1)    # 1: col
train_y = train['품질수준']
test_x = test.drop('품질수준', 1)
test_y = test['품질수준']

hidden_node1_list = [1,2,3,4,5,6,7,8,9,10]
hidden_node2_list = [0,1,2,3,4,5,6,7,8,9,10]
activation_function_list = ['logistic', 'tanh', 'relu']

for h1 in hidden_node1_list:
    for h2 in hidden_node2_list:
        for activation_function in activation_function_list:
            if h2==0:
                model = MLPClassifier(hidden_layer_sizes=(h1,), activation=activation_function).fit(train_x, train_y)
            else:
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), activation=activation_function).fit(train_x, train_y)
            predited = model.predict(test_x)
            print(h1, h2, activation_function, accuracy_score(test_y, predited))

























