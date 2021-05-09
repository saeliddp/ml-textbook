# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:41:43 2021

@author: saeli
"""
import pandas as pd
# LogisticRegression can be configured for multi-class classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

hsb_df = pd.read_csv('hsbdemo.csv')
X = hsb_df[['ses', 'write']]
y = hsb_df['prog']
X['ses'].replace(['low', 'middle', 'high'], [0, 1, 2], inplace=True)
y.replace(['general', 'vocation', 'academic'], [0, 1, 2], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# specify that we will perform non-binary classification
logmodel = LogisticRegression(multi_class='multinomial')
logmodel.fit(X_train, y_train)
print(accuracy_score(logmodel.predict(X_test), y_test))
