# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:48:36 2021

@author: saeli
"""
import pandas as pd
import matplotlib.pyplot as plt

bank_df = pd.read_csv('bank.csv', delimiter=';')
y = bank_df['y']

# plot bar graph
plt.bar([1, 2], [len(y[y == 'yes']), len(y[y == 'no'])], tick_label=['yes', 'no'])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# convert categorical data to int representations of unique categories
for col in bank_df.columns:
    labels, uniques = pd.factorize(bank_df[col])
    bank_df[col] = labels
    
y = bank_df['y']
X = bank_df.drop(columns='y')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
predictions = rfc.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))