# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:14:49 2021

@author: saeli
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

# for graphviz on windows
import os
os.environ['PATH'] += os.pathsep + 'C:/Users/saeli/anaconda3/Library/bin/graphviz/'

bal_df = pd.read_csv('balloons.csv')
X = bal_df.drop(columns='inflated')
y = bal_df['inflated']
X.replace(['YELLOW', 'PURPLE'], [0, 1], inplace=True)
X.replace(['SMALL', 'LARGE'], [0, 1], inplace=True)
X.replace(['STRETCH', 'DIP'], [0, 1], inplace=True)
X.replace(['ADULT', 'CHILD'], [0, 1], inplace=True)
y.replace(['F', 'T'], [0, 1], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=('Color', 'size', 'act', 'age'),
                                class_names=('0','1'),
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render('balloons_dt', view=True)