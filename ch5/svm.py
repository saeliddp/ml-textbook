# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:35:01 2021

@author: saeli
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# read in a small version of the skin data set
skin_df = pd.read_csv('Skin_NonSkin_small.txt', names=['b', 'g', 'r', 'skin'], delimiter='\t')
X = skin_df.drop(columns='skin')
y = skin_df['skin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# train an SVM model
# note that if the full skin data set is used, the fit time
# becomes impractical
print('starting SVM fit')
svm_model = SVC()
svm_model.fit(X_train, y_train)
print('finished SVM fit')

# creating a logistic model will take much less time
# but will ultimately be less accurate for this data
print('starting logistic fit')
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
print('finished logistic fit\n')

svm_preds = svm_model.predict(X_test)
log_preds = log_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('SVM Results:')
print(accuracy_score(y_test, svm_preds))
print(confusion_matrix(y_test, svm_preds))

print('Logistic Regression Results:')
print(accuracy_score(y_test, log_preds))
print(confusion_matrix(y_test, log_preds))