# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:07:43 2021

@author: saeli
"""
# we will rely on existing libraries for logistic regression
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


t_df = pd.read_csv('titanic_data.csv', index_col='id')
# get rid of columns that can't reasonably be converted into numeric
# values
t_df.drop(columns=['name', 'cabin', 'home.dest', 'ticket'], inplace=True)

# convert non-numeric columns into numeric
t_df['sex'].replace(['male', 'female'], [1, 0], inplace=True)
t_df['embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

# form our predictor and response columns
X = t_df.drop(columns=['survived'])
y = t_df['survived']

# split the data into 70% training, 30% evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# create logistic regression model from 70% of the data
logmodel = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=False)
print(logmodel.summary())

from sklearn.metrics import accuracy_score, confusion_matrix
# form our predictions, revert continuous [0, 1] predictions to binary
predictions = logmodel.predict(sm.add_constant(X_test))
bin_predictions = [1 if x >= 0.5 else 0 for x in predictions]
# we can now assess the accuracy, tpr, fpr, tnr, tnr
print(accuracy_score(y_test, bin_predictions))
print(confusion_matrix(y_test, bin_predictions))

# plot the ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

plt.plot(fpr, tpr, label='ROC Curve (area = %0.3f)' % roc_auc)
plt.title('ROC Curve (area = %0.3f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')