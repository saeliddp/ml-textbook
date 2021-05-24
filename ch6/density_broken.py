# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:25:45 2021

@author: saeli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

# load the abalone dataset
abalone = pd.read_csv('abalone.data', names=['sex', 'len', 'diam', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings'])

y_train = abalone['sex']
# we'll build our EM model based on length, diameter, and full weight
X_train = abalone[['len', 'diam', 'whole']]

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(X_train)

# display predicted scores by the model as a contour plot
w = np.linspace(np.min(X_train['len']), np.max(X_train['len']))
x = np.linspace(np.min(X_train['diam']), np.max(X_train['diam']))
y = np.linspace(np.min(X_train['whole']), np.max(X_train['whole']))
W, X, Y = np.meshgrid(w, x, y)
XX = np.array([W.ravel(), X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)

Z = Z.reshape(W.shape)
print(Z)
CS = plt.contour(W, X, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train['len'], X_train['diam'], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()