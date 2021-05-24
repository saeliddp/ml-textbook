# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:43:32 2021

@author: saeli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.mixture import GaussianMixture

abalone = pd.read_csv('abalone.data', names=['sex', 'len', 'diam', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings'])

y = abalone['sex']
X = abalone.drop(columns='sex')

# we create our model under the assumption that there are three underlying
# gaussian distributions for three classes (M/F/I)
em_gaussian = GaussianMixture(n_components=3, init_params='random', covariance_type='full')
# we'll build our EM model based on all attributes besides sex
cluster_preds = em_gaussian.fit_predict(X)
plt.title('Gaussian Mixture Clusters')

# we can pick two dimensions of the input data in order to visualize clusters
# in R^2. Note that this output will look different depending on which
# dimensions you choose to plot
plt.xlabel('len')
plt.ylabel('whole')

# you can view the real classifications with this line
#plt.scatter(X['len'], X['whole'], c=pd.factorize(y)[0], cmap='rainbow')

# or you can view the predicted classifications with this line
plt.scatter(X['len'], X['whole'], c=cluster_preds, cmap='rainbow')
plt.savefig('simple_abalone_clusters.png', dpi=300)

# view the akaike information criterion
print(em_gaussian.aic(X))
# view the bayesian information criterion
print(em_gaussian.bic(X))


