# -*- coding: utf-8 -*-
"""
Created on Sun May 23 22:34:39 2021

@author: saeli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

abalone = pd.read_csv('abalone.data', names=['sex', 'len', 'diam', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings'])
y = abalone['sex']
X = abalone[['len', 'diam', 'whole']]

plt.title('BIC Scores, Components, and CV Types')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')

lowest_bic = np.infty

# we'll compare BIC scores for four different CV types and
# 6 different numbers of components (clusters) to choose the "best"
# model
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    scores = []
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        curr_bic = gmm.bic(X)
        
        scores.append(curr_bic)
        # update tracking variables if new lowest BIC found
        if curr_bic < lowest_bic:
            lowest_bic = curr_bic
            best_gmm = gmm
            
    plt.plot(n_components_range, scores, label=cv_type)

# now we can inspect the "best" model, as decided by BIC score
print('CV:', best_gmm.covariance_type, '| #Components:', best_gmm.n_components, '| BIC:', lowest_bic)
plt.legend()
plt.savefig('BIC_plot.png', dpi=300)
