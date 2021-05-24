# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:43:32 2021

@author: saeli
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.mixture import GaussianMixture

# load the included diabetes dataset
diab = load_diabetes(as_frame=True)
# view information about the columns
print(diab.DESCR)
diab_df = diab.data
print(diab.target)
# since we are not performing regression, we can add the target
# column
diab_df['s7'] = diab.target
# print a summary of our data
print(diab_df.describe())

em_gaussian = GaussianMixture(n_components=4, init_params='random', covariance_type='full')
cluster_preds = em_gaussian.fit_predict(diab_df)
plt.title('Gaussian Mixture Clusters')
# we can pick two dimensions of the input data in order to visualize clusters
# in R^2. Note that this output will look different depending on which
# dimensions you choose to plot
plt.xlabel('bmi')
plt.ylabel('bp')
plt.scatter(diab_df['bmi'], diab_df['bp'], c=cluster_preds, cmap='rainbow')
plt.savefig('simple_diabetes_clusters.png', dpi=300)

# view the akaike information criterion
print(em_gaussian.aic(diab_df))
# view the bayesian information criterion
print(em_gaussian.bic(diab_df))


