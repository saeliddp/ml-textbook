# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:43:32 2021

@author: saeli
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.mixture import GaussianMixture

# generates pairwise scatterplots showing clusters (similar to clPairs in R)
#
# plots the clusters based on three attributes in a 3x3 grid
# such that the x-axis on the first column is the first attribute in
# the passed-in list, the y-axis on the first row is the first attribute
# in the list, and so on for attributes two and three.
#
# parameters:
#   X             - the dataframe containing data points as rows
#   attr_list     - the list of 3 attributes (columns) from X to plot
#   cluster_list  - the list of cluster labels for each row of X
#   fig           - the figure on which to make the plots
def plot_three_attrs(X, attr_list, cluster_list,fig):
    for i, attr_x in enumerate(attr_list):
        col = i + 1
        for j, attr_y in enumerate(attr_list):
            subplot = fig.add_subplot(330 + col + j * 3)
            subplot.axes.xaxis.set_visible(False)
            subplot.axes.yaxis.set_visible(False)
            if i == j:
                subplot.text(0.5, 0.5, attr_x, ha='center', va='center', transform=subplot.transAxes)
            else:
                subplot.scatter(X[attr_x], X[attr_y], c=cluster_list, cmap='rainbow', s=2)


# load the abalone dataset
abalone = pd.read_csv('abalone.data', names=['sex', 'len', 'diam', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings'])

y = abalone['sex']
# we'll build our EM model based on length, diameter, and full weight
X = abalone[['len', 'diam', 'whole']]

# view descriptive statistics
print(y.describe())
print(X.describe())

# build the model using sklearn's GaussianMixture model
em_gaussian = GaussianMixture(n_components=3, init_params='random', covariance_type='full')
cluster_preds = em_gaussian.fit_predict(X)

fig = plt.figure()

# plot the actual clusters first
#plot_three_attrs(X, ['len', 'diam', 'whole'], pd.factorize(y)[0], fig)
#plt.savefig('pairwise_scatter_real_clusters.png', dpi=300)

# now, plot the predicted clusters
# note that if the colors don't match with the actual clusters, that's okay! 
# we are only concerned with whether or not we estimated the clusters well
plot_three_attrs(X, ['len', 'diam', 'whole'], cluster_preds, fig)
plt.savefig('pairwise_scatter_pred_clusters.png', dpi=300)

# view the akaike information criterion
print(em_gaussian.aic(X))
# view the bayesian information criterion
print(em_gaussian.bic(X))