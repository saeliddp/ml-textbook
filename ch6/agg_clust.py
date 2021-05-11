# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:50:09 2021

@author: saeli
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch

# we'll use a preprocessed version of the data which contains no
# N/A values
flakes_df = pd.read_csv('StoneFlakes_clean.csv')
# drop the non-numeric column
flakes_df.drop(columns='ID', inplace=True)

scaler = StandardScaler()
scaler.fit_transform(flakes_df)

# perform agglomerative clustering calculating distance between
# cluster u and v as max(dist(u[i], v[j]))
linkage_matrix = sch.linkage(flakes_df, method='complete')

# plot dendrogram
dendrogram = sch.dendrogram(linkage_matrix)
plt.ylabel('height')
plt.savefig('dendrogram.png', dpi=300)

# for making predictions, you may want to use sklearn's implementation
cluster = AgglomerativeClustering(n_clusters=4, linkage='complete')
cluster_preds = cluster.fit_predict(flakes_df)
print(cluster_preds)