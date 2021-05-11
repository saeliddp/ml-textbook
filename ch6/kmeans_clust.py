# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:54:41 2021

@author: saeli
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# we'll use a preprocessed version of the data which contains no
# N/A values
flakes_df = pd.read_csv('StoneFlakes_clean.csv')
# drop the non-numeric column
flakes_df.drop(columns='ID', inplace=True)

scaler = StandardScaler()
scaler.fit_transform(flakes_df)

# perform k-means clustering and print clusters for data
kmeans = KMeans(n_clusters=4)
cluster_preds = kmeans.fit_predict(flakes_df)
centroids = kmeans.cluster_centers_
print(cluster_preds)

plt.title('K-Means Clusters')
# we can pick two dimensions of the input data in order to visualize clusters
# in R^2. Note that this output will look different depending on which
# dimensions you choose to plot
plt.xlabel('WDI')
plt.ylabel('PROZD')
plt.scatter(flakes_df['WDI'], flakes_df['PROZD'], c=cluster_preds, cmap='rainbow')
# we only want to plot the centroids based on their entries corresponding
# to the plot dimensions
plt.scatter(centroids[:,2], centroids[:,-1], c='black',s=70)
plt.savefig('kmeans_clusters.png', dpi=300)
