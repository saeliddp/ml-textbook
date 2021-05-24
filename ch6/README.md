# Chapter 6: Clustering

## agg_clust.py
Uses scipy and sklearn to demonstrate agglomerative clustering. Creates
dendrogram.png. 
    ORIGINAL: https://archive.ics.uci.edu/ml/datasets/StoneFlakes -> StoneFlakes.dat 
    CLEANED: StoneFlakes_clean.csv

## kmeans_clust.py
Uses sklearn to demonstrate K-Means clustering. Visualizes clusters in
two dimensions. Creates kmeans_clusters.png. 
    ORIGINAL: https://archive.ics.uci.edu/ml/datasets/StoneFlakes -> StoneFlakes.dat 
    CLEANED: StoneFlakes_clean.csv

## simple_em_diabetes.py [DON'T USE--See below for alternatives]
Applies Gaussian mixture model to form clusters. Visualizes clusters in two
dimensions. Creates simple_diabetes_clusters.png
    ORIGINAL: sklearn diabetes data set
    
## simple_em_abalone.py [probably don't use--See below for alternatives]
Same as above, but uses (better) abalone data set. Creates simple_abalone_clusters.png
    ORIGINAL: http://archive.ics.uci.edu/ml/datasets/Abalone -> abalone.data

## pairwise_em_abalone.py
Aligns well with the beginning of the R example. Uses abalone data set, shows pairwise 
clusters for real clusters and predicted clusters (fixed components to 3). 
Creates pairwise_scatter_real_clusters.png, pairwise_scatter_pred_clusters.png
    ORIGINAL: http://archive.ics.uci.edu/ml/datasets/Abalone -> abalone.data

## BIC_selection.py
Demonstrates how one might choose a model based off of BIC scores while varying
number of components and covariance type. Aligns well with R example. Creates
BIC_plot.png
    ORIGINAL: http://archive.ics.uci.edu/ml/datasets/Abalone -> abalone.data

## density_broken.py [not functional]
Started trying to recreate density plots like in the R example. Unsuccessful.