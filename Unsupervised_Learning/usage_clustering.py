import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

CURRENT_DIR = os.path.dirname(__file__)
path_file = os.path.join(CURRENT_DIR, '../data/seeds.csv')

seed_data =  pd.read_csv(path_file)
features = seed_data[seed_data.columns[0:6]]
features.sample(10)


scaled_features =  MinMaxScaler().fit_transform(features[seed_data.columns[0:6]])
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
print(features_2d[0:10])

import matplotlib.pyplot as plt


plt.scatter(features_2d[:,0],features_2d[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show()


import numpy as np
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):

    kmean = KMeans(n_clusters=i)
    # Fit the data points
    kmean.fit(features.values)
    # Get the WCSS (inertia) value
    wcss.append(kmean.inertia_)

#Plot the WCSS values onto a line graph
plt.plot(range(1, 11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# from above for we undrestand K =3 is better than the others
# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
km_clusters



def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)

# Hierarchical
'''
Hierarchical clustering creates clusters by either a divisive method or 
agglomerative method. The divisive method is a "top down" approach starting 
with the entire dataset and then finding partitions in a stepwise manner. 
Agglomerative clustering is a "bottom up** approach. In this lab you will 
work with agglomerative clustering which roughly works as follows:

The linkage distances between each of the data points is computed.
Points are clustered pairwise with their nearest neighbor.
Linkage distances between the clusters are computed.
Clusters are combined pairwise into larger clusters.
Steps 3 and 4 are repeated until all data points are in a single cluster.
The linkage function can be computed in a number of ways:

Ward linkage measures the increase in variance for the clusters being linked,
Average linkage uses the mean pairwise distance between the members of the two clusters,
Complete or Maximal linkage uses the maximum distance between the members of the two clusters.
Several different distance metrics are used to compute linkage functions:

Euclidian or l2 distance is the most widely used. This metric is only choice for the Ward linkage method.
Manhattan or l1 distance is robust to outliers and has other interesting properties.
Cosine similarity, is the dot product between the location vectors divided by the magnitudes of 
the vectors. Notice that this metric is a measure of similarity, whereas the other two metrics 
are measures of difference. Similarity can be quite useful when working with data such as images or text documents.

'''

from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
agg_clusters

import matplotlib.pyplot as plt

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()
