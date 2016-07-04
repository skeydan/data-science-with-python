from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.cluster import hierarchy

print '''
*********************************************************************************************************************
                                 Create dataset
*********************************************************************************************************************
'''

X = np.random.normal(0,1,100).reshape(50,2)
X[:26,0] = X[:26,0] + 3
X[:26,1] = X[:26,1] -4


print '''
*********************************************************************************************************************
                                 Explore data
*********************************************************************************************************************
'''

print('Generated dataset:')
print(X)


print '''
*********************************************************************************************************************
                                 scikit-learn: hierarchical clustering,  n=2
*********************************************************************************************************************
'''

hclust_model = cluster.AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
hclust_model.fit(X)
print('Cluster labels: {}\n'.format(hclust_model.labels_))

hclust_model = cluster.AgglomerativeClustering(n_clusters = 2, linkage = 'average')
hclust_model.fit(X)
print('Cluster labels: {}\n'.format(hclust_model.labels_))

hclust_model = cluster.AgglomerativeClustering(n_clusters = 2, linkage = 'complete')
hclust_model.fit(X)
print('Cluster labels: {}\n'.format(hclust_model.labels_))


print '''
*********************************************************************************************************************
                                 scipy: dendrogram
*********************************************************************************************************************
'''

# from: https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%2010.ipynb

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,18))

for linkage, cluster, ax in zip([hierarchy.complete(X), hierarchy.average(X), hierarchy.single(X)], ['c1','c2','c3'],
                                [ax1,ax2,ax3]):
    cluster = hierarchy.dendrogram(linkage, ax=ax, color_threshold=0)

ax1.set_title('Complete Linkage')
ax2.set_title('Average Linkage')
ax3.set_title('Single Linkage')

plt.show()

