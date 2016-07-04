from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics

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
                                 Perform k-means  with k=2
*********************************************************************************************************************
'''

kmeans2 = cluster.KMeans(n_clusters = 2, n_init = 20)
kmeans2.fit(X)

print('cluster centers: {}\n'.format(kmeans2.cluster_centers_))
print('cluster membership: {}\n'.format(kmeans2.labels_))
print('within cluster sum of squares: {}\n'.format(kmeans2.inertia_))

cluster_1 = X[kmeans2.labels_ == 0]
cluster_2 = X[kmeans2.labels_ == 1]

plt.figure()
plt.title('k means clustering, k = 2')
plt.plot(cluster_1[:,0], cluster_1[:,1], 'bo')
plt.plot(cluster_2[:,0], cluster_2[:,1], 'gv')
plt.show()


print '''
*********************************************************************************************************************
                                 Perform k-means  with k=3
*********************************************************************************************************************
'''

kmeans3 = cluster.KMeans(n_clusters = 3, n_init = 20)
kmeans3.fit(X)

print('cluster centers: {}\n'.format(kmeans3.cluster_centers_))
print('cluster membership: {}\n'.format(kmeans3.labels_))
print('within cluster sum of squares: {}\n'.format(kmeans3.inertia_))

cluster_1 = X[kmeans3.labels_ == 0]
cluster_2 = X[kmeans3.labels_ == 1]
cluster_3 = X[kmeans3.labels_ == 2]

plt.figure()
plt.title('k means clustering, k = 3')
plt.plot(cluster_1[:,0], cluster_1[:,1], 'bo')
plt.plot(cluster_2[:,0], cluster_2[:,1], 'gv')
plt.plot(cluster_3[:,0], cluster_3[:,1], 'rs')
plt.show()


print '''
*********************************************************************************************************************
                                 Evaluate
*********************************************************************************************************************
'''

# Silhouette score
# score = (b - a) / max(a,b)
#    a: The mean distance between a sample and all other points in the same class.
#    b: The mean distance between a sample and all other points in the next nearest cluster.

print('Silhouette score, k = 2: {}\n'.format(metrics.silhouette_score(X, kmeans2.labels_)))
print('Silhouette score, k = 3: {}\n'.format(metrics.silhouette_score(X, kmeans3.labels_)))


      
