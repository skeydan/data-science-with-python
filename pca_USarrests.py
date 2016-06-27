from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition

print '''
*********************************************************************************************************************
                                 Load dataset
*********************************************************************************************************************
'''

arrests_df = pd.read_csv('data/USArrests.csv', index_col=0)


print '''
*********************************************************************************************************************
                                 Explore data
*********************************************************************************************************************
'''

print('\nUSArrests dataset:')
print(arrests_df.head())
print('Columns: {}\n'.format(arrests_df.columns))
print('Index: {}\n'.format(arrests_df.index))
print('Statistics:')
print(arrests_df.describe())
print('\nCorrelation matrix:')
print(arrests_df.corr())


print '''
*********************************************************************************************************************
                                 Standardize
*********************************************************************************************************************
'''

scaler = preprocessing.StandardScaler().fit(arrests_df)
arrests = scaler.transform(arrests_df)



print '''
*********************************************************************************************************************
                                 Perform PCA
*********************************************************************************************************************
'''

pca = decomposition.PCA()
pca.fit(arrests)

print('Explained variance: {}\n'.format(pca.explained_variance_))
print('Explained variance ratio: {}\n'.format(pca.explained_variance_ratio_))

loadings = pca.components_
loadings = -loadings
print('Component loadings:\n{}\n'.format(loadings))   

comp1 = loadings[0]
comp2 = loadings[1]
 
arrests_projected = pca.transform(arrests)
arrests_projected = -arrests_projected


print '''
*********************************************************************************************************************
                                 biplot
*********************************************************************************************************************
'''


fig , ax1 = plt.subplots(figsize=(9,7))

ax1.set_xlim(-3.5,3.5)
ax1.set_ylim(-3.5,3.5)

arrests_projected_df = pd.DataFrame(arrests_projected, index = arrests_df.index)
pca_loadings_df = pd.DataFrame(loadings.T, index=arrests_df.columns)

for i in arrests_projected_df.index:
    ax1.annotate(i, (arrests_projected_df.ix[i,0], arrests_projected_df.ix[i,1]), ha='center', color='blue', fontsize=10)
    
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
    
ax2 = ax1.twinx().twiny() 
ax2.set_ylim(-1,1)
ax2.set_xlim(-1,1)
ax2.tick_params(axis='y')

a = 1.08  
for i in pca_loadings_df.index:
    ax2.annotate(i, (pca_loadings_df.ix[i,0]*a, pca_loadings_df.ix[i,1]*a), color='orange')

for i in range(0,4):
  ax2.arrow(0, 0, pca_loadings_df.iloc[i,0], pca_loadings_df.iloc[i,1], color='orange')

plt.show()
