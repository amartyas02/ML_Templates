import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/Hierarchical-Clustering/Hierarchical_Clustering/Mall_Customers.csv')

X = data.iloc[:, 2:4].values

'''
Agglomerate Hierarchial Clustering:
    1. Each point-cluster(N clusters)
    2. Take 2 closest points as 1 cluster.
    3. Take 2 closest clusters as 1 cluster.
    4. Repeat 3 till 1 cluster is left.
    5. Draw the dendogram .
        i) For optimal clusters, choose the longest line which doesn't cut any horizontal line.
        ii) Draw horizontal line through it. No. of clusters = No. of cuts.
'''

import scipy.cluster.hierarchy as sch
'''Minimizing the variance within each cluster by the 'ward' method.''' 

dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)



'''Visualizing the clusters'''

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c = 'Red', label ='Cluster1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c = 'Blue', label ='Cluster2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c = 'Green', label ='Cluster3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c = 'Cyan', label ='Cluster4')

plt.legend()
plt.show()