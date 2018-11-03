import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amartya/Desktop/Udemy_data/K_Means/K_Means/Mall_Customers.csv')

X = data.iloc[:, 2:4].values

'''K-Means Clustering'''


'''K-Means 1.Choose the k no.of clusters. 2.Select k-centroids(not essentially from the dataset)
3.Assign nearest points to eacg centroid(Form k-clusters) 4.Place the new centroid of each cluster.'''

'''Using elbow method to find the optimal no. of clusters'''
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init= 'k-means++', max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.xlim(1,10)
plt.show()

'''Applying k-means to the dataset'''

kmeans = KMeans(n_clusters = 4, init= 'k-means++', max_iter = 300, random_state = 0)
y_means = kmeans.fit_predict(X)

'''Visualizing the clusters'''

plt.scatter(X[y_means==0, 0], X[y_means==0, 1], s=100, c = 'Red', label ='Cluster1')
plt.scatter(X[y_means==1, 0], X[y_means==1, 1], s=100, c = 'Blue', label ='Cluster2')
plt.scatter(X[y_means==2, 0], X[y_means==2, 1], s=100, c = 'Green', label ='Cluster3')
plt.scatter(X[y_means==3, 0], X[y_means==3, 1], s=100, c = 'Cyan', label ='Cluster4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, c ='Yellow', label = 'Centroids')
plt.legend()
plt.show()


'''Hierarchial Clustering'''


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