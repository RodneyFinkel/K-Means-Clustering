from k_means import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


X, y = make_blobs(n_samples=100, centers=5, random_state=10)  
X = StandardScaler().fit_transform(X)

def plot_data(X):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], edgecolor='k', legend=False)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

plot_data(X)   
# plt.show()

kmeans = KMeans(n_clusters=5, random_seed=1)
kmeans.fit(X)

def plot_clusters(X, labels, centroids):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='tab10', edgecolor='k', legend=False)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker='x', color='k', s=100, legend=False)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
    
plot_clusters(X, kmeans.labels, kmeans.centroids)


