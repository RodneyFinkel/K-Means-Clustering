import numpy as np 
from sklearn.base import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, n_clusters, max_iter=100, random_seed=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)  #provides control over the randomness of the sampling process(random seed)
        self.verbose = verbose
        
    def fit(self, X):
        # Randomly select the initial centroids from the given points (with no replacement)
        # idx are the randomly chosen indices of X, n_clusters in amount, chosen without repetition, it's an array
        idx = self.random_state.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]  # X[idx] retrieves data points from X based on indices in idx. At this stage the centroid are not calculated but randomly assigned for the X dataset
        if self.verbose: print('Centroids:', self.centroids)
        
        # Initialize a distances matrix for  data points and the centroids
        # .zeros() creates a matrix filled with zeros. first parameter is a tuple with dimensions. in this case, len(X) rows and n_clusters collumns
        distances = np.zeros((len(X), self.n_clusters))
        prev_labels = None # during subsequent iterations this will be udpated to contain cluster labels from previous iterations
                           # prev_labels will have a length len(X) each element represents the cluster label assigned to the corresponding data point in previous iteration)
                           # ie prev_labels = [2, 1, 0, 2, 1, 0, 0, 2, 1, ...] n_clusters = 3

        # Run the algorithm until convergence or max_iter has been reached
        for iteration in range(self.max_iter):
            if self.verbose : print('\nIteration', iteration)
            
            # Compute distances to the cluster centroid
            # specify which column of the distance matrix we want to update with each iteration through range(self.clusters)
            # takes the squared euclidean distance of each data point (x,y) with centroid i (x2,y2) / (x-x2)**2,  (y-y2)**2
            # numpy has a feature called broadcasting which enables element wise operations without explicit loops
            # takes the sum along each row of all the squared distances calculated with (X - self.centroids[i])**2 axis=1 is along the rows, axis=0 is along collumns
            for i in range(self.n_clusters):
                distances[:, i] = np.sum((X - self.centroids[i])**2, axis=1)
                
            # Assign each data point to the cluster with the nearest centroid
            self.labels = np.argmin(distances, axis=1)   # assigns the min value of each row by taking the index of the feature (axis=1) to the self.labels array
            if self.verbose: print('labels:', self.labels) # This array of cluster labels indicates which cluster each data point belongs to
                
            # Check if there was no change in the cluster assignments
            if np.all(self.labels == prev_labels):
                break
            prev_labels == self.labels
            
            # Recompute the new centroid for each cluster i
            for i in range(self.n_clusters):
                self.centroids[i] = np.mean(X[self.labels == i], axis=0)
                
                # Handle empty clusters
                if np.isnan(self.centroids[i]).any():
                    self.centroids[i] = X[self.random_state.choice(len(X))]
                    
            if self.verbose: print('Centroids:', self.centroids)
            
        
        
