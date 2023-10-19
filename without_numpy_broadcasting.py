import numpy as np

# calculating the squared euclidean distances without numpy broadcasting: line 35 of k_means.py


X = np.array([[x1, y1],
              [x2, y2],
              [x3, y3]])

# Cluster centroid
centroid = np.array([cx, cy])

# Initialize an array to store squared distances
squared_distances = np.zeros(X.shape[0])

# Loop through data points
for i in range(X.shape[0]):
    # Calculate the squared Euclidean distance for each feature (x and y)
    squared_dist_x = (X[i, 0] - centroid[0]) ** 2
    squared_dist_y = (X[i, 1] - centroid[1]) ** 2
    
    # Sum the squared distances for both features
    total_squared_dist = squared_dist_x + squared_dist_y
    
    # Store the result in the array
    squared_distances[i] = total_squared_dist