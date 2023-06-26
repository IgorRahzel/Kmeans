from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances,adjusted_rand_score
from K_means import generate_labels
import numpy as np
import time

start_time = time.time()

csv_file = 'TCGA_InfoWithGrade.csv'
# Load the CSV file into a NumPy array
data_matrix = np.genfromtxt(csv_file, delimiter=',')
data_matrix = np.delete(data_matrix,0,axis=0)

# Create KMeans object with 2 clusters
kmeans = KMeans(n_clusters=2,n_init='auto')

# Run K-Means algorithm
kmeans.fit(data_matrix)

# Get the cluster centroids
centroids = kmeans.cluster_centers_
#print("Centroids:")
#print(centroids)

# Get the cluster labels for each example
labels = kmeans.labels_
#print("Labels:")
#print(labels)

# Compute the silhouette score
silhouette_avg = silhouette_score(data_matrix, labels)
print("Silhouette score:", silhouette_avg)

#Compute ARI:
rng = np.random.RandomState(0)
ground_truth_labels = generate_labels(data_matrix.shape[0],3,rng)
ARI = adjusted_rand_score(ground_truth_labels,labels)
print("ARI:",ARI)

# Compute the distances between each example and its centroid using Minkowski distance
distances = pairwise_distances(data_matrix, centroids, metric='minkowski', p=2)

# Compute the solution radius (maximum distance to centroid)
radius = np.max(np.min(distances, axis=1))
print("Solution radius:", radius)
print("--- %s seconds ---" % (time.time() - start_time))
