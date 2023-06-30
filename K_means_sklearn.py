from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances,adjusted_rand_score
import numpy as np
import time
import argparse

start_time = time.time()

parser = argparse.ArgumentParser(description='parameters for the kmeans algorithm')
parser.add_argument('-f','--file',type = str,help='file containing dataset')
parser.add_argument('-k','--num_clusters',type=int,help = 'number of clusters')
parser.add_argument('-p','--minkowski_order',type = int,help = 'minkowski order')
parser.add_argument('-c','--column',type = int, help = 'column of target atribute')
parser.add_argument('-d','--delimiter',type = str,help = 'character used to separete data')
parser.add_argument('-sh','--skip_header',type = int,help = 'skips headear from data')
args = parser.parse_args()

file = args.file
# Load the CSV file into a NumPy array
if args.delimiter == ' ':
    data_matrix = np.genfromtxt(file,skip_header=args.skip_header)
else:
    data_matrix = np.genfromtxt(file,delimiter=args.delimiter,skip_header=args.skip_header)


data_matrix = np.delete(data_matrix,0,axis=0)

#Extracting Ground Truth label
ground_truth_labels = data_matrix[:,args.column].astype(int)
data_matrix = np.delete(data_matrix,args.column, axis = 1)

# Create KMeans object with 2 clusters
kmeans = KMeans(n_clusters=args.num_clusters,n_init='auto')

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

# Compute the distances between each example and its centroid using Minkowski distance
distances = pairwise_distances(data_matrix, centroids, metric='minkowski', p=args.minkowski_order)


# Compute the silhouette score
silhouette_avg = silhouette_score(data_matrix, labels)
# Compute Adjusted Rand Index
ARI = adjusted_rand_score(ground_truth_labels,labels)
# Compute the solution radius (maximum distance to centroid)
radius = np.max(np.min(distances, axis=1))

print("Solution radius:", radius)
print("Silhouette score:", silhouette_avg)
print("ARI:",ARI)
print("--- %s seconds ---" % (time.time() - start_time))
   
