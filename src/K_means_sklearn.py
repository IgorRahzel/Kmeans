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
# Load the file into a NumPy array
if args.delimiter == ' ':
    data_matrix = np.genfromtxt(file,skip_header=args.skip_header)
else:
    data_matrix = np.genfromtxt(file,delimiter=args.delimiter,skip_header=args.skip_header)


data_matrix = np.delete(data_matrix,0,axis=0)

#Extracting Ground Truth label
ground_truth_labels = data_matrix[:,args.column].astype(int)
data_matrix = np.delete(data_matrix,args.column, axis = 1)

ari = np.zeros(30)
silhouette = np.zeros(30)
r = np.zeros(30)
exec_time = np.zeros(30)

for i in range(30):
    # Create KMeans object with 2 clusters
    kmeans = KMeans(n_clusters=args.num_clusters,n_init='auto')

    # Run K-Means algorithm
    kmeans.fit(data_matrix)

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_
   

    # Get the cluster labels for each example
    labels = kmeans.labels_
   
    # Compute the distances between each example and its centroid using Minkowski distance
    distances = pairwise_distances(data_matrix, centroids, metric='minkowski', p=args.minkowski_order)


    # Compute the silhouette score
    silhouette_avg = silhouette_score(data_matrix, labels)
    # Compute Adjusted Rand Index
    ARI = adjusted_rand_score(ground_truth_labels,labels)
    # Compute the solution radius (maximum distance to centroid)
    radius = np.max(np.min(distances, axis=1))
    runtime = time.time() - start_time
    start_time = time.time()

    #store metrics
    ari[i] = ARI
    silhouette[i] = silhouette_avg
    r[i] = radius
    exec_time[i] = runtime

#Show results
print("radius std:", r.std())
print("radius mean:", r.mean())

print("Silhouette score std:", silhouette.std())
print("Silhouette score mean:", silhouette.mean())

print("ARI std:", ari.std())
print("ARI mean:", ari.mean())

print("runtime std",exec_time.std())
print("runtime mean",exec_time.mean())


