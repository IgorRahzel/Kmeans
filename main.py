import argparse
import time
import numpy as np
from K_means import Kmeans_aprox
from metrics import metrics

parser = argparse.ArgumentParser(description='parameters for the kmeans algorithm')
parser.add_argument('-f','--file',type = str,help='file containing dataset')
parser.add_argument('-k','--num_clusters',type=int,help = 'number of clusters')
parser.add_argument('-p','--minkowski_order',type = int,help = 'minkowski order')
parser.add_argument('-c','--column',type = int, help = 'column of target atribute')
parser.add_argument('-d','--delimiter',type = str,help = 'character used to separete data')
parser.add_argument('-sh','--skip_header',type = int,help = 'skips headear from data')
args = parser.parse_args()
#esses dados serão recebidos pelo parser a ser implementado
k = args.num_clusters
p = args.minkowski_order
column = args.column

start_time = time.time()
kmeans = Kmeans_aprox(k=k,p=p)
file = args.file
kmeans.get_data(file,args.delimiter,args.skip_header)
Metrics = metrics()
kmeans.data = Metrics.get_ground_truth(kmeans.data,column)
kmeans.get_distance_matrix()

results_matrix = np.zeros((30,4))

for i in range(30):
    kmeans.kmeans()
    Metrics.compute_labels(kmeans.data,kmeans.distance_matrix,kmeans.C)
    Metrics.solution_radius(kmeans.distance_matrix,kmeans.data,kmeans.C)
    Metrics.compute_silhouette(kmeans.distance_matrix)
    Metrics.compute_ARI()

    runtime = time.time() - start_time
    start_time = time.time()

    results_matrix[i,0] = Metrics.radius
    results_matrix[i,1] = Metrics.silhouette
    results_matrix[i,2] = Metrics.ARI
    results_matrix[i,3] = runtime


print(results_matrix)
print('------------------------------------\n\n')
print('std Radius: ', np.std(results_matrix[:,0]))
print('std silhouette: ', np.std(results_matrix[:,1]))
print('std ARI: ', np.std(results_matrix[:,2]))
print('std runtime: ', np.std(results_matrix[:,3]))

print('------------------------------------\n\n')
print('mean Radius: ', np.mean(results_matrix[:,0]))
print('mean silhouette: ', np.mean(results_matrix[:,1]))
print('mean ARI: ', np.mean(results_matrix[:,2]))
print('mean runtime: ', np.mean(results_matrix[:,3]))



