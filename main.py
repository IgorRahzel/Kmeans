import sys
import time
from K_means import Kmeans_aprox
from metrics import metrics

#esses dados serã recebidos pelo parser a ser implementado
k = 2
p = 2
column = 0

start_time = time.time()
kmeans = Kmeans_aprox(k=k,p=p)
csv_file = 'TCGA_InfoWithGrade.csv'
kmeans.get_data(csv_file)
Metrics = metrics()
kmeans.data = Metrics.get_ground_truth(kmeans.data,column)
kmeans.get_distance_matrix()

for i in range(30):
    kmeans.kmeans()
    Metrics.compute_labels(kmeans.data,kmeans.C,kmeans.p)
    Metrics.solution_radius(kmeans.distance_matrix,kmeans.data,kmeans.C)
    Metrics.compute_silhouette(kmeans.distance_matrix)
    Metrics.compute_ARI()

    runtime = time.time() - start_time
    start_time = time.time()

    print("------------teste",i,"------------")
    Metrics.print_metrics()
    print('runtime: ',runtime)
    print('------------------------------------\n\n')

