import numpy as np
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score

start_time = time.time()

# Imlementing function for the Minkowski distance
def minkowski_distance(x1,x2,p):
    
    distance = np.abs(x1 - x2)
    distance = np.power(distance,p)
    distance = np.sum(distance)
    distance = distance**(1/p)
    
    return distance

# Compute distance matrix based on the minkowski distance
def distance_matrix(data,n,p):
    col_limit = 1 # used to avoid redundant computations
    matrix = np.zeros((n,n))
    i = 0
    for i in range(n):
        for j in range(col_limit):
            if j < i:
                matrix[i,j] = minkowski_distance(data[i],data[j],p)
                matrix[j,i] = matrix[i,j]
            if i == j:
                col_limit+=1
    return matrix
        
# 2-approximative kmeans algorithm
def kmeans(k,num_points,data,p,dist_matrix):
    if k >= num_points:
        C = list(range(0,num_points))
        C = np.array(C)
    else:
        # choose initial random center
        center = np.random.choice(data.shape[0])
        C = np.array([center])
        while C.size < k:
            new_center = center
            distance = []
            for i in range(data.shape[0]):
                distance_i_to_center = []
                for c in C:
                    distance_i_to_center.append(dist_matrix[c,i])
                distance.append([np.max(distance_i_to_center)])
            new_center = np.argmax(distance)
            # includes farthest point from the centers as center
            C = np.append(C,new_center)
    
    return C

# Find the radius for a solution of the algorithm
def solution_radius(C,data):
    radius = 0
    for i in range(data.shape[0]):
        dist_to_centers = []
        for center in C:
            dist_to_centers.append(dist_matrix[center,i])
        if(min(dist_to_centers) > radius):
            radius = min(dist_to_centers)
    
    return radius

# Indetify the label of each instance
def labels(data,C):
    labels = np.zeros(data_matrix.shape[0])
    for i, point in enumerate(data_matrix):
        distances = []
        for center in C:
            distances.append(minkowski_distance(point, data_matrix[center], 2))
        labels[i] = np.argmin(distances)
    return labels

# Load the CSV file into a NumPy array
csv_file = 'TCGA_InfoWithGrade.csv'
data_matrix = np.genfromtxt(csv_file, delimiter=',')
data_matrix = np.delete(data_matrix,0,axis=0)

# Get the number of instances
num_rows = data_matrix.shape[0]

#Extracting Ground Truth label
ground_truth_labels = data_matrix[:,0].astype(int)
data_matrix = np.delete(data_matrix,0, axis = 1)

#distance matrix
dist_matrix = distance_matrix(data_matrix,num_rows,2)

for i in range(30):
    #Compute metrics
    C = kmeans(2,num_rows,data_matrix,2,dist_matrix)
    r = solution_radius(C,data_matrix)
    pointslabels = labels(data_matrix,C)
    silhouette = silhouette_score(dist_matrix, pointslabels,metric='precomputed')
    ARI = adjusted_rand_score(ground_truth_labels, pointslabels)
    runtime = time.time() - start_time
    start_time = time.time()

    #Print metrics
    print("------------teste",i,"------------")
    print('radius: ',r)
    print("Silhouette score:", silhouette)
    print("ARI:",ARI)
    print("--- %s seconds ---" % runtime)
    print('------------------------------------\n\n')