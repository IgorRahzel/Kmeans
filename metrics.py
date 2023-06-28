import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from distance import minkowski_metric

class metrics:
    ground_truth_label = None
    labels = None
    radius = None
    runtime = None
    silhouette = None
    ARI = None

    def __init__(self) -> None:
        pass

    #Extracting Ground Truth label
    def get_ground_truth(self,data,column):
        self.ground_truth_label = data[:,column].astype(int)
        data = np.delete(data,0, axis = 1)
        return data

    # Find the radius for a solution of the algorithm
    def solution_radius(self,distance_matrix,data,C):
        radius = 0
        num_points = data.shape[0]
        for i in range(num_points):
            dist_to_centers = []
            for center in C:
                dist_to_centers.append(distance_matrix[center,i])
            if(min(dist_to_centers) > radius):
                radius = min(dist_to_centers)
        
        self.radius = radius

    # Indetify the label of each instance
    def compute_labels(self,data,C,p):
        m = minkowski_metric(p)
        labels = np.zeros(data.shape[0])
        for i, point in enumerate(data):
            distances = []
            for center in C:
                distances.append(m.minkowski_distance(point, data[center]))
            labels[i] = np.argmin(distances)
        self.labels = labels
    
    def compute_silhouette(self,dist_matrix):
        self.silhouette = silhouette_score(dist_matrix,self.labels,metric='precomputed')

    def compute_ARI(self):
        self.ARI = adjusted_rand_score(self.ground_truth_label, self.labels)

    def print_metrics(self):
        print('radius: ',self.radius)
        print("Silhouette score:", self.silhouette)
        print("ARI:",self.ARI)
