import numpy as np
from distance import minkowski_metric
from metrics import metrics

class Kmeans_aprox:
    
    data = None
    k = None
    p = None
    distance_matrix = None
    C = None

    def __init__(self,k,p):
        self.k = k
        self.p = p
    
    def get_data(self,file):
        self.data = np.genfromtxt(file, delimiter=',')
        self.data = np.delete(self.data,0,axis=0)

    def get_distance_matrix(self):
        metric = minkowski_metric(self.p)
        self.distance_matrix = metric.compute_distance_matrix(self.data)
    
        
    # 2-approximative kmeans algorithm
    def kmeans(self):
        num_points = self.data.shape[0]
        if self.k >= num_points:
            self.C = list(range(0,num_points))
            self.C = np.array(self.C)
        else:
            # choose initial random center
            center = np.random.choice(num_points)
            self.C = np.array([center])
            while self.C.size < self.k:
                new_center = center
                distance = []
                for i in range(num_points):
                    distance_i_to_center = []
                    for c in self.C:
                        distance_i_to_center.append(self.distance_matrix[c,i])
                    distance.append([np.max(distance_i_to_center)])
                new_center = np.argmax(distance)
                # includes farthest point from the centers as center
                self.C = np.append(self.C,new_center)
        
        
    

