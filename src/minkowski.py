import numpy as np

class minkowski:
    
    p = None

    def __init__(self,p):
        self.p = p

    # Imlementing function for the Minkowski distance
    def minkowski_distance(self,x1,x2):
        
        distance = np.abs(x1 - x2)
        distance = np.power(distance,self.p)
        distance = np.sum(distance)
        distance = distance**(1/self.p)
        
        return distance

    # Compute distance matrix based on the minkowski distance
    def compute_distance_matrix(self,data):
        n = data.shape[0]
        col_limit = 1 # used to avoid redundant computations
        matrix = np.zeros((n,n))
        i = 0
        for i in range(n):
            for j in range(col_limit):
                if j < i:
                    matrix[i,j] = self.minkowski_distance(data[i],data[j])
                    matrix[j,i] = matrix[i,j]
                if i == j:
                    col_limit+=1
        distance_matrix = matrix
        return distance_matrix