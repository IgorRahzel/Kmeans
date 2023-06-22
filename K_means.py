import numpy as np


def minkowski_distance(x1,x2,p):
    
    distance = np.abs(x1 - x2)
    distance = np.power(distance,p)
    distance = np.sum(distance)
    distance = distance**(1/p)
    
    return distance

def distance_matrix(data,n,p):
    col_limit = 1 #used to avoid redundant computations
    matrix = np.zeros((n,n))
    i = 0
    for i in range(n):
        for j in range(col_limit):
            if j < i:
                matrix[i,j] = minkowski_distance(data[i],data[j],p)
            if i == j:
                col_limit+=1
    return matrix
        

def kmeans(k,num_points,data,p):
    if k > num_points:
        return num_points #||mudar depois para retornar o conjunto de pontos||
    else:
        #choose initial random center
        center = np.random.choice(data.shape[0])
        C = np.array([center])
        while C.size < k:
            max_dist = 0
            new_center = center
            for i in range(data.shape[0]):
                sum_dist = 0
                for c in C:
                    if i < c:
                        dist = dist_matrix[c,i]
                    else:
                        dist = dist_matrix[i,c]
                    sum_dist += dist
                avg_dist = sum_dist/C.size
                if avg_dist > max_dist:
                    max_dist = avg_dist
                    new_center = i
            #includes point of max average distance from the other centers as a center
            C = np.append(C,new_center)
    
    return C

def solution_radius(C,data):

    radius = 0
    for i in range(data.shape[0]):
        dist_to_centers = []
        for center in C:
            if i < center:
                dist_to_centers.append(dist_matrix[center,i])
            else:
                dist_to_centers.append(dist_matrix[i,center])
        if(min(dist_to_centers) > radius):
            radius = min(dist_to_centers)
    
    return radius





#Simulando leitura dos dados uma vez que o link não funciona
with open("teste.txt",'r') as file:

    lines = file.readlines()

    data_matrix = []

    for line in lines:
        coordinates = line.split()
        coordinates = [float(val) for val in coordinates]
        data_matrix.append(coordinates)
    
    data_matrix = np.array(data_matrix)
        
    dist_matrix = distance_matrix(data_matrix,5,2)

    C = kmeans(3,5,data_matrix,2)

    r = solution_radius(C,data_matrix)

    print(r)

    print(dist_matrix)
