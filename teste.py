from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

# Dados de exemplo
X = np.array([[1, 2,0], [1, 4,0], [1, 0,1], [4, 2,0], [4, 4,0], [4, 0,1]])

labels_true = X[:,-1]

# Parâmetros do K-Means
n_clusters = 2
n_iterations = 5

# Listas para armazenar os resultados
silhouette_scores = []
ari_scores = []

for _ in range(n_iterations):
    # Criação do objeto KMeans
    kmeans = KMeans(n_clusters=n_clusters,n_init="auto")

    # Executa o algoritmo K-Means
    kmeans.fit(X)

    # Obtém as etiquetas dos clusters para cada exemplo
    labels = kmeans.labels_

    # Calcula o coeficiente de silhueta
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

    # Calcula o ARI
    ari = adjusted_rand_score(labels_true, labels)
    print(ari)
    ari_scores.append(ari)

# Calcula a média e o desvio padrão dos resultados
silhouette_mean = np.mean(silhouette_scores)
silhouette_std = np.std(silhouette_scores)

ari_mean = np.mean(ari_scores)
ari_std = np.std(ari_scores)

# Imprime os resultados
print("Coeficiente de Silhueta - Média:", silhouette_mean)
print("Coeficiente de Silhueta - Desvio Padrão:", silhouette_std)

print("ARI - Média:", ari_mean)
print("ARI - Desvio Padrão:", ari_std)