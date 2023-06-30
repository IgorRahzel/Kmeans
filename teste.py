import numpy as np

def contar_numeros_diferentes(matriz):
    num_colunas = matriz.shape[1]
    numeros_diferentes = []

    for coluna in range(num_colunas):
        numeros_coluna = np.unique(matriz[:, coluna])
        numeros_diferentes.append(len(numeros_coluna))

    return numeros_diferentes

# Exemplo de uso
matriz = np.genfromtxt('winequality-white.csv',delimiter=';')

resultado = contar_numeros_diferentes(matriz)
print(resultado)