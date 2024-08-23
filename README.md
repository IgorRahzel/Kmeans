# Algoritmo Aproximativo Para O Problema De K-Centros

Este relatório descreve a implementação de um algoritmo 2-
aproximado para o problema de K-centros, bem como a sua avaliação através
de métricas como: raio da solução, coeficiente de silhueta, ı́ndice de Rand ajus-
tado(ARI) e tempo de execução. Essas métricas serão avaliadas para 10 con-
juntos de dados diferentes, para os quais o algoritmo realiza 30 testes para
cada um deles. Por fim o algoritmo aproximativo implementado também será
comparado ao algoritmo de K-centros disponı́vel no Scikit Learn.

## 1. Introdução

O problema em questão consiste em encontrar K centros em um conjunto de dados de
n pontos, de modo que os pontos estejam o mais próximo possı́vel de seus centros, em
outras palavras, busca-se minimizar a distâncias entre os pontos e seus respectivos centros.
Esse problema pertence à classe NP-difı́cil,ou seja, não se conhece nenhum al-
goritmo determinı́stico com tempo polinomial para o mesmo. Uma forma de abordar
problemas pertencentes à essa classe é através de algoritmos aproximativos, os quais exe-
cutam em tempo polinomial, porém retornam soluções com um fator de qualidade, o qual
é possı́vel de se estimar.
Neste relatório vamos avaliar o desempenho de um algoritmo 2-aproximado para
o problema de K-centros, isto é, as soluções desse algoritmo são no máximo duas vezes
pior que a resposta ótima. Para fazer tal avaliação foram utilizadas métricas como o raio
da solução, coeficiente de silhueta, ı́ndice de Rand ajustado(ARI) e o tempo de execução,
cada um desses parâmetros será explicado nas próximas seções do relatório.
Foi também implementado o algoritmo de K-centros disponı́vel no Scikit Learn,
para que esse fosse comparado diretamente ao algoritmo aproximativo desenvolvido, am-
bos os algoritmos foram implementados na linguagem python3.

## 2. Métodos e Métricas
Nesta seção será apresentada um descrição das métricas e métodos utilizados para a
avaliação do desempenho do algoritmos.

## 2.1. Métricas

Inicialmente e importante mencionar que a métrica de distância utilizada no algoritmo foi a distância de Minkowski, a qual é definida da seguinte forma:

$$d(x,y)=\left(\sum_{i=1}^{n}|x_{i}-y_{i}|^{p}\right)^{\frac{1}{p}}$$

onde x e y são pontos, os quais deseja-se calcular a distância entre eles e p é dito
ser a ordem da distancia de Minkowski. 
A primeira metrica/parâmetro a ser explorada é o raio da solução, ou raio máximo 
dos clusters. O raio de um cluster e definido como sendo a maior distância entre um 
ponto qualquer daquele cluster e seu centro, como mencionado anteriormente o objetivo do problema e minimizar tal distância. Consequentemente o raio da soluçaão é o maior
valor de raio obtido ao se avaliar todos os clusters.

O coeficiente de silhueta avalia o quão coesa é a solução dada pelo algoritmo, ou
seja, essa e uma medida que avalia a similaridade dos objetos designados a um mesmo cluster, os valores dessa metrica variam dentro do intervalo [-1,1], quanto mais próximo de 1 e o valor dessa métrica temos que os clusters estão mais distantes entre si e estão melhor definidos. Essa metrica é definida da seguinte forma: ´

$$\frac{b\!-\!a}{m a x(a,b)}$$

onde a e a distância média entre os pontos de um mesmo cluster é b e a distância
entre um ponto e o cluster mais proximo que não inclui esse ponto. No algoritmo aproximativo foi utilizada a implementação do coeficiente de silhueta disponível na biblioteca sklearn, essa função retorna a média do coeficiente de silhueta dentre todos os pontos do
conjunto de dados.

O Índice de Rand Ajustado, essa metrica calcula a similaridade entre dois clus- ´
ters de um mesmo conjunto de dados, o ARI e um aprimoramento do Índice de Rand(RI)
que basicamente verifica o numero de pares de pontos que foi classificado de forma igual é diferente nos dois clusters avaliados. O ARI diferentemente do RI, leva em conta a aleatoriedade esperada nos agrupamentos quando os agrupamentos tem tamanhos varíados. Essa metrica assim como o coeficiente de silhueta, assume valores no intervalo
[-1,1], onde quanto mais proximo de 1 for o valor obtido, melhor é a concordância entre
os agrupamentos.

Por fim, o ultimo parâmetro utilizado na avaliação do desempenho do algoritmo
foi o tempo de processamento de cada execução. Nesse trabalho prático foi avaliado o funcionamento do algoritmo em 10 conjuntos de dados diferentes e foram realizadas 30 execuções do algoritmo para cada um deles.

## 2.2. Métodos

O algoritmo tem um funcionamento bem simples, inicialmente é calculada a matriz de
distância entre os pontos seguindo a métrica de Minkowski com o parâmetro p desejado.
O próximo passo é a execução do algoritmo 2-aproximativo, o qual inicialmente
escolhe aleatoriamente um dos pontos do conjunto de dados como sendo um centro, então
enquanto o número de centros escolhidos pelo algoritmo for menor que k, o algoritmo irá
selecionar como um novo centro o ponto que possui a maior distância para seu centro.
O algoritmo é executado trinta vezes e a cada execução são salvos os valores das
métricas desejadas em uma matriz. Ao final são utilizados os métodos mean e std da
biblopteca numpy para obter a média e o desvio padrão de cada uma das métricas.

## 3. Implementação

O algoritmo aproximativo foi desenvolvido na linguagem python na versão 3.10.9.
Para melhorar a modularização do código foram implementadas três classes sendo elas
minkowski, Kmeans aprox, metrics.
A classe minkowski implementa os métodos relacionados a distância de
Minkowski, sendo estes a própria função de distância e a matriz de distância dos pon-
tos do conjunto de dados.
Na classe Kmeans aprox é onde está implementado o algoritmo aproximativo,
essa implementação é realizada no método kmeans. Além desse método, essa classe
também possui funções auxiliares para guardar os conjuntos de dados em uma matriz,
bem como obter a matriz de distância dos pontos que compõem esse conjunto com o
auxı́lio dos métodos implementados na classe minkowski.
Já a classe metrics é responsável pela implementação das métricas de avaliação
das soluções do algoritmo através dos métodos solution radius, compute silhouette, com-
pute ARI que retornam o raio da solução, a silhoueta e o ARI respectivamente, vale men-
cionar que os métodos compute silhouette e compute ARI, utilizam funções disponı́veis
na biblioteca sklearn para realizar o cálculo dessas métricas.
As classes descritas acima estão implementadas nos arquivos minkowski.py,
Kmeans aprox.py, metrics.py, mas é no arquivo main.py onde são realizadas as leituras
dos inputs e realizadas as chamadas dos métodos de classe para a execução devida do
algoritmo.
É nesse arquivo onde também é feito o cálculo do tempo de execução do algoritmo
através do uso da biblioteca time. É importante mencionar que o no primeiro dos trinta
testes realizados para cada um dos dez conjuntos de dados selecionados é computada a
matriz de distância, tal operação é contabilizada no tempo de execução do algoritmo.Para
a execução do algoritmo no Linux basta executar o seguinte comando:
python3 main.py -f file -k num clusters -p p val -c col -d delimitador -sh
num linhas
onde os valores os parâmetros file corresponde ao arquivo contendo o conjunto de
dados, num clusters é o número de clusters desejado, p val é valor da ordem da distância
de Minkowski, col é a coluna do conjunto de dados do atributo classe, num linhas é o
número de linhas que se deseja saltar caso o conjunto de dados possua um cabeçalho,
caso contrário esse número deve ser zero.
Já o arquivo K means sklearn.py é responsável por implementar o algoitmo de
k-centros disponı́vel na biblioteca sklearn, bem como computar as métricas da solução
retornada por esse algoritmo. Esse arquivo é executado da mesma forma que o anterior,
basta substituir main.py por K means sklearn.py no comando de execução.

## Descrição dos Experimentos

Os conjuntos de dados utilizados no teste do algoritmo foram retirados do *UCI Machine* Learning Repository,foram selecionados conjuntos de dados contendo no m´ınimo 700 instancias, e sendo eles exclusivamente num ˆ ericos. ´
Os conjuntos de dados selecionados foram:Spambase[1],Glioma Grading Clinical and Mutation Features Dataset[2], Blood Transfusion Service Center[3], Wireless Indoor Localization[4], banknote authentication[5], Diabetic Retinopathy Debrecen Data Set[6],
South German Credit (UPDATE)[7], Wine Quality(do qual foram utilizados os dois arquivos)[8][9] e Pen-Based Recognition of Handwritten Digits[10].

Para cada um dos experimentos foi utilizado um valor diferente para a ordem da distancia de Minkowski, o valor para esse par ˆ ametro foi inicialmente 1 para o experimento ˆ realizado com o primeiro conjunto de dados e a cada novo conjunto ele foi acrescido em uma unidade.

O numero de clusters escolhidos para cada um dos conjuntos de dados foi de- ´
terminado com base nas informac¸oes dispon ˜ ´ıveis em suas descric¸oes. Tais informac¸ ˜ oes ˜ tambem permitiram a extrac¸ ´ ao do valor verdade para os r ˜ otulos(ground truth label) dos ´
clusters, esses valores sao empregados no c ˜ alculo do ´ ARI. E importante mencionar que ´
nos conjuntos de dados disponibilizados em Wine Quality foi considerado um numero ´ menor de clusters do que aquele "sugerido" pela descric¸ao do mesmo, apesar dos vinhos ˜ serem agrupados de acordo com as suas notas que variam de 0 ate 10, ao analisar cada ´
um dos conjuntos de dados verificou-se que algumas das notas nao haviam sido atribu ˜ ´ıdas a nenhum dos vinhos, o que justificou um uma escolha menor em relac¸ao ao n ˜ umero de ´ agrupametos.

## Apresentação e Análise dos Resultados

Os conjuntos de dados utilizados no teste do algoritmo foram retirados do UCI Machine
Learning Repository,foram selecionados conjuntos de dados contendo no mı́nimo 700
instâncias, e sendo eles exclusivamente numéricos.Os conjuntos de dados selecionados foram:Spambase[1],Glioma Grading Clinical
and Mutation Features Dataset[2], Blood Transfusion Service Center[3], Wireless Indoor
Localization[4], banknote authentication[5], Diabetic Retinopathy Debrecen Data Set[6],
South German Credit (UPDATE)[7], Wine Quality(do qual foram utilizados os dois ar-
quivos)[8][9] e Pen-Based Recognition of Handwritten Digits[10].
Para cada um dos experimentos foi utilizado um valor diferente para a ordem da
distância de Minkowski, o valor para esse parâmetro foi inicialmente 1 para o experimento
realizado com o primeiro conjunto de dados e a cada novo conjunto ele foi acrescido em
uma unidade.
O número de clusters escolhidos para cada um dos conjuntos de dados foi de-
terminado com base nas informações disponı́veis em suas descrições. Tais informações
também permitiram a extração do valor verdade para os rótulos(ground truth label) dos
clusters, esses valores são empregados no cálculo do ARI. É importante mencionar que
nos conjuntos de dados disponibilizados em Wine Quality foi considerado um número
menor de clusters do que aquele ”sugerido” pela descrição do mesmo, apesar dos vinhos
serem agrupados de acordo com as suas notas que variam de 0 até 10, ao analisar cada
um dos conjuntos de dados verificou-se que algumas das notas não haviam sido atribuı́das
a nenhum dos vinhos, o que justificou um uma escolha menor em relação ao número de
agrupametos.

| Table 2. Media e Desvio Padrão Obtidos na Implementação do Sklearn   |         |          |         |       |         |      |         |      |    |    |
|-------------------------------------------------------------------------------|---------|----------|---------|-------|---------|------|---------|------|----|----|
| DataSet                                                                       | Raio    | silhueta | ARI     | Tempo |         |      |         |      |    |    |
| std                                                                           | media ´ | std      | media ´ | std   | media ´ | std  | media ´ | k    | p  |    |
| 1                                                                             | 2835.4  | 16620.6  | 0.03    | 0.86  | 0.01    | 0.03 | 0.15    | 0.73 | 2  | 1  |
| 2                                                                             | 0.09    | 25.67    | 0.00    | 0.58  | 0.00    | 0.20 | 0.02    | 0.05 | 2  | 2  |
| 3                                                                             | 131.1   | 7812.6   | 0.00    | 0.69  | 0.00    | 0.07 | 0.01    | 0.05 | 2  | 3  |
| 4                                                                             | 0.34    | 29.2     | 0.02    | 0.4   | 0.05    | 0.87 | 0.02    | 0.18 | 4  | 4  |
| 5                                                                             | 0.01    | 12.49    | 0.00    | 0.43  | 0.00    | 0.04 | 0.02    | 0.10 | 2  | 5  |
| 6                                                                             | 2.85    | 260.3    | 0.00    | 0.43  | 0.00    | 0.00 | 0.02    | 0.08 | 2  | 6  |
| 7                                                                             | 17.86   | 10017.6  | 0.00    | 0.72  | 0.00    | 0.05 | 0.02    | 0.07 | 2  | 7  |
| 8                                                                             | 53.1    | 96.1     | 0.01    | 0.39  | 0.00    | 0.00 | 0.02    | 0.14 | 7  | 8  |
| 9                                                                             | 28.03   | 229.8    | 0.00    | 0.30  | 0.00    | 0.01 | 0.062   | 0.61 | 8  | 9  |
| 10                                                                            | 2.09    | 86.0     | 0.01    | 0.29  | 0.03    | 0.57 | 0.17    | 1.7  | 10 | 10 |

|         | Table 1. Media e Desvio Padr ´ ao Obtidos no Algoritmo Aproximativo ˜   |         |          |         |       |         |       |         |    |    |
|---------|-------------------------------------------------------------------------|---------|----------|---------|-------|---------|-------|---------|----|----|
| DataSet |                                                                         | Raio    | silhueta | ARI     | Tempo |         |       |         |    |    |
|         | std                                                                     | media ´ | std      | media ´ | std   | media ´ | std   | media ´ | k  | p  |
| 1       | 258.8                                                                   | 15659.0 | 0.003    | 0.96    | 0.00  | 0.00    | 13.73 | 2.77    | 2  | 1  |
| 2       | 3.76                                                                    | 31.39   | 0.059    | 0.48    | 0.073 | 0.10    | 1.70  | 0.36    | 2  | 2  |
| 3       | 561.74                                                                  | 5033.3  | 0.01     | 0.83    | 0.00  | 0.02    | 0.92  | 0.20    | 2  | 3  |
| 4       | 1.70                                                                    | 24.6    | 0.07     | 0.28    | 0.13  | 0.38    | 6.49  | 1.53    | 4  | 4  |
| 5       | 1.64                                                                    | 12.79   | 0.02     | 0.47    | 0.02  | 0.06    | 3.20  | 0.66    | 2  | 5  |
| 6       | 15.65                                                                   | 174.6   | 0.02     | 0.64    | 0.00  | 0.00    | 2.4   | 0.5     | 2  | 6  |
| 7       | 653.4                                                                   | 8088.9  | 0.00     | 0.73    | 0.00  | 0.04    | 1.7   | 0.37    | 2  | 7  |
| 8       | 1.9                                                                     | 33.4    | 0.02     | 0.44    | 0.00  | 0.00    | 4.47  | 1.05    | 7  | 8  |
| 9       | 3.95                                                                    | 62.2    | 0.04     | 0.32    | 0.00  | 0.01    | 39.2  | 8.2     | 8  | 9  |
| 10      | 1.97                                                                    | 95.1    | 0.03     | 0.16    | 0.04  | 0.31    | 94.9  | 19.4    | 10 | 10 |

## 6. Conclusao˜

Este trabalho mostrou que mesmo que um determinado problema seja da classe NP, isso nao significa que ele n ˜ ao deve ser abordado pelo fato de n ˜ ao se conhecer um algoritmo ˜ polinomial determin´ıstico para o mesmo, pois atraves de algoritmos aproximativos pode- ´
mos muitas vezes obter soluc¸oes boas o suficientes para muitos casos. ˜
Embora o algoritmo implementado seja 2-aproximado, temos os resultados obtidos para ele nos casos de teste realizados foi no geral tao boa quanto a aquela obtida ˜
utilizando-se a biblioteca do *Scikit Learn*, em alguns casos o raio obtido foi ate melhor. A ´
diferenc¸a nos resultados das metricas de coefiente de silhueta e ARI, como mostrado na ´ sec¸ao anterior foi inferior a um d ˜ ecimo. O que mostra uma similaridade entre as soluc¸ ´ oes ˜ apresentadas, quando feita uma analise global. ´
A grande desvantagem do algoritmo aproximativo nao foi em relac¸ ˜ ao ao resul- ˜
tados retornados, mas sim o seu tempo de execuc¸ao que foi consideravelmente pior em ˜
alguns casos quando comparado ao outro algoritmo utilizado.

## 7. References

[Keiblerg and Tardos 2006]

## References

[Keiblerg and Tardos 2006] Keiblerg, J. and Tardos, E. (2006). Algorithm desing. In Ciencias da Computac¸ ˆ ao˜ . Pearson/Addison-Wesley.
