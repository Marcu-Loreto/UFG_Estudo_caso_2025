
# Um estudo de caso 

## Malware e Segurança Digital: Um Estudo a partir do Dataset TUNADROMD

TUNADROMD Dataset. UCI Machine Learning Repository. Disponível em: https://archive.ics.uci.edu/dataset/813/tunadromd.
 Autores: Abhishek Singh e Pankaj Kumar.

 Passos:
 

Ler e entender a origem e o objetivo do dataset em sua origem 
 Carregar o Dataset 
 Analisar o dataset
       Resolver os dados faltantes ( foram deletados)
       Resolver o desbalanceamneto na feature target ( Gerado dados complementares)

 Treinar 3 modelos de regressao logistica e comparar os resultado dos 3 
 escolher o melhor conforme as metricas:
        Acurácia (Accuracy): porcentagem de previsões corretas sobre o total de exemplos. É simples e intuitiva, mas pode ser enganosa em conjuntos de dados desbalanceados.

        Precisão (Precision): fração de verdadeiros positivos entre todos os exemplos classificados como positivos. Mede a qualidade das classificações positivas.

        Recall (Sensibilidade): fração de verdadeiros positivos entre todos os exemplos que são realmente positivos. Mede a capacidade do modelo de capturar os positivos.

        F1-Score: média harmônica entre precisão e recall, útil para dados desbalanceados, equilibrando os dois aspectos.

        Área sob a curva ROC (AUC-ROC): mede a capacidade do modelo em distinguir entre classes positivas e negativas, independente do limiar escolhido.

        Matriz de confusão: tabela que apresenta verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos, útil para análise detalhada do desempenho.
 Apos escolher o melhor  metodo por comparação absoluta dos indicadores

 Foi Criado um novo dataset reduzido, usando  o PCA e comparando os resultados ( metricas) com o Metodo escolhido

       O Dataset "otimizado" trouxe os seguintes beneficios:
              Reduçao do numero absoluto de features de 241 para 66 ( representativas) - Reduçao de 72,6% no custo computacional
                                          "A aplicação de PCA no dataset TUNADROMD de
                            monstrou que a redução de dimensionalidade pode ser rea
                            lizada de forma eficiente, mantendo aproximadamente 95%
                            da variância explicada e diminuindo o espaço de atributos
                            em cerca de 64%. Essa simplificação do conjunto de dados
                            trouxe ganhos claros em termos de custo computacional
                            e compacidade do modelo, facilitando o treinamento e a
                            inferência." ( Trecho do relatório final)

