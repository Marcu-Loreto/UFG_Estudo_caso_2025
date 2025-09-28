
# Um estudo de caso 

## Malware e Segurança Digital: Um Estudo a partir do Dataset TUNADROMD

TUNADROMD Dataset. UCI Machine Learning Repository. Disponível em: https://archive.ics.uci.edu/dataset/813/tunadromd.
 Autores: Abhishek Singh e Pankaj Kumar.

 Passos:
 

Ler e entender a origem e o objetivo do dataset em sua origem 
 Carregar o Dataset 
 analisar o dataset

 Treinar 3 modelos de regressao logistica e comparar os resultado dos 3 
 escolher o melhor conforme as lmetricas:
        Acurácia (Accuracy): porcentagem de previsões corretas sobre o total de exemplos. É simples e intuitiva, mas pode ser enganosa em conjuntos de dados desbalanceados.

        Precisão (Precision): fração de verdadeiros positivos entre todos os exemplos classificados como positivos. Mede a qualidade das classificações positivas.

        Recall (Sensibilidade): fração de verdadeiros positivos entre todos os exemplos que são realmente positivos. Mede a capacidade do modelo de capturar os positivos.

        F1-Score: média harmônica entre precisão e recall, útil para dados desbalanceados, equilibrando os dois aspectos.

        Área sob a curva ROC (AUC-ROC): mede a capacidade do modelo em distinguir entre classes positivas e negativas, independente do limiar escolhido.

        Matriz de confusão: tabela que apresenta verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos, útil para análise detalhada do desempenho.
 Apos escolher o melhro metodo
 Criar um novo dataset e rodat o PCA e um terceiro e rodar o SelectKBest

 Treinar o modelo escolhido na primeira rodada usando os dois novos datasets ( otimizados) e comparar os resultados

 Definir o melhor modelo a ser usado