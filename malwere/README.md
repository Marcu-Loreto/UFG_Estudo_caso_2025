# Um Estudo de Caso

## Malware e Segurança Digital: Análise Comparativa com o Dataset TUNADROMD

**Fonte:** [UCI Machine Learning Repository – TUNADROMD Dataset](https://archive.ics.uci.edu/dataset/813/tunadromd)
**Autores:** Abhishek Singh e Pankaj Kumar

---

### 🎯 Objetivo do Estudo

Este projeto realiza uma **análise exploratória e preditiva** do dataset *TUNADROMD*, com foco na detecção de malwares por meio de modelos de Machine Learning. A abordagem inclui:

* Tratamento e balanceamento da base de dados.
* Treinamento e comparação de **três modelos principais**.
* Aplicação de **PCA (Principal Component Analysis)** para redução de dimensionalidade e avaliação do *trade-off* entre desempenho e custo computacional.

---

### 🧩 Etapas do Estudo

#### 1. Importação e Preparação do Ambiente

Foram importadas bibliotecas de análise de dados (*pandas, numpy*), machine learning (*scikit-learn*), estatística (*scipy*) e visualização (*matplotlib, seaborn, plotly*).
Essas bibliotecas permitiram estruturar o fluxo de análise e facilitar a comparação gráfica entre os resultados dos modelos.

#### 2. Leitura e Análise Inicial do Dataset

O dataset **TUNADROMD.csv** foi carregado e inspecionado para entender:

* Quantidade de registros e atributos (`tabela.shape`).
* Distribuição das classes (coluna `Label`).
* Existência de dados ausentes (`dropna()` foi aplicado para limpeza inicial).

Foi identificado **desbalanceamento de classes**, o que motivou o uso de técnicas de **oversampling** para obter uma proporção próxima de **50/50** entre amostras benignas e maliciosas.

#### 3. Balanceamento da Base

A base foi equilibrada com replicação controlada da classe minoritária (`sample(replace=True)`), resultando em uma distribuição balanceada:

```python
classe_0_oversampled = classe_0.sample(len(classe_1), replace=True, random_state=42)
tabela_balanceada = pd.concat([classe_1, classe_0_oversampled])
```

Isso garantiu uma amostra robusta para o treino, reduzindo o risco de *viés de predição*.

#### 4. Divisão de Dados

A base balanceada foi dividida em **80% treino** e **20% teste**:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

#### 5. Treinamento dos Modelos

Três modelos supervisionados foram treinados e comparados:

| Modelo                  | Tipo                | Finalidade                                      |
| :---------------------- | :------------------ | :---------------------------------------------- |
| **Regressão Logística** | Linear              | Modelo base e interpretável                     |
| **Random Forest**       | Ensemble            | Modelo robusto com múltiplas árvores            |
| **Gradient Boosting**   | Ensemble sequencial | Modelo otimizado para minimizar erros residuais |

Cada modelo foi avaliado com base em **Acurácia, Precisão, Recall, F1-Score, MSE e AUC-ROC**.

#### 6. Resultados Sem PCA

Os três modelos apresentaram alta performance, sendo o **Random Forest** o mais consistente nas métricas principais.

**Comparação geral (sem PCA):**

| Modelo              | Accuracy | Precision | Recall | F1-Score |   MSE  |
| :------------------ | :------: | :-------: | :----: | :------: | :----: |
| Regressão Logística |  0.9843  |   0.9944  | 0.9861 |  0.9902  | 0.0157 |
| Random Forest       |  0.9944  |   0.9958  | 0.9972 |  0.9965  | 0.0056 |
| Gradient Boosting   |  0.9866  |   0.9876  | 0.9958 |  0.9917  | 0.0134 |

O **Random Forest** apresentou o menor erro quadrático médio e maior *recall*, tornando-se o modelo mais promissor para o cenário original.

---

### 🔍 Aplicação de PCA e Análise de Trade-offs

Após a definição do melhor modelo, aplicou-se o **PCA (Principal Component Analysis)** sobre o dataset balanceado, mantendo **95% da variância explicada**. O número de componentes foi reduzido de forma significativa, diminuindo o custo computacional.

**Resultados da Regressão Logística com PCA:**

| Métrica  |  Valor |
| :------- | :----: |
| Acurácia | 0.9520 |
| Precisão | 0.9483 |
| Recall   | 0.9586 |
| F1-Score | 0.9534 |
| MSE      | 0.0480 |
| AUC-ROC  | 0.9712 |

**Redução dimensional:**

* Atributos originais: 242
* Atributos após PCA: 58
* **Redução de ~76%**, mantendo 95% da variância dos dados.

**Conclusão da Etapa PCA:**
O PCA reduziu drasticamente o custo computacional e manteve boa performance — um excelente *trade-off* entre eficiência e precisão.

---

### 📈 Visualizações e Análises Complementares

Foram gerados gráficos interativos com **Plotly** para:

* Matrizes de confusão (valores absolutos e normalizados) dos quatro cenários principais:
  *Regressão Logística, Random Forest, Gradient Boosting e Regressão Logística + PCA.*
* Comparação de métricas entre todos os modelos (gráfico de barras agrupadas).

Essas visualizações destacaram que o **Random Forest** continua com melhor desempenho geral, enquanto o uso de **PCA** proporcionou um modelo mais leve e rápido para execuções contínuas.

---

### ✅ Conclusões Gerais

1. O tratamento e balanceamento da base eliminaram viés de classificação.
2. O **Random Forest** se destacou como melhor modelo em precisão e estabilidade.
3. A aplicação de **PCA** mostrou ser eficaz para reduzir dimensionalidade com baixo impacto na performance.
4. O custo computacional foi reduzido em mais de 70%, mantendo **95% da variância** e alta acurácia.
5. Este pipeline pode ser replicado para outros datasets de cibersegurança com ajustes mínimos.

---

📘 **Resumo Final:**
Este estudo evidencia que a combinação de **pré-processamento de dados**, **avaliação de múltiplos modelos** e **redução de dimensionalidade com PCA** é uma estratégia eficiente para aplicações reais de detecção de malware — unindo **precisão preditiva** e **otimização de recursos computacionais**.
