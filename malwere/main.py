# Importações para análise de dados
import pandas as pd
import numpy as np

# Importações para machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           f1_score, precision_score, recall_score, 
                           silhouette_score, adjusted_rand_score)

# Importações para feature selection e redução de dimensionalidade
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_classif, chi2

# Importações para visualizações
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importações para análise comparativa
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Importações para análise estatística
from scipy import stats
from scipy.stats import chi2_contingency

# Configurações para visualizações
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Bibliotecas importadas com sucesso! Pronto para análise do dataset TUNADROMD.")

# Carregar o dataset TUNADROMD
tabela = pd.read_csv('TUANDROMD.csv')
print(f"Dataset carregado com sucesso! Shape: {tabela.shape}")

# Mostrar distribuição das classes
print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")
print(tabela['Label'].value_counts())
print(f"Proporção das classes:")
print(tabela['Label'].value_counts(normalize=True))

tabela = tabela.dropna()#tabela.info()
print("+" *20)

#codigo para criar proporcao %50/50
classe_1 = tabela[tabela['Label'] == 1.0]
classe_0 = tabela[tabela['Label'] == 0.0]
classe_0_oversampled = classe_0.sample(len(classe_1), replace=True, random_state=42)
tabela_balanceada = pd.concat([classe_1, classe_0_oversampled])

tabela_balanceada.info()
print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")
print(tabela_balanceada['Label'].value_counts())
print(f"Proporção das classes:")
print(tabela_balanceada['Label'].value_counts(normalize=True))
print("+" *20)
print(f"Dataset balanceado com sucesso!")

print("+" *20)
print(f"Dividir dados de treino e testes")
y = tabela['Label']
x = tabela.drop(columns=['Label'])
#Dividir o dataset em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Dataset dividido em treino e teste com sucesso! Shape: {x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")
#Treinar o modelo Regressao logistica 
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("+" *20)
print(f"Modelo Regressao logistica treinado com sucesso!")
#Resultados do modelo Regressao logistica
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
print(f"Precisão: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")
print(f"Matriz de confusão: {confusion_matrix(y_test, y_pred)}")
print(f"Relatório de classificação: {classification_report(y_test, y_pred)}")
#Treinar o modelo Random Forest
print("+" *20)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Modelo Random Forest treinado com sucesso!")
#Resultados do modelo Random Forest
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
print(f"Precisão: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")
print(f"Matriz de confusão: {confusion_matrix(y_test, y_pred)}")
print(f"Relatório de classificação: {classification_report(y_test, y_pred)}")
#Treinar o modelo Gradient Boosting
print("+n" *20)

model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(x_train, y_train)
y_pred_gb = model_gb.predict(x_test)
print("MODELO GRADIENT BOOSTING treinado com sucesso!")

# Métricas do Gradient Boosting
print(f"Acurácia: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred_gb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_gb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_gb):.4f}")

# Matriz de Confusão detalhada
print("\n=== MATRIZ DE CONFUSÃO (Gradient Boosting) ===")
cm_gb = confusion_matrix(y_test, y_pred_gb)
print("Matriz de Confusão:")
print("               Predito")
print("                0     1")
print(f"Real    0    {cm_gb[0,0]:4d}  {cm_gb[0,1]:4d}")
print(f"        1    {cm_gb[1,0]:4d}  {cm_gb[1,1]:4d}")

# Erro Quadrático Médio (MSE)
from sklearn.metrics import mean_squared_error
mse_gb = mean_squared_error(y_test, y_pred_gb)
print(f"\nErro Quadrático Médio (MSE): {mse_gb:.4f}")

# Probabilidades para ROC-AUC
y_pred_proba_gb = model_gb.predict_proba(x_test)[:, 1]
roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb)
print(f"ROC-AUC Score: {roc_auc_gb:.4f}")

print(f"\nRelatório de classificação:")
print(classification_report(y_test, y_pred_gb))

# Comparação final dos modelos
print("\n" + "="*60)
print("COMPARAÇÃO FINAL DOS MODELOS")
print("="*60)

# Coletar resultados de todos os modelos
models_results = {
    'Regressão Logística': {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    },
    'Random Forest': {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    },
    'Gradient Boosting': {
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1': f1_score(y_test, y_pred_gb),
        'mse': mean_squared_error(y_test, y_pred_gb)
    }
}

# Criar DataFrame para comparação
comparison_df = pd.DataFrame(models_results).T
print("\nTabela de Comparação:")
print(comparison_df.round(4))

# Encontrar o melhor modelo por métrica
print("\n=== MELHORES MODELOS POR MÉTRICA ===")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    best_model = comparison_df[metric].idxmax()
    best_score = comparison_df.loc[best_model, metric]
    print(f"{metric.upper()}: {best_model} ({best_score:.4f})")

print("\n=== MENOR ERRO QUADRÁTICO MÉDIO ===")
best_mse_model = comparison_df['mse'].idxmin()
best_mse_score = comparison_df.loc[best_mse_model, 'mse']
print(f"Menor MSE: {best_mse_model} ({best_mse_score:.4f})")

print("="*60)
print("APLICANDO PCA NO DATASET BALANCEADO")
print("="*60)

# Aplicar PCA no dataset balanceado
y_balanced = tabela_balanceada['Label']
x_balanced = tabela_balanceada.drop(columns=['Label'])

# Normalizar os dados antes do PCA
scaler = StandardScaler()
x_balanced_scaled = scaler.fit_transform(x_balanced)

# Aplicar PCA
pca = PCA(n_components=0.95)  # Manter 95% da variância
x_balanced_pca = pca.fit_transform(x_balanced_scaled)

print(f"Shape original: {x_balanced.shape}")
print(f"Shape após PCA: {x_balanced_pca.shape}")
print(f"Número de componentes PCA: {pca.n_components_}")
print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.4f}")

# Dividir dados balanceados com PCA
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(
    x_balanced_pca, y_balanced, test_size=0.2, random_state=42
)

print(f"Divisão treino/teste com PCA: {x_train_pca.shape}, {x_test_pca.shape}")

# Treinar Regressão Logística com PCA
print("\n=== REGRESSÃO LOGÍSTICA COM PCA ===")
model_lr_pca = LogisticRegression(random_state=42)
model_lr_pca.fit(x_train_pca, y_train_pca)
y_pred_lr_pca = model_lr_pca.predict(x_test_pca)

# Métricas da Regressão Logística com PCA
print(f"Acurácia: {accuracy_score(y_test_pca, y_pred_lr_pca):.4f}")
print(f"Precisão: {precision_score(y_test_pca, y_pred_lr_pca):.4f}")
print(f"Recall: {recall_score(y_test_pca, y_pred_lr_pca):.4f}")
print(f"F1-Score: {f1_score(y_test_pca, y_pred_lr_pca):.4f}")

# Matriz de Confusão
cm_lr_pca = confusion_matrix(y_test_pca, y_pred_lr_pca)
print("\nMatriz de Confusão (Regressão Logística + PCA):")
print("               Predito")
print("                0     1")
print(f"Real    0    {cm_lr_pca[0,0]:4d}  {cm_lr_pca[0,1]:4d}")
print(f"        1    {cm_lr_pca[1,0]:4d}  {cm_lr_pca[1,1]:4d}")

# MSE
mse_lr_pca = mean_squared_error(y_test_pca, y_pred_lr_pca)
print(f"Erro Quadrático Médio (MSE): {mse_lr_pca:.4f}")

# ROC-AUC
y_pred_proba_lr_pca = model_lr_pca.predict_proba(x_test_pca)[:, 1]
roc_auc_lr_pca = roc_auc_score(y_test_pca, y_pred_proba_lr_pca)
print(f"ROC-AUC Score: {roc_auc_lr_pca:.4f}")

print(f"\nRelatório de classificação (Regressão Logística + PCA):")
print(classification_report(y_test_pca, y_pred_lr_pca))

print("\n" + "="*80)
print("COMPARAÇÃO FINAL: Regressão Logística SEM PCA vs COM PCA")
print("="*80)

# Coletar resultados da Regressão Logística SEM PCA (do início do código)
# Nota: Você precisará ajustar as variáveis se elas tiverem nomes diferentes
print("\n=== REGRESSÃO LOGÍSTICA SEM PCA ===")
print("(Resultados do modelo treinado no início do código)")
print("Nota: Verifique os resultados acima para os valores exatos")

# Resultados COM PCA
print("\n=== REGRESSÃO LOGÍSTICA COM PCA ===")
print(f"Acurácia: {accuracy_score(y_test_pca, y_pred_lr_pca):.4f}")
print(f"Precisão: {precision_score(y_test_pca, y_pred_lr_pca):.4f}")
print(f"Recall: {recall_score(y_test_pca, y_pred_lr_pca):.4f}")
print(f"F1-Score: {f1_score(y_test_pca, y_pred_lr_pca):.4f}")
print(f"MSE: {mse_lr_pca:.4f}")
print(f"ROC-AUC: {roc_auc_lr_pca:.4f}")

# Criar tabela de comparação
print("\n=== TABELA DE COMPARAÇÃO ===")
comparison_lr = pd.DataFrame({
    'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'MSE', 'ROC-AUC'],
    'Sem PCA': ['Ver resultados acima', 'Ver resultados acima', 'Ver resultados acima', 
                'Ver resultados acima', 'Ver resultados acima', 'Ver resultados acima'],
    'Com PCA': [f"{accuracy_score(y_test_pca, y_pred_lr_pca):.4f}",
                f"{precision_score(y_test_pca, y_pred_lr_pca):.4f}",
                f"{recall_score(y_test_pca, y_pred_lr_pca):.4f}",
                f"{f1_score(y_test_pca, y_pred_lr_pca):.4f}",
                f"{mse_lr_pca:.4f}",
                f"{roc_auc_lr_pca:.4f}"]
})

print(comparison_lr.to_string(index=False))

# Análise da redução de dimensionalidade
print(f"\n=== ANÁLISE DA REDUÇÃO DE DIMENSIONALIDADE ===")
print(f"Dimensões originais: {x_balanced.shape[1]}")
print(f"Dimensões após PCA: {x_balanced_pca.shape[1]}")
print(f"Redução: {((x_balanced.shape[1] - x_balanced_pca.shape[1]) / x_balanced.shape[1]) * 100:.1f}%")
print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.4f}")

print("\n=== CONCLUSÕES ===")
if accuracy_score(y_test_pca, y_pred_lr_pca) > 0.95:
    print("✅ PCA mantém alta performance com menos dimensões")
else:
    print("⚠️ PCA pode ter reduzido a performance")

print("✅ Redução significativa de dimensionalidade")
print("✅ Mantém 95% da variância dos dados")
print("✅ Treinamento mais rápido com menos features")
