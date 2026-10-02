# 🤖 Databricks ML - Projetos Avançados em Machine Learning

Um repositório completo de projetos de **Machine Learning** construídos e otimizados via **Databricks**, cobrindo desde modelos supervisionados até técnicas avançadas de ML em escala.

## 📋 Descrição do Projeto

Este repositório organiza estudos e implementações práticas de algoritmos de machine learning com foco em:

- **Aprendizado Supervisionado**: Classificação e Regressão
- **Aprendizado Não-Supervisionado**: Clustering e Redução de Dimensionalidade
- **Aprendizado por Reforço**: Técnicas avançadas
- **ML em Produção**: Pipelines, deploy e monitoramento
- **Otimização de Modelos**: Tuning de hiperparâmetros e feature engineering
- **Integração com Databricks**: MLflow, Spark MLlib e bibliotecas populares

## 🎯 Caso de Uso

Ideal para:
- 🔬 **Cientistas de Dados**: Aprender e praticar algoritmos de ML
- 👨‍💻 **Engenheiros de ML**: Implementar pipelines em produção
- 📊 **Analistas**: Criar modelos preditivos e insights
- 🎓 **Aprendizado**: Referência prática para ML em escala
- 🏢 **Empresas**: Templates reutilizáveis para casos de uso reais

## 🏗️ Arquitetura do Fluxo ML

```
┌──────────────────────────────────────────────────────────────┐
│                     Dados Brutos                             │
│              (ELT via Spark/Delta Lake)                      │
└────────────────┬─────────────────────────────────────────────┘
                 │
         ┌───────▼──────────┐
         │  Exploração (EDA)│
         │  Visualização    │
         │  Profiling       │
         └───────┬──────────┘
                 │
         ┌───────▼──────────────────────┐
         │   Feature Engineering         │
         │   ├─ Transformação            │
         │   ├─ Normalização             │
         │   ├─ Seleção de Features      │
         │   └─ Feature Importance       │
         └───────┬──────────────────────┘
                 │
    ┌────────────┴──────────────┐
    │                           │
┌───▼──────────┐       ┌───────▼────────┐
│  Modelo 1    │       │  Modelo 2      │
│  Training    │       │  Training      │
│  & Tuning    │       │  & Tuning      │
└───┬──────────┘       └───────┬────────┘
    │                           │
    └───────────┬──────────────┘
                │
         ┌──────▼──────────┐
         │  Avaliação      │
         │  Métricas       │
         │  Comparação     │
         └──────┬──────────┘
                │
         ┌──────▼──────────┐
         │  Modelo Final   │
         │  Registro MLflow│
         │  Versionamento  │
         └──────┬──────────┘
                │
         ┌──────▼──────────┐
         │  Deploy         │
         │  Monitoring     │
         │  Predictions    │
         └─────────────────┘
```

## 📁 Estrutura do Projeto

```
databricks_ML/
├── README.md                          # Este arquivo
│
├── Supervisionado/                    # Aprendizado Supervisionado
│   │
│   ├── Classificação/                 # Problemas de Classificação
│   │   ├── [Notebook: Binary Classification]
│   │   ├── [Notebook: Multi-Class Classification]
│   │   ├── [Notebook: Imbalanced Classification]
│   │   └── [Outros notebooks de classificação]
│   │
│   └── Regressão/                     # Problemas de Regressão
│       ├── [Notebook: Linear Regression]
│       ├── [Notebook: Polynomial Regression]
│       ├── [Notebook: Time Series Forecasting]
│       └── [Outros notebooks de regressão]
│
├── Nao-Supervisionado/                # Aprendizado Não-Supervisionado
│   ├── Clustering/
│   ├── Reducao-Dimensionalidade/
│   └── Deteccao-Anomalias/
│
└── [Outros tópicos podem ser adicionados]
```

## 🎓 Projetos por Categoria

### 1️⃣ Aprendizado Supervisionado

#### Classificação

**O que é:** Prever categorias discretas (sim/não, A/B/C, etc.)

**Algoritmos Cobertos:**
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machines (SVM)
- Neural Networks

**Métricas de Avaliação:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Cross-validation

**Exemplo de Use Case:**
- Detecção de fraude (fraude/legítimo)
- Previsão de churn (permanece/sai)
- Classificação de sentimento (positivo/negativo/neutro)
- Diagnóstico médico (doente/saudável)

**Código Base:**
```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Criar modelo
rf = RandomForestClassifier(numTrees=100, maxDepth=10)

# Pipeline
pipeline = Pipeline(stages=[feature_assembler, rf])

# Treinamento
model = pipeline.fit(train_data)

# Predição
predictions = model.transform(test_data)

# Avaliação
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")
```

---

#### Regressão

**O que é:** Prever valores contínuos (preços, temperatura, quantidade, etc.)

**Algoritmos Cobertos:**
- Linear Regression
- Ridge & Lasso Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Time Series Forecasting (ARIMA, Prophet)

**Métricas de Avaliação:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE), Root MSE (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

**Exemplo de Use Case:**
- Previsão de preços (imóvel, ações, produtos)
- Forecasting de vendas
- Estimativa de demanda
- Previsão de consumo de energia

**Código Base:**
```python
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Criar modelo
rf_reg = RandomForestRegressor(numTrees=100, maxDepth=10)

# Pipeline
pipeline = Pipeline(stages=[feature_assembler, rf_reg])

# Treinamento
model = pipeline.fit(train_data)

# Predição
predictions = model.transform(test_data)

# Avaliação
evaluator = RegressionEvaluator(metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")
```

---

### 2️⃣ Aprendizado Não-Supervisionado

**Em Desenvolvimento**

Tópicos planejados:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- PCA (Principal Component Analysis)
- t-SNE / UMAP
- Detecção de Anomalias (Isolation Forest, LOF)

---

### 3️⃣ Aprendizado por Reforço

**Em Desenvolvimento**

Tópicos planejados:
- Q-Learning
- Policy Gradient
- Actor-Critic Methods
- Deep Reinforcement Learning

---

## 🚀 Como Usar Este Repositório

### 1️⃣ Pré-requisitos

- Conta Databricks (Community Edition ou Pro)
- Python 3.8+
- Conhecimento básico de ML e Spark
- Familiaridade com scikit-learn (opcional mas recomendado)

### 2️⃣ Importar Notebooks

**Via Databricks UI:**
1. Acesse seu workspace Databricks
2. Clique em "Import" → "URL"
3. Cole a URL do notebook raw do GitHub
4. Clique em "Import"

**Via Git Integration:**
1. Configure seu repositório Git no Databricks
2. Os notebooks serão sincronizados automaticamente

### 3️⃣ Configurar Cluster

```python
# Especificações recomendadas:
# - Runtime: 13.0 LTS ou superior
# - Driver: 8GB RAM minimum
# - Workers: 2x workers com 4GB cada
# - Python 3.9+
```

### 4️⃣ Executar Notebooks

```python
# Primeiro cell: Configurações e imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import *
import pandas as pd

# Confirme que Spark está disponível
print(f"Spark Version: {spark.version}")
print(f"MLflow Version: {mlflow.__version__}")
```

## 🔧 Componentes Principais

### Bibliotecas e Dependências

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| PySpark MLlib | 3.0+ | ML distributed |
| scikit-learn | 1.0+ | ML algorithms |
| XGBoost | 1.5+ | Gradient boosting |
| LightGBM | 3.2+ | Fast boosting |
| MLflow | 1.20+ | Experiment tracking |
| Pandas | 1.3+ | Data manipulation |
| Numpy | 1.20+ | Numerical computing |
| Matplotlib/Plotly | Latest | Visualization |

### MLflow Integration

Rastreamento automático de experimentos:

```python
import mlflow
from mlflow.sklearn import log_model

with mlflow.start_run():
    # Treinamento
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Logging de parâmetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Logging de métricas
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Registro do modelo
    log_model(model, "random_forest_model")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

## 📊 Fluxo Típico de um Projeto ML

### Fase 1: Preparação de Dados
```python
# Carregar dados
df = spark.read.format("delta").load("/path/to/data")

# Exploração
df.describe().show()
df.printSchema()

# Limpeza
df_clean = df.dropna().drop_duplicates()
```

### Fase 2: Feature Engineering
```python
from pyspark.ml.feature import StandardScaler, OneHotEncoder

# Seleção de features
features = ["feature1", "feature2", "feature3"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Normalização
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler])
df_features = pipeline.fit(df_clean).transform(df_clean)
```

### Fase 3: Treinamento
```python
# Split dados
train, test = df_features.randomSplit([0.8, 0.2], seed=42)

# Treinar modelo
rf = RandomForestClassifier(numTrees=100, maxDepth=10)
model = rf.fit(train)
```

### Fase 4: Avaliação
```python
# Predições
predictions = model.transform(test)

# Métricas
evaluator = BinaryClassificationEvaluator()
auc = evaluator.evaluate(predictions)

# Feature Importance
importance = model.stages[-1].featureImportances
```

### Fase 5: Deploy
```python
# Registrar no MLflow Model Registry
mlflow.register_model(f"runs:/{run_id}/model", "my_model")

# Carregar modelo em produção
loaded_model = mlflow.sklearn.load_model(f"models:/my_model/production")

# Fazer predições
predictions = loaded_model.predict(X_new)
```

## 💡 Dicas e Boas Práticas

### Performance

- ✅ Use `caching` para DataFrames reutilizados
- ✅ Particione dados apropriadamente
- ✅ Use `broadcast` para joins com pequenos DataFrames
- ✅ Ajuste o tamanho do batch training
- ❌ Evite `collect()` em datasets grandes
- ❌ Não treine em dados desbalanceados sem tratar

### Qualidade de Modelo

- ✅ Sempre use cross-validation
- ✅ Valide em dados completamente separados
- ✅ Monitore feature drift em produção
- ✅ Documente decisões de design
- ❌ Não ajuste demais (overfitting)
- ❌ Não ignore desbalanceamento de classes

### Reprodutibilidade

```python
# Defina seeds para reprodutibilidade
import random
random.seed(42)
import numpy as np
np.random.seed(42)

# Em Spark
spark.sparkContext.setRandomSeed(42)
```

## 📈 Métricas por Tipo de Problema

### Classificação Binária

| Métrica | Fórmula | Quando Usar |
|---------|---------|-----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Datasets balanceados |
| **Precision** | TP/(TP+FP) | Falsos positivos custosos |
| **Recall** | TP/(TP+FN) | Falsos negativos custosos |
| **F1-Score** | 2·Precision·Recall/(Precision+Recall) | Balance entre P e R |
| **AUC-ROC** | Área sob curva | Rankings gerais |

### Regressão

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MAE** | Σ\|y-ŷ\|/n | Erro médio em unidades reais |
| **RMSE** | √(Σ(y-ŷ)²/n) | Penaliza erros grandes |
| **R²** | 1-SS_res/SS_tot | Proporção de variância explicada |
| **MAPE** | Σ\|y-ŷ\|/(y·n)·100 | Erro percentual |

## 🛡️ Segurança e Governança

- ✅ Controle de acesso por role no Databricks
- ✅ Auditoria de experiências no MLflow
- ✅ Versionamento de modelos
- ✅ Monitoramento de model performance
- ✅ Lineage de dados e modelos

## 📈 Extensões Futuras

- [ ] Notebooks de Clustering e Redução de Dimensionalidade
- [ ] Exemplos de Deep Learning com PyTorch/TensorFlow
- [ ] Reinforcement Learning applications
- [ ] AutoML com Databricks AutoML
- [ ] Explainability (SHAP, LIME)
- [ ] Model Monitoring e Retraining Pipelines
- [ ] Exemplos de Production Deployment
- [ ] Benchmarks e Comparações de Algoritmos

## 🤝 Contribuindo

Contribuições são bem-vindas! Você pode:
1. Adicionar novos notebooks de ML
2. Melhorar exemplos existentes
3. Adicionar casos de uso práticos
4. Expandir documentação
5. Reportar bugs ou sugestões

## 📞 Suporte

Para dúvidas ou problemas:
- Abra uma issue no GitHub
- Consulte a documentação Databricks: https://docs.databricks.com/machine-learning
- Spark MLlib: https://spark.apache.org/docs/latest/ml-guide.html
- MLflow: https://mlflow.org/

## 🎓 Recursos de Aprendizado

- [Databricks Academy - ML](https://www.databricks.com/learn)
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)

## 📝 Licença

Este projeto é fornecido como material educacional e de referência.

---

**Desenvolvido com ❤️ para a comunidade de Machine Learning e Databricks**
