# Análise Comparativa de Algoritmos de Classificação

Este projeto implementa uma análise comparativa de algoritmos de classificação, incluindo pré-processamento, otimização de hiperparâmetros e análise estatística.

## Estrutura do Projeto

- **`classificacao.py`** - Arquivo principal que executa a análise completa
- **`pre_processador.py`** - Classe para pré-processamento (normalização, seleção de características, balanceamento)
- **`otimizador.py`** - Classe para otimização de hiperparâmetros com Optuna
- **`visualizador.py`** - Classe para geração de gráficos e visualizações
- **`output_logger.py`** - Classe para logging estruturado dos resultados
- **`requirements.txt`** - Dependências do projeto

## Algoritmos Implementados

- **Gradient Boosting Classifier**
- **Support Vector Classifier (SVC)**
- **Multi-Layer Perceptron (MLP)**
- **ANFIS Classifier**

## Experimentos

O projeto executa dois experimentos principais:

1. **Apenas Normalização**: Usa apenas normalização RobustScaler
2. **Processamento Completo**: Normalização + Seleção de características + Balanceamento SMOTE + Otimização de hiperparâmetros

## Datasets

- `winequality-red.csv` - Qualidade do vinho
- `fetal_health.csv` - Saúde fetal

## Instalação

Para instalar as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Execução

Para executar a análise completa:

```bash
python classificacao.py
```

## Resultados

Os resultados são salvos automaticamente em:
- `analise_classificacao.txt` - Log detalhado da análise
- Gráficos exibidos durante a execução

## Funcionalidades

### Pré-processamento
- Normalização com RobustScaler
- Seleção automática de características com Random Forest e cotovelo
- Balanceamento de classes com SMOTE

### Otimização
- Otimização de hiperparâmetros com Optuna
- Cross-validation para validação
- 15 trials por algoritmo

### Análise Estatística
- Teste de Friedman
- Post-hoc de Nemenyi
- 30 repetições por experimento

### Visualizações
- Boxplots comparativos
- Matrizes de confusão
- Análise de importância de características
- Comparação entre experimentos
