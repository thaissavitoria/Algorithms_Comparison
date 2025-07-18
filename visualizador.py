import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon

class Visualizador:
    def __init__(self, logger):
        self.cmap = plt.cm.viridis
        self.logger = logger
        
    def plotar_comparacao_performance(self, resultados, algoritmos):
        df_comparacao = pd.DataFrame()

        for alg in algoritmos:
            df_temp = pd.DataFrame({
                'Algoritmo': [alg] * len(resultados[alg]['acuracia']),
                'Acurácia': resultados[alg]['acuracia'],
                'F1-Score': resultados[alg]['f1_score'],
                'Precisão': resultados[alg]['precisao'],
                'Recall': resultados[alg]['recall']
            })
            df_comparacao = pd.concat([df_comparacao, df_temp], ignore_index=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        metricas = ['Acurácia', 'F1-Score', 'Precisão', 'Recall']

        for i, metrica in enumerate(metricas):
            sns.boxplot(x='Algoritmo', y=metrica, data=df_comparacao, ax=axes[i])
            axes[i].set_title(metrica)
            axes[i].set_ylabel(metrica)
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.suptitle("Comparação Detalhada de Performance", y=1.02)
        plt.show()

    def plotar_significancia_estatistica(self, resultados, algoritmos):
        n_algoritmos = len(algoritmos)
        matriz_p = np.ones((n_algoritmos, n_algoritmos))

        for i, alg1 in enumerate(algoritmos):
            for j, alg2 in enumerate(algoritmos):
                if i != j:
                    try:
                        _, p_value = wilcoxon(
                            resultados[alg1]['acuracia'],
                            resultados[alg2]['acuracia']
                        )
                        matriz_p[i, j] = p_value
                    except:
                        matriz_p[i, j] = 1.0

        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_p, annot=True, cmap='RdYlBu_r', xticklabels=algoritmos, yticklabels=algoritmos, fmt=".4f")
        plt.title("Significância Estatística entre Algoritmos (p-values)")
        plt.xlabel("Algoritmos")
        plt.ylabel("Algoritmos")
        plt.show()

    def plotar_matriz_confusao(self, y_true, y_pred, algoritmo, classes=None):
        cm = confusion_matrix(y_true, y_pred)

        if classes is None:
            classes = np.unique(y_true)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Matriz de Confusão - {algoritmo}')
        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Real')
        plt.show()


    def plotar_matrizes_confusao(self, y_teste, y_pred_dict, classes=None):
        self.logger.write("\nGerando matrizes de confusão...\n")

        for nome, y_pred in y_pred_dict.items():
            self.plotar_matriz_confusao(y_teste, y_pred, nome, classes=classes)
