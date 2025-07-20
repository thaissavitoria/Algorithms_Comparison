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
        plt.title(f'{algoritmo}')
        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Real')
        plt.show()


    def plotar_matrizes_confusao(self, y_teste, y_pred_dict, classes=None):
        self.logger.write("\nGerando matrizes de confusão...\n")

        for nome, y_pred in y_pred_dict.items():
            self.plotar_matriz_confusao(y_teste, y_pred, nome, classes=classes)

    def plotar_comparacao_experimentos(self, resultados_simples, resultados_completos, algoritmos):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metricas = ['acuracia', 'f1_score', 'precisao', 'recall']
        titulos = ['Acurácia', 'F1-Score', 'Precisão', 'Recall']
        
        for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
            df_plot = pd.DataFrame()
            
            for alg in algoritmos:
                df_temp_simples = pd.DataFrame({
                    'Algoritmo': [alg] * len(resultados_simples[alg][metrica]),
                    'Configuração': ['Apenas Normalização'] * len(resultados_simples[alg][metrica]),
                    'Valor': resultados_simples[alg][metrica]
                })
                 
                df_temp_completo = pd.DataFrame({
                    'Algoritmo': [alg] * len(resultados_completos[alg][metrica]),
                    'Configuração': ['Processamento Completo'] * len(resultados_completos[alg][metrica]),
                    'Valor': resultados_completos[alg][metrica]
                })
                
                df_plot = pd.concat([df_plot, df_temp_simples, df_temp_completo], ignore_index=True)
            
            sns.boxplot(data=df_plot, x='Algoritmo', y='Valor', hue='Configuração', ax=axes[i])
            axes[i].set_title(f'{titulo}')
            axes[i].set_ylabel(titulo)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle("Comparação: Apenas Normalização vs Processamento Completo", y=1.02, fontsize=16)
        plt.show()
