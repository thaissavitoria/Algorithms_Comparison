import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from scipy.stats import wilcoxon
from sklearn.base import clone
from xanfis import AnfisClassifier
import warnings
warnings.filterwarnings('ignore')

class PreProcessador:
    def __init__(self):
        self.threshold_importancia = 0.01
        self.normalizador = RobustScaler()
        self.seletor = None
        self.balanceador = None
        self.nomes_caracteristicas = None
        self.rf_info = None

    def analisar_rf_para_selecao_caracteristicas(self, X_normalizado, y, plot=True):
        """
        Analisa usando Random Forest para determinar o número ideal de características
        baseado na importância das características.
        """
        print(f"\n--- Análise Random Forest para Seleção de Características ---")
        
        rf_analyzer = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_analyzer.fit(X_normalizado, y)
        
        importancias = rf_analyzer.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'caracteristica': [f'feat_{i}' if self.nomes_caracteristicas is None else self.nomes_caracteristicas[i] 
                             for i in range(len(importancias))],
            'importancia': importancias
        }).sort_values('importancia', ascending=False)
        
        print(f"Número total de características: {len(importancias)}")
        print(f"\nTop 10 características mais importantes:")
        for i in range(min(10, len(feature_importance_df))):
            print(f"  {i+1}. {feature_importance_df.iloc[i]['caracteristica']}: {feature_importance_df.iloc[i]['importancia']:.4f}")
        
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
        print(f"\nAnálise por threshold de importância:")
        
        threshold_results = {}
        for threshold in thresholds:
            n_features = len(feature_importance_df[feature_importance_df['importancia'] >= threshold])
            reduction = len(importancias) - n_features
            reduction_pct = (reduction / len(importancias)) * 100
            threshold_results[threshold] = n_features
            print(f"  {threshold:.3f}: {n_features} características (redução: {reduction} = {reduction_pct:.1f}%)")
        
        importancia_cumulativa = np.cumsum(feature_importance_df['importancia'].values)
        
        perc_importancia = [0.80, 0.85, 0.90, 0.95, 0.99]
        print(f"\nAnálise por percentual de importância total:")
        
        perc_results = {}
        for perc in perc_importancia:
            n_comp = np.argmax(importancia_cumulativa >= perc) + 1
            n_comp = min(n_comp, len(importancias))
            perc_results[perc] = n_comp
            reduction = len(importancias) - n_comp
            reduction_pct = (reduction / len(importancias)) * 100
            print(f"  {perc*100:.0f}%: {n_comp} características (redução: {reduction} = {reduction_pct:.1f}%)")
        
        importancias_sorted = feature_importance_df['importancia'].values
        if len(importancias_sorted) > 3:
            diffs = np.diff(importancias_sorted)
            second_diffs = np.diff(diffs)
            
            abs_second_diffs = np.abs(second_diffs)
            threshold_diff = np.percentile(abs_second_diffs, 75)
            cotovelo_candidates = np.where(abs_second_diffs < threshold_diff)[0] + 2
            
            if len(cotovelo_candidates) > 0:
                ponto_cotovelo = cotovelo_candidates[0]
            else:
                ponto_cotovelo = perc_results.get(0.90, len(importancias) // 2)
        else:
            ponto_cotovelo = len(importancias)
        
        print(f"\nPonto de cotovelo detectado: {ponto_cotovelo} características")
        
        n_por_threshold = threshold_results.get(self.threshold_importancia, len(importancias))
        
        min_features = max(3, int(len(importancias) * 0.15)) 
        max_features = int(len(importancias) * 0.85)
        
        n_recomendado = max(n_por_threshold, perc_results.get(0.80, min_features))
        n_recomendado = max(n_recomendado, min_features)
        n_recomendado = min(n_recomendado, max_features, len(importancias))
        
        print(f"\nESTRATÉGIA DE RECOMENDAÇÃO:")
        print(f"  Por threshold ({self.threshold_importancia:.3f}): {n_por_threshold}")
        print(f"  80% importância total: {perc_results.get(0.80, 'N/A')}")
        print(f"  Ponto de cotovelo: {ponto_cotovelo}")
        print(f"  Mínimo permitido (15%): {min_features}")
        print(f"  Máximo permitido (85%): {max_features}")
        print(f"  RECOMENDAÇÃO FINAL: {n_recomendado}")

        
        self.rf_info = {
            'importancias': importancias,
            'importancia_cumulativa': importancia_cumulativa,
            'feature_importance_df': feature_importance_df,
            'threshold_results': threshold_results,
            'perc_results': perc_results,
            'ponto_cotovelo': ponto_cotovelo,
            'n_recomendado': n_recomendado,
            'min_features': min_features,
            'max_features': max_features,
            'rf_model': rf_analyzer
        }
        
        if plot:
            self._plotar_analise_rf()
        
        return n_recomendado
    
    def _plotar_analise_rf(self):
        """Plota gráficos da análise Random Forest"""
        if self.rf_info is None:
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 8))
        
        top_15 = self.rf_info['feature_importance_df'].head(15)
        ax1.barh(range(len(top_15)), top_15['importancia'].values)
        ax1.set_yticks(range(len(top_15)))
        ax1.set_yticklabels(top_15['caracteristica'].values)
        ax1.set_xlabel('Importância')
        ax1.set_title('Top 15 Características Mais Importantes')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        n_features = len(self.rf_info['importancias'])
        ax2.plot(range(1, n_features + 1), self.rf_info['importancia_cumulativa'], 'bo-', linewidth=2)
        
        ax2.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='80%')
        ax2.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='90%')
        ax2.axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='95%')
        
        if 0.90 in self.rf_info['perc_results']:
            ax2.axvline(x=self.rf_info['perc_results'][0.90], color='red', linestyle=':', alpha=0.7)
        ax2.axvline(x=self.rf_info['ponto_cotovelo'], color='green', linestyle=':', alpha=0.7, label='Cotovelo')
        ax2.axvline(x=self.rf_info['n_recomendado'], color='blue', linestyle=':', alpha=0.7, label='Recomendado')
        
        ax2.set_xlabel('Número de Características')
        ax2.set_ylabel('Importância Cumulativa')
        ax2.set_title('Importância Cumulativa das Características')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.show()

        print(f"\n--- Resumo da Análise Random Forest ---")
        total_features = len(self.rf_info['importancias'])
        print(f"Características originais: {total_features}")
        print(f"Características recomendadas: {self.rf_info['n_recomendado']}")
        print(f"Redução: {total_features - self.rf_info['n_recomendado']} características")
        print(f"Percentual de redução: {((total_features - self.rf_info['n_recomendado'])/total_features*100):.1f}%")
        
        top_features = self.rf_info['feature_importance_df'].head(self.rf_info['n_recomendado'])
        print(f"\nCaracterísticas que seriam mantidas (top {self.rf_info['n_recomendado']}):")
        for i, row in top_features.iterrows():
            print(f"  {row['caracteristica']}: {row['importancia']:.4f}")

    def ajustar_e_transformar(self, X, y):
        print(f"Iniciando pré-processamento:")

        if hasattr(X, 'columns'):
            self.nomes_caracteristicas = X.columns.tolist()
            X = X.values
        else:
            self.nomes_caracteristicas = [f'caracteristica_{i}' for i in range(X.shape[1])]

        X_normalizado = self.normalizador.fit_transform(X)
        print(f"Dados normalizados: {X_normalizado.shape}")

        print("\nUsando análise Random Forest para determinar número ideal de características...")
        n_caracteristicas_rf = self.analisar_rf_para_selecao_caracteristicas(X_normalizado, y)
        n_caracteristicas = n_caracteristicas_rf
        print(f"\nNúmero de características definido pelo Random Forest: {n_caracteristicas}")

        self.seletor = SelectKBest(score_func=f_classif, k=n_caracteristicas)
        X_selecionado = self.seletor.fit_transform(X_normalizado, y)

        indices_selecionados = self.seletor.get_support(indices=True)
        self.nomes_caracteristicas = [self.nomes_caracteristicas[i] for i in indices_selecionados]
        print(f"Características selecionadas: {X_selecionado.shape[1]}")
        print(f"Características mantidas: {self.nomes_caracteristicas}")

        self.balanceador = SMOTE(random_state=42)
        X_balanceado, y_balanceado = self.balanceador.fit_resample(X_selecionado, y)
        print(f"Classes balanceadas: {X_balanceado.shape}")

        unicos, contagens = np.unique(y_balanceado, return_counts=True)
        print("Distribuição após balanceamento:")
        for classe, contagem in zip(unicos, contagens):
            print(f" Classe {classe}: {contagem} amostras")

        return X_balanceado, y_balanceado

class OtimizadorHiperparametros:
    def __init__(self):
        pass

    def _definir_espaco_de_busca(self, trial, nome_modelo):
        if nome_modelo == 'Gradient Boosting':
            return GradientBoostingClassifier(
                n_estimators=trial.suggest_categorical('n_estimators', [50, 100, 200]),
                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.01),
                max_depth=trial.suggest_int('max_depth', 3, 7)
            )

        elif nome_modelo == 'SVC':
            return SVC(
                C=trial.suggest_float('C', 0.1, 100, log=True),
                kernel=trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
                probability=True
            )

        elif nome_modelo == 'MLPClassifier':
            return MLPClassifier(
                hidden_layer_sizes=trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
                activation=trial.suggest_categorical('activation', ['tanh', 'relu']),
                alpha=trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                learning_rate_init=trial.suggest_float('learning_rate_init',0.001, 0.01, log=True),
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            )

        elif nome_modelo == 'AnfisClassifier':
            return AnfisClassifier(
                num_rules=trial.suggest_int('num_rules', 5, 12),
                mf_class=trial.suggest_categorical('mf_class', ['Gaussian', 'Triangular', 'Trapezoidal']),
                vanishing_strategy=trial.suggest_categorical('vanishing_strategy', ['prod', 'mean', 'blend']),
                epochs=trial.suggest_categorical('epochs', [30, 45]),
                batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
                optim_params={'lr': trial.suggest_float('lr', 0.001, 0.01, log=True)},
                verbose=False,
                device='cpu'
            )
        
        else:
            raise ValueError(f"Modelo '{nome_modelo}' não suportado")
        
    def otimizar_modelo(self, modelo_base, nome_modelo, X, y):
        print(f"\nOtimizando {nome_modelo} com Optuna...")

        def objective(trial):
            modelo = self._definir_espaco_de_busca(trial, nome_modelo)
            return cross_val_score(modelo, X, y, scoring='f1_macro', cv=3).mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=15, n_jobs=-1)

        melhor_trial = study.best_trial
        print(f"Melhor score: {melhor_trial.value:.4f}")
        print(f"Melhores parâmetros: {melhor_trial.params}")

        melhor_modelo = self._definir_espaco_de_busca(study.best_trial, nome_modelo)
        return melhor_modelo, melhor_trial.params, melhor_trial.value
    
class VisualizadorAvancado:
    def __init__(self):
        self.cmap = plt.cm.viridis
        
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
        print("\nGerando matrizes de confusão...")

        for nome, y_pred in y_pred_dict.items():
            self.plotar_matriz_confusao(y_teste, y_pred, nome, classes=classes)

def get_modelos():
    return {
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(random_state=42, probability=True),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000),
        'AnfisClassifier': AnfisClassifier(verbose=False)
    }

def executar_experimento_avancado(X, y, modelos, preprocessador, otimizador, n_execucoes=30):
    print(f"Executando experimento com {n_execucoes} repetições...")

    label_encoder = None
    unique_classes = np.unique(y)
    print(f"Classes originais: {unique_classes}")

    if min(unique_classes) != 0:
        #classes devem ser convertidas para 0, 1, 2 porque o anfis não aceita classes começando de 1
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Classes convertidas: {np.unique(y)}")

    print("\n"+"-"*50)
    print("ETAPA 1: PRÉ-PROCESSAMENTO E BALANCEAMENTO")
    print("-"*50)
    
    X_processado, y_processado = preprocessador.ajustar_e_transformar(X, y)
    print(f"Dados finais processados: {X_processado.shape}")
    print(f"Distribuição das classes processadas: {np.unique(y_processado, return_counts=True)}")

    print("\n"+"-"*50)
    print("ETAPA 2: OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("-"*50)
    
    modelos_otimizados = {}
    for nome, modelo in modelos.items():
        print(f"\nOtimizando {nome}...")
        
        modelo_otimo, melhores_params, melhor_score = otimizador.otimizar_modelo(
            modelo, nome, X_processado, y_processado
        )
        modelos_otimizados[nome] = modelo_otimo
        print(f"{nome} otimizado - Score: {melhor_score:.4f}")

    print("\n" + "-"*50)
    print("ETAPA 3: AVALIAÇÃO COM 30 REPETIÇÕES")
    print("-"*50)

    todos_resultados = {nome: {
        'acuracia': [], 'f1_score': [], 'precisao': [], 'recall': []
    } for nome in modelos_otimizados.keys()}

    X_teste_final = None
    y_teste_final = None
    y_pred_final = {}  
    modelos_treinados = {}

    for i in range(1, n_execucoes + 1):
        print(f"Repetição {i}/{n_execucoes}", end='\r')

        X_treino, X_teste_processado, y_treino, y_teste_processado = train_test_split(
            X_processado, y_processado, test_size=0.2, random_state=i, stratify=y_processado
        )

        X_original_treino, X_original_teste, y_original_treino, y_original_teste = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )

        if i == n_execucoes:
            X_teste_final = X_original_teste
            y_teste_final = y_original_teste

        for nome, modelo in modelos_otimizados.items():
            try:
                modelo_clone = clone(modelo)
                
                # treinar com dados processados (balanceados)
                modelo_clone.fit(X_treino, y_treino)
                
                # testar com dados ORIGINAIS (aplicando apenas normalização + seleção)
                X_original_teste_normalizado = preprocessador.normalizador.transform(X_original_teste)
                X_original_teste_selecionado = preprocessador.seletor.transform(X_original_teste_normalizado)
                
                y_pred = modelo_clone.predict(X_original_teste_selecionado)
                
                if i == n_execucoes:
                    modelos_treinados[nome] = modelo_clone
                    y_pred_final[nome] = y_pred 

                # Avaliar contra y ORIGINAL (sem transformação)
                todos_resultados[nome]['acuracia'].append(accuracy_score(y_original_teste, y_pred))
                todos_resultados[nome]['f1_score'].append(f1_score(y_original_teste, y_pred, average='weighted'))
                todos_resultados[nome]['precisao'].append(precision_score(y_original_teste, y_pred, average='weighted'))
                todos_resultados[nome]['recall'].append(recall_score(y_original_teste, y_pred, average='weighted'))
            except Exception as e:
                print(f"\nErro ao treinar {nome}: {e}")
                todos_resultados[nome]['acuracia'].append(0.0)
                todos_resultados[nome]['f1_score'].append(0.0)
                todos_resultados[nome]['precisao'].append(0.0)
                todos_resultados[nome]['recall'].append(0.0)

    print(f"\nExperimento concluído! {n_execucoes} repetições executadas.")
    return todos_resultados, modelos_otimizados, X_teste_final, y_teste_final, y_pred_final

def analise_estatistica_abrangente(resultados):
    algoritmos = list(resultados.keys())
    dados_acuracia = [resultados[alg]['acuracia'] for alg in algoritmos]

    print("\nTESTE DE FRIEDMAN")
    print("-" * 30)
    stat_friedman, p_friedman = friedmanchisquare(*dados_acuracia)
    print(f"Estatística: {stat_friedman:.4f}")
    print(f"P-value: {p_friedman:.6f}")

    if p_friedman < 0.05:
        print("Há diferenças significativas entre os algoritmos")

        print("\nPOST-HOC DE NEMENYI")
        print("-" * 25)
        df_dados = pd.DataFrame(dados_acuracia).T
        df_dados.columns = algoritmos

        try:
            posthoc = sp.posthoc_nemenyi_friedman(df_dados.values)
            posthoc.index = algoritmos
            posthoc.columns = algoritmos
            print(posthoc.round(4))
        except Exception as e:
            print(f"Erro no cálculo: {e}")
    else:
        print("Não há diferenças significativas entre os algoritmos")

    print("RANKING DOS ALGORITMOS")
    print("-" * 30)
    ranking = []
    for alg in algoritmos:
        media_acc = np.mean(resultados[alg]['acuracia'])
        media_f1 = np.mean(resultados[alg]['f1_score'])
        ranking.append((alg, media_acc, media_f1))

    ranking.sort(key=lambda x: x[1], reverse=True)

    for i, (alg, acc, f1) in enumerate(ranking):
        print(f"{i+1}. {alg}: Acc={acc:.4f}, F1={f1:.4f}")

def carregar_dataset(caminho_arquivo, coluna_alvo):
    df = pd.read_csv(caminho_arquivo)
    X = df.drop(coluna_alvo, axis=1)
    y = df[coluna_alvo]
    return X, y

def main():
    preprocessador = PreProcessador()

    otimizador = OtimizadorHiperparametros()

    visualizador = VisualizadorAvancado()

    modelos = get_modelos()

    print("\n" + "*"*60)
    print("CONJUNTO DE DADOS DE QUALIDADE DO VINHO")
    print("*"*60)

    try:
        X_vinho, y_vinho = carregar_dataset('winequality-red.csv', 'quality')
        print(f"Dataset carregado: {X_vinho.shape[0]} amostras, {X_vinho.shape[1]} características")
        print(f"Classes: {sorted(y_vinho.unique())}")

        resultados_vinho, modelos_otimizados_vinho, X_teste_vinho, y_teste_vinho, y_pred_vinho = executar_experimento_avancado(
            X_vinho, y_vinho, modelos, preprocessador, otimizador, n_execucoes=1
        )

        print("\n" + "-"*50)
        print("ANÁLISE DE RESULTADOS - QUALIDADE DO VINHO")
        print("-"*50)

        algoritmos = list(resultados_vinho.keys())

        resumo_df = pd.DataFrame()
        for alg in algoritmos:
            resumo_df = pd.concat([resumo_df, pd.DataFrame({
                'Algoritmo': [alg],
                'Acc_Média': [np.mean(resultados_vinho[alg]['acuracia'])],
                'Acc_Desvio': [np.std(resultados_vinho[alg]['acuracia'])],
                'F1_Média': [np.mean(resultados_vinho[alg]['f1_score'])],
                'F1_Desvio': [np.std(resultados_vinho[alg]['f1_score'])]
            })], ignore_index=True)

        print(resumo_df.to_string(index=False, float_format='%.4f'))

        analise_estatistica_abrangente(resultados_vinho)

        print("\nGerando visualizações estáticas...")
        visualizador.plotar_comparacao_performance(resultados_vinho, algoritmos)

        visualizador.plotar_matrizes_confusao(y_teste_vinho, y_pred_vinho)

    except FileNotFoundError:
        print("ERRO: Arquivo 'winequality-red.csv' não encontrado!")

    print("\n" + "*"*60)
    print("CONJUNTO DE DADOS DE SAÚDE FETAL")
    print("*"*60)

    try:
        X_fetal, y_fetal = carregar_dataset('fetal_health.csv', 'fetal_health')
        print(f"Dataset carregado: {X_fetal.shape[0]} amostras, {X_fetal.shape[1]} características")
        print(f"Classes: {sorted(y_fetal.unique())}")

        resultados_fetal, modelos_otimizados_fetal, X_teste_fetal, y_teste_fetal, y_pred_fetal = executar_experimento_avancado(
            X_fetal, y_fetal, modelos, preprocessador, otimizador, n_execucoes=1
        )

        print("\n" + "-"*50)
        print("ANÁLISE DE RESULTADOS - SAÚDE FETAL")
        print("-"*50)

        algoritmos_fetal = list(resultados_fetal.keys())

        resumo_fetal_df = pd.DataFrame()
        for alg in algoritmos_fetal:
            resumo_fetal_df = pd.concat([resumo_fetal_df, pd.DataFrame({
                'Algoritmo': [alg],
                'Acc_Média': [np.mean(resultados_fetal[alg]['acuracia'])],
                'Acc_Desvio': [np.std(resultados_fetal[alg]['acuracia'])],
                'F1_Média': [np.mean(resultados_fetal[alg]['f1_score'])],
                'F1_Desvio': [np.std(resultados_fetal[alg]['f1_score'])],
                'Recall_Média': [np.mean(resultados_fetal[alg]['recall'])],
                'Precisão_Média': [np.mean(resultados_fetal[alg]['precisao'])]
            })], ignore_index=True)

        print(resumo_fetal_df.to_string(index=False, float_format='%.4f'))

        analise_estatistica_abrangente(resultados_fetal)

        print("\nGerando visualizações estáticas...")
        visualizador.plotar_comparacao_performance(resultados_fetal, algoritmos_fetal)

        classes_fetal = ['Normal', 'Suspect', 'Pathological']
        visualizador.plotar_matrizes_confusao(y_teste_fetal, y_pred_fetal, classes=classes_fetal)

        print("\n" + "="*50)
        print("COMPARAÇÃO ENTRE DATASETS")
        print("="*50)

        print("Resumo Wine Quality:")
        print(f"Melhor algoritmo: {max(algoritmos, key=lambda x: np.mean(resultados_vinho[x]['f1_score']))}")
        print(f"Melhor F1-Score: {max([np.mean(resultados_vinho[alg]['f1_score']) for alg in algoritmos]):.4f}")

        print("\nResumo Saúde Fetal:")
        print(f"Melhor algoritmo: {max(algoritmos_fetal, key=lambda x: np.mean(resultados_fetal[x]['f1_score']))}")
        print(f"Melhor F1-Score: {max([np.mean(resultados_fetal[alg]['f1_score']) for alg in algoritmos_fetal]):.4f}")

    except FileNotFoundError:
        print("ERRO: Arquivo fetal_health.csv' não encontrado!")


if __name__ == '__main__':
    main()