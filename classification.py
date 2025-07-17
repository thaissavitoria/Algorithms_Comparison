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
        
        importancias_sorted = importancias
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
        
        min_features = max(3, int(len(importancias) * 0.15)) 
        max_features = int(len(importancias) * 0.85)
        
        # Usar a melhor estratégia entre 80% de importância e ponto de cotovelo
        n_recomendado = perc_results.get(0.80, ponto_cotovelo)
        if n_recomendado is None:
            n_recomendado = ponto_cotovelo
            
        n_recomendado = max(n_recomendado, min_features)
        n_recomendado = min(n_recomendado, max_features, len(importancias))
        
        print(f"\nESTRATÉGIA DE RECOMENDAÇÃO:")
        print(f"  80% importância total: {perc_results.get(0.80, 'N/A')}")
        print(f"  Ponto de cotovelo: {ponto_cotovelo}")
        print(f"  Mínimo permitido (15%): {min_features}")
        print(f"  Máximo permitido (85%): {max_features}")
        print(f"  RECOMENDAÇÃO FINAL: {n_recomendado}")

        
        self.rf_info = {
            'importancias': importancias,
            'importancia_cumulativa': importancia_cumulativa,
            'feature_importance_df': feature_importance_df,
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
        if self.rf_info is None:
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 8))
        
        top = self.rf_info['feature_importance_df'].head(15)
        ax1.barh(range(len(top)), top['importancia'].values)
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top['caracteristica'].values)
        ax1.set_xlabel('Importância')
        ax1.set_title('Top Características Mais Importantes')
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
                mf_class=trial.suggest_categorical('mf_class', ['Gaussian', 'Trapezoidal']),
                vanishing_strategy=trial.suggest_categorical('vanishing_strategy', ['prod', 'mean']),
                epochs=trial.suggest_categorical('epochs', [20, 35]),
                batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
                optim_params={'lr': trial.suggest_float('lr', 0.001, 0.01, log=True)},
                optim=trial.suggest_categorical('optim', ['Adam', 'SGD']),
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

def processar_labels(y):
    """Processa as labels convertendo para 0, 1, 2 se necessário para o ANFIS"""
    label_encoder = None
    unique_classes = np.unique(y)
    print(f"Classes originais: {unique_classes}")

    if min(unique_classes) != 0:
        # Classes devem ser convertidas para 0, 1, 2 porque o anfis não aceita classes começando de 1
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Classes convertidas: {np.unique(y)}")
    
    return y, label_encoder

def executar_loop_repeticoes(modelos, X_data, y_data, n_execucoes, processar_fn):
    """Executa o loop de repetições para avaliação dos modelos"""
    todos_resultados = {nome: {
        'acuracia': [], 'f1_score': [], 'precisao': [], 'recall': []
    } for nome in modelos.keys()}

    X_teste_final = None
    y_teste_final = None
    y_pred_final = {}

    for i in range(1, n_execucoes + 1):
        print(f"Repetição {i}/{n_execucoes}", end='\r')

        dados_split = processar_fn(X_data, y_data, i)
        X_treino, X_teste, y_treino, y_teste = dados_split

        if i == n_execucoes:
            X_teste_final = X_teste
            y_teste_final = y_teste

        for nome, modelo in modelos.items():
            try:
                modelo_clone = clone(modelo)
                modelo_clone.fit(X_treino, y_treino)
                y_pred = modelo_clone.predict(X_teste)
                
                if i == n_execucoes:
                    y_pred_final[nome] = y_pred 

                todos_resultados[nome]['acuracia'].append(accuracy_score(y_teste, y_pred))
                todos_resultados[nome]['f1_score'].append(f1_score(y_teste, y_pred, average='weighted'))
                todos_resultados[nome]['precisao'].append(precision_score(y_teste, y_pred, average='weighted'))
                todos_resultados[nome]['recall'].append(recall_score(y_teste, y_pred, average='weighted'))
            except Exception as e:
                print(f"\nErro ao treinar {nome}: {e}")
                todos_resultados[nome]['acuracia'].append(0.0)
                todos_resultados[nome]['f1_score'].append(0.0)
                todos_resultados[nome]['precisao'].append(0.0)
                todos_resultados[nome]['recall'].append(0.0)

    return todos_resultados, X_teste_final, y_teste_final, y_pred_final

def executar_analise_completa_dataset(X, y, nome_dataset, modelos, otimizador, visualizador, n_execucoes=30):
    print(f"\n" + "*"*60)
    print(f"CONJUNTO DE DADOS DE {nome_dataset.upper()}")
    print("*"*60)
    
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} características")
    print(f"Classes: {sorted(y.unique())}")

    print("\n" + "="*80)
    print("EXECUTANDO EXPERIMENTO 1: APENAS NORMALIZAÇÃO")
    print("="*80)
    
    resultados_simples, X_teste_simples, y_teste_simples, y_pred_simples = executar_experimento_apenas_normalizacao(
        X, y, modelos, n_execucoes=n_execucoes
    )

    print("\n" + "="*80)
    print("EXECUTANDO EXPERIMENTO 2: PROCESSAMENTO COMPLETO")
    print("="*80)
    
    # Criar novo preprocessador para o experimento completo
    preprocessador_completo = PreProcessador()
    
    resultados_completo, modelos_otimizados, X_teste_completo, y_teste_completo, y_pred_completo = executar_experimento_avancado(
        X, y, modelos, preprocessador_completo, otimizador, n_execucoes=n_execucoes
    )

    print(f"\n" + "="*80)
    print(f"ANÁLISE COMPARATIVA - {nome_dataset.upper()}")
    print("="*80)

    algoritmos = list(resultados_simples.keys())
    
    comparacao_df = comparar_experimentos(resultados_simples, resultados_completo, algoritmos)
    
    print("\n" + "-"*50)
    print("ANÁLISE ESTATÍSTICA - APENAS NORMALIZAÇÃO")
    print("-"*50)
    analise_estatistica_abrangente(resultados_simples)
    
    print("\n" + "-"*50)
    print("ANÁLISE ESTATÍSTICA - PROCESSAMENTO COMPLETO")
    print("-"*50)
    analise_estatistica_abrangente(resultados_completo)

    print("\nGerando visualizações comparativas...")
    plotar_comparacao_experimentos(resultados_simples, resultados_completo, algoritmos)
    
    print("\nGerando visualizações detalhadas...")
    visualizador.plotar_comparacao_performance(resultados_simples, algoritmos)
    visualizador.plotar_comparacao_performance(resultados_completo, algoritmos)
    
    return resultados_simples, resultados_completo, algoritmos

def get_modelos():
    return {
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(random_state=42, probability=True),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000),
        'AnfisClassifier': AnfisClassifier(epochs=25, optim_params={'lr': 0.005}, verbose=False, device='cpu')
    }

def executar_experimento_apenas_normalizacao(X, y, modelos, n_execucoes=30):
    print(f"Executando experimento APENAS COM NORMALIZAÇÃO - {n_execucoes} repetições...")

    y, label_encoder = processar_labels(y)

    print("\n"+"-"*50)
    print("ETAPA 1: APENAS NORMALIZAÇÃO (SEM SELEÇÃO NEM BALANCEAMENTO)")
    print("-"*50)
    
    normalizador = RobustScaler()
    X_normalizado = normalizador.fit_transform(X)
    print(f"Dados apenas normalizados: {X_normalizado.shape}")
    print(f"Distribuição original das classes: {np.unique(y, return_counts=True)}")

    print("\n"+"-"*50)
    print("ETAPA 2: AVALIAÇÃO COM MODELOS PADRÃO")
    print("-"*50)

    def processar_split_simples(X_data, y_data, random_state):
        return train_test_split(X_data, y_data, test_size=0.2, random_state=random_state, stratify=y_data)

    todos_resultados, X_teste_final, y_teste_final, y_pred_final = executar_loop_repeticoes(
        modelos, X_normalizado, y, n_execucoes, processar_split_simples
    )

    print(f"\nExperimento APENAS NORMALIZAÇÃO concluído! {n_execucoes} repetições executadas.")
    return todos_resultados, X_teste_final, y_teste_final, y_pred_final

def executar_experimento_avancado(X, y, modelos, preprocessador, otimizador, n_execucoes=30):
    print(f"Executando experimento COMPLETO (Normalização + Seleção + Balanceamento) - {n_execucoes} repetições...")

    y, label_encoder = processar_labels(y)

    print("\n"+"-"*50)
    print("ETAPA 1: PRÉ-PROCESSAMENTO COMPLETO")
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
    print("ETAPA 3: AVALIAÇÃO COM REPETIÇÕES")
    print("-"*50)

    def processar_split_avancado(X_orig, y_orig, random_state):
        # dados processados
        X_treino, X_teste_processado, y_treino, y_teste_processado = train_test_split(
            X_processado, y_processado, test_size=0.2, random_state=random_state, stratify=y_processado
        )
        
        # dados originais para teste
        X_original_treino, X_original_teste, y_original_treino, y_original_teste = train_test_split(
            X_orig, y_orig, test_size=0.2, random_state=random_state, stratify=y_orig
        )
        
        return X_treino, y_treino, X_original_teste, y_original_teste

    todos_resultados = {nome: {
        'acuracia': [], 'f1_score': [], 'precisao': [], 'recall': []
    } for nome in modelos_otimizados.keys()}

    X_teste_final = None
    y_teste_final = None
    y_pred_final = {}  
    modelos_treinados = {}

    for i in range(1, n_execucoes + 1):
        print(f"Repetição {i}/{n_execucoes}", end='\r')

        X_treino, y_treino, X_original_teste, y_original_teste = processar_split_avancado(X, y, i)

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

                # avaliar contra y ORIGINAL (sem transformação)
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

    print(f"\nExperimento COMPLETO concluído! {n_execucoes} repetições executadas.")
    return todos_resultados, modelos_otimizados, X_teste_final, y_teste_final, y_pred_final

def comparar_experimentos(resultados_simples, resultados_completos, algoritmos):
    print("\n" + "="*70)
    print("COMPARAÇÃO: APENAS NORMALIZAÇÃO vs PROCESSAMENTO COMPLETO")
    print("="*70)

    comparacao_df = pd.DataFrame()
    
    for alg in algoritmos:
        acc_simples = np.mean(resultados_simples[alg]['acuracia'])
        f1_simples = np.mean(resultados_simples[alg]['f1_score'])
        prec_simples = np.mean(resultados_simples[alg]['precisao'])
        rec_simples = np.mean(resultados_simples[alg]['recall'])
        
        acc_completo = np.mean(resultados_completos[alg]['acuracia'])
        f1_completo = np.mean(resultados_completos[alg]['f1_score'])
        prec_completo = np.mean(resultados_completos[alg]['precisao'])
        rec_completo = np.mean(resultados_completos[alg]['recall'])
        
        diff_acc = acc_completo - acc_simples
        diff_f1 = f1_completo - f1_simples
        diff_prec = prec_completo - prec_simples
        diff_rec = rec_completo - rec_simples
        
        linha = pd.DataFrame({
            'Algoritmo': [alg],
            'Acc_Simples': [acc_simples],
            'Acc_Completo': [acc_completo],
            'Diff_Acc': [diff_acc],
            'F1_Simples': [f1_simples],
            'F1_Completo': [f1_completo],
            'Diff_F1': [diff_f1],
            'Prec_Simples': [prec_simples],
            'Prec_Completo': [prec_completo],
            'Diff_Prec': [diff_prec],
            'Rec_Simples': [rec_simples],
            'Rec_Completo': [rec_completo],
            'Diff_Rec': [diff_rec]
        })
        
        comparacao_df = pd.concat([comparacao_df, linha], ignore_index=True)
    
    print("\nRESULTADOS DETALHADOS:")
    print("-" * 50)
    for _, row in comparacao_df.iterrows():
        print(f"\n{row['Algoritmo']}:")
        print(f"  Acurácia:  {row['Acc_Simples']:.4f} → {row['Acc_Completo']:.4f} (Δ: {row['Diff_Acc']:+.4f})")
        print(f"  F1-Score:  {row['F1_Simples']:.4f} → {row['F1_Completo']:.4f} (Δ: {row['Diff_F1']:+.4f})")
        print(f"  Precisão:  {row['Prec_Simples']:.4f} → {row['Prec_Completo']:.4f} (Δ: {row['Diff_Prec']:+.4f})")
        print(f"  Recall:    {row['Rec_Simples']:.4f} → {row['Rec_Completo']:.4f} (Δ: {row['Diff_Rec']:+.4f})")
    
    print("\n" + "-"*50)
    print("RESUMO DAS MELHORIAS:")
    print("-"*50)
    
    melhorias = {
        'Acurácia': comparacao_df['Diff_Acc'].tolist(),
        'F1-Score': comparacao_df['Diff_F1'].tolist(),
        'Precisão': comparacao_df['Diff_Prec'].tolist(),
        'Recall': comparacao_df['Diff_Rec'].tolist()
    }
    
    for metrica, valores in melhorias.items():
        melhores = sum(1 for v in valores if v > 0)
        piores = sum(1 for v in valores if v < 0)
        iguais = sum(1 for v in valores if abs(v) < 0.001)
        media_melhoria = np.mean(valores)
        
        print(f"{metrica}:")
        print(f"  Algoritmos que melhoraram: {melhores}/{len(algoritmos)}")
        print(f"  Algoritmos que pioraram: {piores}/{len(algoritmos)}")
        print(f"  Melhoria média: {media_melhoria:+.4f}")
    
    print("\n" + "-"*50)
    print("MELHORES ALGORITMOS POR CONFIGURAÇÃO:")
    print("-"*50)
    
    # F1-Score como métrica principal
    melhor_simples = comparacao_df.loc[comparacao_df['F1_Simples'].idxmax(), 'Algoritmo']
    melhor_completo = comparacao_df.loc[comparacao_df['F1_Completo'].idxmax(), 'Algoritmo']
    
    f1_melhor_simples = comparacao_df.loc[comparacao_df['F1_Simples'].idxmax(), 'F1_Simples']
    f1_melhor_completo = comparacao_df.loc[comparacao_df['F1_Completo'].idxmax(), 'F1_Completo']
    
    print(f"Apenas Normalização: {melhor_simples} (F1: {f1_melhor_simples:.4f})")
    print(f"Processamento Completo: {melhor_completo} (F1: {f1_melhor_completo:.4f})")
    
    melhoria_absoluta = f1_melhor_completo - f1_melhor_simples
    print(f"Melhoria absoluta no melhor resultado: {melhoria_absoluta:+.4f}")
    
    return comparacao_df

def plotar_comparacao_experimentos(resultados_simples, resultados_completos, algoritmos):
    """Plota gráficos comparativos entre os dois experimentos"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metricas = ['acuracia', 'f1_score', 'precisao', 'recall']
    titulos = ['Acurácia', 'F1-Score', 'Precisão', 'Recall']
    
    for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        dados_simples = []
        dados_completos = []
        labels = []
        
        for alg in algoritmos:
            dados_simples.extend(resultados_simples[alg][metrica])
            dados_completos.extend(resultados_completos[alg][metrica])
            labels.extend([f'{alg}\n(Simples)'] * len(resultados_simples[alg][metrica]))
            labels.extend([f'{alg}\n(Completo)'] * len(resultados_completos[alg][metrica]))
        
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
        axes[i].set_title(f'Comparação: {titulo}')
        axes[i].set_ylabel(titulo)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("Comparação: Apenas Normalização vs Processamento Completo", y=1.02, fontsize=16)
    plt.show()

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
    otimizador = OtimizadorHiperparametros()
    visualizador = VisualizadorAvancado()
    modelos = get_modelos()

    n_execucoes = 30

    try:
        X_vinho, y_vinho = carregar_dataset('winequality-red.csv', 'quality')
        resultados_vinho_simples, resultados_vinho_completo, algoritmos_vinho = executar_analise_completa_dataset(
            X_vinho, y_vinho, "Qualidade do Vinho", modelos, otimizador, visualizador, n_execucoes
        )
    except FileNotFoundError:
        print("ERRO: Arquivo 'winequality-red.csv' não encontrado!")
        resultados_vinho_simples = resultados_vinho_completo = algoritmos_vinho = None

    try:
        X_fetal, y_fetal = carregar_dataset('fetal_health.csv', 'fetal_health')
        resultados_fetal_simples, resultados_fetal_completo, algoritmos_fetal = executar_analise_completa_dataset(
            X_fetal, y_fetal, "Saúde Fetal", modelos, otimizador, visualizador, n_execucoes
        )
    except FileNotFoundError:
        print("ERRO: Arquivo 'fetal_health.csv' não encontrado!")
        resultados_fetal_simples = resultados_fetal_completo = algoritmos_fetal = None

    print("\n" + "="*70)
    print("RESUMO FINAL DOS DOIS DATASETS")
    print("="*70)
    
    if resultados_vinho_simples and resultados_vinho_completo:
        print("\nQUALIDADE DO VINHO:")
        melhor_vinho_simples = max(algoritmos_vinho, key=lambda x: np.mean(resultados_vinho_simples[x]['f1_score']))
        melhor_vinho_completo = max(algoritmos_vinho, key=lambda x: np.mean(resultados_vinho_completo[x]['f1_score']))
        
        f1_vinho_simples = np.mean(resultados_vinho_simples[melhor_vinho_simples]['f1_score'])
        f1_vinho_completo = np.mean(resultados_vinho_completo[melhor_vinho_completo]['f1_score'])
        
        print(f"  Melhor (Apenas Normalização): {melhor_vinho_simples} (F1: {f1_vinho_simples:.4f})")
        print(f"  Melhor (Processamento Completo): {melhor_vinho_completo} (F1: {f1_vinho_completo:.4f})")
        print(f"  Melhoria: {f1_vinho_completo - f1_vinho_simples:+.4f}")

    if resultados_fetal_simples and resultados_fetal_completo:
        print("\nSAÚDE FETAL:")
        melhor_fetal_simples = max(algoritmos_fetal, key=lambda x: np.mean(resultados_fetal_simples[x]['f1_score']))
        melhor_fetal_completo = max(algoritmos_fetal, key=lambda x: np.mean(resultados_fetal_completo[x]['f1_score']))
        
        f1_fetal_simples = np.mean(resultados_fetal_simples[melhor_fetal_simples]['f1_score'])
        f1_fetal_completo = np.mean(resultados_fetal_completo[melhor_fetal_completo]['f1_score'])
        
        print(f"  Melhor (Apenas Normalização): {melhor_fetal_simples} (F1: {f1_fetal_simples:.4f})")
        print(f"  Melhor (Processamento Completo): {melhor_fetal_completo} (F1: {f1_fetal_completo:.4f})")
        print(f"  Melhoria: {f1_fetal_completo - f1_fetal_simples:+.4f}")

    print(f"\n🎉 Análise completa finalizada com {n_execucoes} repetições por experimento! 🎉")


if __name__ == '__main__':
    main()