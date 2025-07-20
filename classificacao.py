import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xanfis import AnfisClassifier
from sklearn.base import clone
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

from pre_processador import PreProcessador
from output_logger import OutputLogger
from otimizador import OtimizadorHiperparametros
from visualizador import Visualizador

warnings.filterwarnings('ignore')

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

def executar_analise_completa_dataset(X, y, nome_dataset, modelos, otimizador, visualizador, logger, n_execucoes=30):
    logger.section(f"CONJUNTO DE DADOS DE {nome_dataset.upper()}")
    
    logger.write(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} características\n")
    logger.write(f"Classes: {sorted(y.unique())}\n")

    logger.section("EXECUTANDO EXPERIMENTO 1: APENAS NORMALIZAÇÃO")
    
    # Criar preprocessador para o experimento simples
    preprocessador = PreProcessador(logger)
    
    resultados_simples, X_teste_simples, y_teste_simples, y_pred_simples = executar_experimento_apenas_normalizacao(
        X, y, modelos, preprocessador, logger, n_execucoes=n_execucoes
    )

    logger.section("EXECUTANDO EXPERIMENTO 2: PROCESSAMENTO COMPLETO")

    resultados_completo, modelos_otimizados, X_teste_completo, y_teste_completo, y_pred_completo = executar_experimento_avancado(
        X, y, modelos, preprocessador, otimizador, logger, n_execucoes=n_execucoes
    )

    logger.section(f"ANÁLISE COMPARATIVA - {nome_dataset.upper()}")

    algoritmos = list(resultados_simples.keys())
    
    comparar_experimentos(resultados_simples, resultados_completo, algoritmos, logger)
    
    logger.subsection("ANÁLISE ESTATÍSTICA - APENAS NORMALIZAÇÃO")
    analise_estatistica_abrangente(resultados_simples, logger)
    
    logger.subsection("ANÁLISE ESTATÍSTICA - PROCESSAMENTO COMPLETO")
    analise_estatistica_abrangente(resultados_completo, logger)

    logger.write("\nGerando visualizações comparativas...\n")
    visualizador.plotar_comparacao_experimentos(resultados_simples, resultados_completo, algoritmos)
    
    logger.write("\nGerando visualizações detalhadas...\n")
    visualizador.plotar_comparacao_performance(resultados_simples, algoritmos)
    visualizador.plotar_comparacao_performance(resultados_completo, algoritmos)
    
    logger.write("\nGerando matrizes de confusão - Apenas Normalização...\n")
    visualizador.plotar_matrizes_confusao(y_teste_simples, y_pred_simples)
    
    logger.write("\nGerando matrizes de confusão - Processamento Completo...\n")
    visualizador.plotar_matrizes_confusao(y_teste_completo, y_pred_completo)
    
    return resultados_simples, resultados_completo, algoritmos

def get_modelos():
    return {
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(random_state=42, probability=True),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000),
        'AnfisClassifier': AnfisClassifier(epochs=25, optim_params={'lr': 0.005}, verbose=False, device='cpu')
    }

def executar_experimento_apenas_normalizacao(X, y, modelos, preProcessador, logger, n_execucoes=30):
    logger.write(f"Executando experimento APENAS COM NORMALIZAÇÃO - {n_execucoes} repetições...\n")

    y = preProcessador.processar_labels(y)

    logger.subsection("ETAPA 1: APENAS NORMALIZAÇÃO (SEM SELEÇÃO NEM BALANCEAMENTO)")
    
    X_normalizado = preProcessador.normalizar(X)
    logger.write(f"Dados apenas normalizados: {X_normalizado.shape}\n")
    logger.write(f"Distribuição original das classes: {np.unique(y, return_counts=True)}\n")

    logger.subsection("ETAPA 2: AVALIAÇÃO COM MODELOS PADRÃO")

    def processar_split_simples(X_data, y_data, random_state):
        return train_test_split(X_data, y_data, test_size=0.2, random_state=random_state, stratify=y_data)

    todos_resultados, X_teste_final, y_teste_final, y_pred_final = executar_loop_repeticoes(
        modelos, X_normalizado, y, n_execucoes, processar_split_simples
    )

    logger.write(f"\nExperimento APENAS NORMALIZAÇÃO concluído! {n_execucoes} repetições executadas.\n")
    return todos_resultados, X_teste_final, y_teste_final, y_pred_final

def executar_experimento_avancado(X, y, modelos, preprocessador, otimizador,  logger, n_execucoes=30):
    logger.write(f"Executando experimento COMPLETO (Normalização + Seleção + Balanceamento) - {n_execucoes} repetições...\n")

    y = preprocessador.processar_labels(y)

    logger.subsection("ETAPA 1: PRÉ-PROCESSAMENTO COMPLETO")
    
    X_processado, y_processado = preprocessador.ajustar_e_transformar(X, y)
    logger.write(f"Dados finais processados: {X_processado.shape}\n")
    logger.write(f"Distribuição das classes processadas: {np.unique(y_processado, return_counts=True)}\n")

    logger.subsection("ETAPA 2: OTIMIZAÇÃO DE HIPERPARÂMETROS")
    
    modelos_otimizados = {}
    for nome, modelo in modelos.items():
        logger.write(f"\nOtimizando {nome}...\n")
        
        modelo_otimo, melhores_params, melhor_score = otimizador.otimizar_modelo(
            modelo, nome, X_processado, y_processado
        )
        modelos_otimizados[nome] = modelo_otimo
        logger.write(f"{nome} otimizado - Score: {melhor_score:.4f}\n")

    logger.subsection("ETAPA 3: AVALIAÇÃO COM REPETIÇÕES")

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
                logger.write(f"\nErro ao treinar {nome}: {e}\n")
                todos_resultados[nome]['acuracia'].append(0.0)
                todos_resultados[nome]['f1_score'].append(0.0)
                todos_resultados[nome]['precisao'].append(0.0)
                todos_resultados[nome]['recall'].append(0.0)

    logger.write(f"\nExperimento COMPLETO concluído! {n_execucoes} repetições executadas.\n")
    return todos_resultados, modelos_otimizados, X_teste_final, y_teste_final, y_pred_final

def comparar_experimentos(resultados_simples, resultados_completos, algoritmos, logger):
    logger.subsection("COMPARAÇÃO: APENAS NORMALIZAÇÃO vs PROCESSAMENTO COMPLETO")

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
    
    logger.write("\nRESULTADOS DETALHADOS:\n")
    for _, row in comparacao_df.iterrows():
        logger.write(f"\n{row['Algoritmo']}:\n")
        logger.write(f"  Acurácia:  {row['Acc_Simples']:.4f} → {row['Acc_Completo']:.4f} (Δ: {row['Diff_Acc']:+.4f})\n")
        logger.write(f"  F1-Score:  {row['F1_Simples']:.4f} → {row['F1_Completo']:.4f} (Δ: {row['Diff_F1']:+.4f})\n")
        logger.write(f"  Precisão:  {row['Prec_Simples']:.4f} → {row['Prec_Completo']:.4f} (Δ: {row['Diff_Prec']:+.4f})\n")
        logger.write(f"  Recall:    {row['Rec_Simples']:.4f} → {row['Rec_Completo']:.4f} (Δ: {row['Diff_Rec']:+.4f})\n")
    
    logger.write("\nRESUMO DAS MELHORIAS:\n")
    
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
        
        logger.write(f"{metrica}:\n")
        logger.write(f"  Algoritmos que melhoraram: {melhores}/{len(algoritmos)}\n")
        logger.write(f"  Algoritmos que pioraram: {piores}/{len(algoritmos)}\n")
        logger.write(f"  Melhoria média: {media_melhoria:+.4f}\n")
    
    logger.write("\nMELHORES ALGORITMOS POR CONFIGURAÇÃO:\n")
    
    # F1-Score como métrica principal
    melhor_simples = comparacao_df.loc[comparacao_df['F1_Simples'].idxmax(), 'Algoritmo']
    melhor_completo = comparacao_df.loc[comparacao_df['F1_Completo'].idxmax(), 'Algoritmo']
    
    f1_melhor_simples = comparacao_df.loc[comparacao_df['F1_Simples'].idxmax(), 'F1_Simples']
    f1_melhor_completo = comparacao_df.loc[comparacao_df['F1_Completo'].idxmax(), 'F1_Completo']
    
    logger.write(f"Apenas Normalização: {melhor_simples} (F1: {f1_melhor_simples:.4f})\n")
    logger.write(f"Processamento Completo: {melhor_completo} (F1: {f1_melhor_completo:.4f})\n")
    
    melhoria_absoluta = f1_melhor_completo - f1_melhor_simples
    logger.write(f"Melhoria absoluta no melhor resultado: {melhoria_absoluta:+.4f}\n")

def analise_estatistica_abrangente(resultados, logger):
    algoritmos = list(resultados.keys())
    dados_acuracia = [resultados[alg]['acuracia'] for alg in algoritmos]

    logger.write("\nTESTE DE FRIEDMAN\n")
    stat_friedman, p_friedman = friedmanchisquare(*dados_acuracia)
    logger.write(f"Estatística: {stat_friedman:.4f}\n")
    logger.write(f"P-value: {p_friedman:.6f}\n")

    if p_friedman < 0.05:
        logger.write("Há diferenças significativas entre os algoritmos\n")

        logger.write("\nPOST-HOC DE NEMENYI\n")
        df_dados = pd.DataFrame(dados_acuracia).T
        df_dados.columns = algoritmos

        try:
            posthoc = sp.posthoc_nemenyi_friedman(df_dados.values)
            posthoc.index = algoritmos
            posthoc.columns = algoritmos
            logger.write(f"{posthoc.round(4)}\n")
        except Exception as e:
            logger.write(f"Erro no cálculo: {e}\n")
    else:
        logger.write("Não há diferenças significativas entre os algoritmos\n")

    logger.write("RANKING DOS ALGORITMOS\n")
    ranking = []
    for alg in algoritmos:
        media_acc = np.mean(resultados[alg]['acuracia'])
        media_f1 = np.mean(resultados[alg]['f1_score'])
        ranking.append((alg, media_acc, media_f1))

    ranking.sort(key=lambda x: x[1], reverse=True)

    for i, (alg, acc, f1) in enumerate(ranking):
        logger.write(f"{i+1}. {alg}: Acc={acc:.4f}, F1={f1:.4f}\n")

def carregar_dataset(caminho_arquivo, coluna_alvo):
    df = pd.read_csv(caminho_arquivo)
    X = df.drop(coluna_alvo, axis=1)
    y = df[coluna_alvo]
    return X, y

def main():
    logger = OutputLogger()
    
    logger.section("ANÁLISE DE ALGORITMOS DE CLASSIFICAÇÃO")
    logger.write("Iniciando análise comparativa de algoritmos de classificação\n")
    
    otimizador = OtimizadorHiperparametros(logger)
    visualizador = Visualizador(logger)
    modelos = get_modelos()

    try:
        X_vinho, y_vinho = carregar_dataset('winequality-red.csv', 'quality')
        resultados_vinho_simples, resultados_vinho_completo, algoritmos_vinho = executar_analise_completa_dataset(
            X_vinho, y_vinho, "Qualidade do Vinho", modelos, otimizador, visualizador, logger
        )
    except FileNotFoundError:
        logger.write("ERRO: Arquivo 'winequality-red.csv' não encontrado!\n")
        resultados_vinho_simples = resultados_vinho_completo = algoritmos_vinho = None

    try:
        X_fetal, y_fetal = carregar_dataset('fetal_health.csv', 'fetal_health')
        resultados_fetal_simples, resultados_fetal_completo, algoritmos_fetal = executar_analise_completa_dataset(
            X_fetal, y_fetal, "Saúde Fetal", modelos, otimizador, visualizador, logger
        )
    except FileNotFoundError:
        logger.write("ERRO: Arquivo 'fetal_health.csv' não encontrado!\n")
        resultados_fetal_simples = resultados_fetal_completo = algoritmos_fetal = None

    logger.section("RESUMO FINAL DOS DOIS DATASETS")
    
    if resultados_vinho_simples and resultados_vinho_completo:
        logger.write("\nQUALIDADE DO VINHO:\n")
        melhor_vinho_simples = max(algoritmos_vinho, key=lambda x: np.mean(resultados_vinho_simples[x]['f1_score']))
        melhor_vinho_completo = max(algoritmos_vinho, key=lambda x: np.mean(resultados_vinho_completo[x]['f1_score']))
        
        f1_vinho_simples = np.mean(resultados_vinho_simples[melhor_vinho_simples]['f1_score'])
        f1_vinho_completo = np.mean(resultados_vinho_completo[melhor_vinho_completo]['f1_score'])
        
        logger.write(f"  Melhor (Apenas Normalização): {melhor_vinho_simples} (F1: {f1_vinho_simples:.4f})\n")
        logger.write(f"  Melhor (Processamento Completo): {melhor_vinho_completo} (F1: {f1_vinho_completo:.4f})\n")
        logger.write(f"  Melhoria: {f1_vinho_completo - f1_vinho_simples:+.4f}\n")

    if resultados_fetal_simples and resultados_fetal_completo:
        logger.write("\nSAÚDE FETAL:\n")
        melhor_fetal_simples = max(algoritmos_fetal, key=lambda x: np.mean(resultados_fetal_simples[x]['f1_score']))
        melhor_fetal_completo = max(algoritmos_fetal, key=lambda x: np.mean(resultados_fetal_completo[x]['f1_score']))
        
        f1_fetal_simples = np.mean(resultados_fetal_simples[melhor_fetal_simples]['f1_score'])
        f1_fetal_completo = np.mean(resultados_fetal_completo[melhor_fetal_completo]['f1_score'])
        
        logger.write(f"  Melhor (Apenas Normalização): {melhor_fetal_simples} (F1: {f1_fetal_simples:.4f})\n")
        logger.write(f"  Melhor (Processamento Completo): {melhor_fetal_completo} (F1: {f1_fetal_completo:.4f})\n")
        logger.write(f"  Melhoria: {f1_fetal_completo - f1_fetal_simples:+.4f}\n")

    logger.write(f"\nAnálise completa finalizada com 30 repetições por experimento!\n")
    logger.write(f"Arquivo de saída salvo como: {logger.filename}\n")
    logger.close()


if __name__ == '__main__':
    main()