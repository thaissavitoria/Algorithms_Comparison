from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PreProcessador:
    def __init__(self, logger):
        self.normalizador = RobustScaler()
        self.seletor = None
        self.balanceador = SMOTE(random_state=42)
        self.nomes_caracteristicas = None
        self.rf_info = None
        self.logger = logger

    def normalizar(self, X):
        self.logger.write("Normalizando dados...\n")
        X_normalizado = self.normalizador.fit_transform(X)
        self.logger.write(f"Dados normalizados: {X_normalizado.shape}\n")
        return X_normalizado

    def analisar_rf_para_selecao_caracteristicas(self, X_normalizado, y, plot=True):
        self.logger.section("Análise Random Forest para Seleção de Características", level=2)
        
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
        
        self.logger.write(f"Número total de características: {len(importancias)}\n")
        self.logger.write(f"\nTop 10 características mais importantes:\n")
        for i in range(min(10, len(feature_importance_df))):
            self.logger.write(f"  {i+1}. {feature_importance_df.iloc[i]['caracteristica']}: {feature_importance_df.iloc[i]['importancia']:.4f}\n")
    
        importancia_cumulativa = np.cumsum(feature_importance_df['importancia'].values)
        
        perc_importancia = [0.80, 0.85, 0.90, 0.95, 0.99]
        self.logger.write(f"\nAnálise por percentual de importância total:\n")
        
        perc_results = {}
        for perc in perc_importancia:
            n_comp = np.argmax(importancia_cumulativa >= perc) + 1
            n_comp = min(n_comp, len(importancias))
            perc_results[perc] = n_comp
            reduction = len(importancias) - n_comp
            reduction_pct = (reduction / len(importancias)) * 100
            self.logger.write(f"  {perc*100:.0f}%: {n_comp} características (redução: {reduction} = {reduction_pct:.1f}%)\n")
        
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
        
        self.logger.write(f"\nPonto de cotovelo detectado: {ponto_cotovelo} características\n")
        
        min_features = max(3, int(len(importancias) * 0.15)) 
        max_features = int(len(importancias) * 0.85)
        
        # Usar a melhor estratégia entre 80% de importância e ponto de cotovelo
        n_recomendado = perc_results.get(0.80, ponto_cotovelo)
        if n_recomendado is None:
            n_recomendado = ponto_cotovelo
            
        n_recomendado = max(n_recomendado, min_features)
        n_recomendado = min(n_recomendado, max_features, len(importancias))
        
        self.logger.write(f"\nESTRATÉGIA DE RECOMENDAÇÃO:\n")
        self.logger.write(f"  80% importância total: {perc_results.get(0.80, 'N/A')}\n")
        self.logger.write(f"  Ponto de cotovelo: {ponto_cotovelo}\n")
        self.logger.write(f"  Mínimo permitido (15%): {min_features}\n")
        self.logger.write(f"  Máximo permitido (85%): {max_features}\n")
        self.logger.write(f"  RECOMENDAÇÃO FINAL: {n_recomendado}\n")

        
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
    
    def processar_labels(self, y):
        unique_classes = np.unique(y)
        self.logger.write(f"Classes originais: {unique_classes}\n")

        if min(unique_classes) != 0:
            # Classes devem ser convertidas para 0, 1, 2 porque o anfis só aceita assim
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self.logger.write(f"Classes convertidas: {np.unique(y)}\n")

        return y
    
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

        self.logger.subsection("Resumo da Análise Random Forest")
        total_features = len(self.rf_info['importancias'])
        self.logger.write(f"Características originais: {total_features}\n")
        self.logger.write(f"Características recomendadas: {self.rf_info['n_recomendado']}\n")
        self.logger.write(f"Redução: {total_features - self.rf_info['n_recomendado']} características\n")
        self.logger.write(f"Percentual de redução: {((total_features - self.rf_info['n_recomendado'])/total_features*100):.1f}%\n")
        
        top_features = self.rf_info['feature_importance_df'].head(self.rf_info['n_recomendado'])
        self.logger.write(f"\nCaracterísticas que seriam mantidas (top {self.rf_info['n_recomendado']}):\n")
        for i, row in top_features.iterrows():
            self.logger.write(f"  {row['caracteristica']}: {row['importancia']:.4f}\n")

    def ajustar_e_transformar(self, X, y):
        self.logger.write(f"Iniciando pré-processamento:\n")

        if hasattr(X, 'columns'):
            self.nomes_caracteristicas = X.columns.tolist()
            X = X.values
        else:
            self.nomes_caracteristicas = [f'caracteristica_{i}' for i in range(X.shape[1])]

        X_normalizado = self.normalizar(X)
        self.logger.write(f"Dados normalizados: {X_normalizado.shape}\n")

        self.logger.write("\nUsando análise Random Forest para determinar número ideal de características...\n")
        n_caracteristicas_rf = self.analisar_rf_para_selecao_caracteristicas(X_normalizado, y)
        n_caracteristicas = n_caracteristicas_rf
        self.logger.write(f"\nNúmero de características definido pelo Random Forest: {n_caracteristicas}\n")

        self.seletor = SelectKBest(score_func=f_classif, k=n_caracteristicas)
        X_selecionado = self.seletor.fit_transform(X_normalizado, y)

        indices_selecionados = self.seletor.get_support(indices=True)
        self.nomes_caracteristicas = [self.nomes_caracteristicas[i] for i in indices_selecionados]
        self.logger.write(f"Características selecionadas: {X_selecionado.shape[1]}\n")
        self.logger.write(f"Características mantidas: {self.nomes_caracteristicas}\n")

        X_balanceado, y_balanceado = self.balanceador.fit_resample(X_selecionado, y)
        self.logger.write(f"Classes balanceadas: {X_balanceado.shape}\n")

        unicos, contagens = np.unique(y_balanceado, return_counts=True)
        self.logger.write("Distribuição após balanceamento:\n")
        for classe, contagem in zip(unicos, contagens):
            self.logger.write(f" Classe {classe}: {contagem} amostras\n")

        return X_balanceado, y_balanceado