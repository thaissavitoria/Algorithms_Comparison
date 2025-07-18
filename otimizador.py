import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xanfis import AnfisClassifier

class OtimizadorHiperparametros:
    def __init__(self, logger):
        self.logger = logger

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
        self.logger.write(f"\nOtimizando {nome_modelo} com Optuna...\n")

        def objective(trial):
            modelo = self._definir_espaco_de_busca(trial, nome_modelo)
            return cross_val_score(modelo, X, y, scoring='f1_macro', cv=3).mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=15, n_jobs=4)

        melhor_trial = study.best_trial
        self.logger.write(f"Melhor score: {melhor_trial.value:.4f}\n")
        self.logger.write(f"Melhores parâmetros: {melhor_trial.params}\n")

        melhor_modelo = self._definir_espaco_de_busca(study.best_trial, nome_modelo)
        return melhor_modelo, melhor_trial.params, melhor_trial.value