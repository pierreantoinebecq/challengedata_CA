import sys
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score
import warnings

# Désactivation des warnings de Scikit-Learn (pour un terminal lisible)
#warnings.filterwarnings('ignore')

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from src.config import load_config, save_config, DATA_DIR
from src.features import initial_cleaning, get_preprocessor
from src.models import get_model

def create_objective(X_prep, y_data, model_name, loss_objective, config):
    """
    Générateur d'objectif pour Optuna.
    CRITIQUE : X_prep est DÉJÀ pré-traité. Le CPU a fini son travail.
    """
    def objective(trial):
        if model_name == "xgboost":
            params = {
                'objective': loss_objective, 
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
        else:
            raise ValueError(f"Espace de recherche non défini pour {model_name}")

        model = get_model(model_name, params)
        
        cv_folds = config['training']['cv_folds']
        
        scores = cross_val_score(
            model, X_prep, y_data, 
            cv=cv_folds, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1
        )
        return -np.mean(scores)
        
    return objective


def main():
    CONFIG = load_config("config.yaml")
    
    print("📂 Loading Data for Tuning...")
    df_train = pd.merge(pd.read_csv(DATA_DIR / "x_train.csv", engine="pyarrow"), 
                        pd.read_csv(DATA_DIR / "y_train.csv", engine="pyarrow"))

    # 1. Nettoyage de base (Regex, Thresholds...)
    target_freq = CONFIG['features']['target_freq']
    target_cm = CONFIG['features']['target_cm']
    id_col = CONFIG['features']['id_col']
    cols_to_drop = [target_freq, target_cm, id_col] + CONFIG['features']['drop_col']

    df_train = initial_cleaning(df_train)

    # 2. Séparation Fréquence / Sévérité
    X_global = df_train.drop(columns=cols_to_drop, errors='ignore')
    y_freq = df_train[target_freq]

    mask_sinistres = (df_train[target_freq] > 0) & (df_train[target_cm] > 0)
    X_cm = X_global[mask_sinistres]
    y_cm = df_train.loc[mask_sinistres, target_cm]

    model_name = CONFIG['training']['model_name']
    
    # ========================================================
    # Pré-traitement hors de la boucle
    # ========================================================
    print("\n⏳ Pré-traitement lourd (CPU) en cours... (Cela peut prendre 1 à 2 minutes)")
    
    preprocessor_freq = get_preprocessor(X_global, CONFIG)
    X_global_prep = preprocessor_freq.fit_transform(X_global, y_freq)
    
    preprocessor_cm = get_preprocessor(X_cm, CONFIG)
    X_cm_prep = preprocessor_cm.fit_transform(X_cm, y_cm)
    
    print("✅ Préprocessing fait.")

    # Chargement des anciens paramètres pour ne pas écraser d'autres modèles potentiels
    try:
        best_params_dict = load_config("best_params.yaml")
    except FileNotFoundError:
        best_params_dict = {}

    n_trials = 25

    # ========================================================
    # TUNING 1 : FRÉQUENCE
    # ========================================================
    print(f"\n🔥 Optimisation de {model_name.upper()} pour la FRÉQUENCE (Poisson)...")
    freq_objective = create_objective(X_global_prep, y_freq, model_name, 'count:poisson', CONFIG)
    
    study_freq = optuna.create_study(direction='minimize')
    study_freq.optimize(freq_objective, n_trials=n_trials)
    
    print(f"🏆 Meilleur RMSE Fréquence: {study_freq.best_value:.4f}")
    best_params_dict[f"{model_name}_freq"] = study_freq.best_params
    best_params_dict[f"{model_name}_freq"]['objective'] = 'count:poisson'


    # ========================================================
    # TUNING 2 : SÉVÉRITÉ (CM)
    # ========================================================
    print(f"\n💰 Optimisation de {model_name.upper()} pour le COÛT MOYEN (Gamma)...")
    cm_objective = create_objective(X_cm_prep, y_cm, model_name, 'reg:gamma', CONFIG)
    
    study_cm = optuna.create_study(direction='minimize')
    study_cm.optimize(cm_objective, n_trials=n_trials)
    
    print(f"🏆 Meilleur RMSE Coût: {study_cm.best_value:.4f}")
    best_params_dict[f"{model_name}_cm"] = study_cm.best_params
    best_params_dict[f"{model_name}_cm"]['objective'] = 'reg:gamma'


    # ========================================================
    # SAUVEGARDE FINALE
    # ========================================================
    save_config(best_params_dict, "best_params.yaml")
    print("\n✅ Paramètres optimisés et sauvegardés dans best_params.yaml.")

if __name__ == "__main__":
    main()