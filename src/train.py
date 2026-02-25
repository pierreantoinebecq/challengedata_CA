import sys
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from src.config import load_config, DATA_DIR, OUTPUTS_DIR, MODELS_DIR
from src.features import initial_cleaning, get_preprocessor
from src.models import get_model


def main():
    #  1. Load Configuration
    config = load_config("config.yaml")
    cv_folds = config['training']['cv_folds']
    model_name = config['training']['model_name']
    
    try:
        tuned_params = load_config("best_params.yaml")
    except FileNotFoundError:
        tuned_params = {}

    print("Loading Data...")
    df_train = pd.merge(pd.read_csv(DATA_DIR / "x_train.csv", engine="pyarrow"), 
                        pd.read_csv(DATA_DIR / "y_train.csv", engine="pyarrow"))
    df_test = pd.read_csv(DATA_DIR / "x_test.csv", engine="pyarrow")

    target_freq = config['features']['target_freq']
    target_cm = config['features']['target_cm']
    id_col = config['features']['id_col']
    drop_col = [target_freq, target_cm, id_col] + config['features']['drop_col']

    print("Cleaning Data...")
    df_train = initial_cleaning(df_train)
    df_test = initial_cleaning(df_test)

    X = df_train.drop(drop_col, axis=1)
    y_freq = df_train[target_freq]

    mask_sinistres = (df_train[target_freq] > 0) & (df_train[target_cm] > 0)
    X_cm = X[mask_sinistres]
    y_cm = df_train.loc[mask_sinistres, target_cm]
    X_test = df_test.drop([id_col], axis=1)


    # 3. Model Selection 
    print(f"Selected Main Model: {model_name.upper()}")


    # ==========================================================
    # MODEL 1 : FREQUENCE
    # ==========================================================

    print(f"#===# MODEL 1 : FREQ #===#")
    preprocessor_freq = get_preprocessor(X, config)

    # --- BASELINE GLM ---
    print("--- BASELINE GLM : POISSON (FREQ) ---")
    pipeline_baseline_freq = Pipeline([
        ('preprocessor', preprocessor_freq), 
        ('model', get_model("glm_poisson", config['model_params'].get('glm_poisson', {})))
    ])
    cv_baseline_freq = cross_val_score(pipeline_baseline_freq, X, y_freq, cv=cv_folds, scoring='neg_root_mean_squared_error', n_jobs=-1)
    print(f"RMSE Baseline Poisson : {-np.mean(cv_baseline_freq):.4f}")

    # --- MAIN MODEL ---
    
    freq_key = f"{model_name}_freq"
    params_freq = tuned_params.get(freq_key, config['model_params'].get(freq_key, {}))

    pipeline_freq = Pipeline([
        ('preprocessor', preprocessor_freq), 
        ('model', get_model(model_name, params_freq))
    ])

    # Validation 
    print(f"📊 Validating (CV={cv_folds})...")
    
    cv_scores_freq = cross_val_score(
        pipeline_freq, X, y_freq, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1
    )
    print(f"RMSE Main Model: {-np.mean(cv_scores_freq):.4f} (+/- {np.std(cv_scores_freq):.4f})")

    # Training & Saving
    print("Retraining on full data...")
    pipeline_freq.fit(X, y_freq)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ==========================================================
    # MODEL 2 : COUT MOYEN (CM)
    # ==========================================================
    print(f"#===# MODEL 2 : CM #===#")
    preprocessor_cm = get_preprocessor(X_cm, config)

    print("\n--- BASELINE GLM : GAMMA (CM) ---")
    pipeline_baseline_cm = Pipeline([
        ('preprocessor', preprocessor_cm), 
        ('model', get_model("glm_gamma", config['model_params'].get('glm_gamma', {})))
    ])

    cv_baseline_cm = cross_val_score(
        pipeline_baseline_cm, X_cm, y_cm, 
        cv=cv_folds, scoring='neg_root_mean_squared_error', n_jobs=-1
    )

    print(f"RMSE Baseline Gamma : {-np.mean(cv_baseline_cm):.4f}")

    # --- MAIN MODEL ---
    cm_key = f"{model_name}_cm"
    
    params_cm = tuned_params.get(cm_key, config['model_params'].get(cm_key, {}))
    preprocessor = get_preprocessor(X_cm, config)
    
    pipeline_cm = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', get_model(model_name, params_cm))
    ])

    # Validation 
    print(f"Validating (CV={cv_folds})...")
    
    cv_scores_cm = cross_val_score(
        pipeline_cm, X_cm, y_cm, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1
    )
    print(f"RMSE Main Model: {-np.mean(cv_scores_cm):.4f} (+/- {np.std(cv_scores_cm):.4f})")

    # Training & Saving
    print("Retraining on full data...")
    pipeline_cm.fit(X_cm, y_cm)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ==========================================================
    # COMBINING RESULTS & COMPARISON
    # ==========================================================

    print("\n#===# COMBINING RESULTS #===#")

    print("Entraînement des baselines GLM sur l'ensemble des données...")
    pipeline_baseline_freq.fit(X, y_freq)
    pipeline_baseline_cm.fit(X_cm, y_cm)

    # Prédictions XGBoost
    preds_f_xgb = np.clip(pipeline_freq.predict(X), 0, np.inf)
    preds_c_xgb = np.clip(pipeline_cm.predict(X), 0, np.inf)
    charge_xgb = preds_f_xgb * preds_c_xgb * df_train['ANNEE_ASSURANCE']
    rmse_xgb = root_mean_squared_error(df_train['CHARGE'], charge_xgb)

    # Prédictions GLM
    preds_f_glm = np.clip(pipeline_baseline_freq.predict(X), 0, np.inf)
    preds_c_glm = np.clip(pipeline_baseline_cm.predict(X), 0, np.inf)
    charge_glm = preds_f_glm * preds_c_glm * df_train['ANNEE_ASSURANCE']
    rmse_glm = root_mean_squared_error(df_train['CHARGE'], charge_glm)

    print("\n🏆 --- COMPARAISON FINALE SUR LA CHARGE (RMSE) --- 🏆")
    print(f"🔴 Baseline GLM (Poisson/Gamma) : {rmse_glm:,.2f} €")
    print(f"🟢 XGBoost Optimisé             : {rmse_xgb:,.2f} €")
    print(f"Différence absolue              : {rmse_glm - rmse_xgb:,.2f} €")

    
    print("\n Predicting Final CHARGE on Test Set...")
    
    preds_freq = pipeline_freq.predict(X_test)
    preds_cm = pipeline_cm.predict(X_test)
    preds_freq = np.clip(preds_freq, 0, np.inf)
    preds_cm = np.clip(preds_cm, 0, np.inf)
    preds_charge = preds_freq * preds_cm * df_test['ANNEE_ASSURANCE']

    # Création et sauvegarde du fichier de soumission
    submission = pd.DataFrame({
        id_col: df_test[id_col],
        'FREQ': preds_freq, 
        'CM': preds_cm, 
        'ANNEE_ASSURANCE':df_test['ANNEE_ASSURANCE'] , 
        'CHARGE': preds_charge
        })
    
    OUTPUTS_DIR.mkdir(exist_ok=True)
    sub_path = OUTPUTS_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"✅ Soumission sauvegardée avec succès : {sub_path}")

    # Sauvegarde des modèles principaux
    if config['training']['save_model']:
        MODELS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        joblib.dump(pipeline_freq, MODELS_DIR / f"{model_name}_FREQ_{timestamp}.pkl")
        joblib.dump(pipeline_cm, MODELS_DIR / f"{model_name}_CM_{timestamp}.pkl")
        print("💾 Modèles sauvegardés.")

if __name__ == "__main__":
    main()