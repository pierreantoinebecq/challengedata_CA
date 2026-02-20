import sys
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from src.config import load_config, DATA_DIR, OUTPUTS_DIR, MODELS_DIR
from src.features import initial_cleaning, get_preprocessor
from src.models import get_model


def main():
    # 1. Load Configuration
    # We load config.yaml first, then overlay best_params.yaml if available
    config = load_config("config.yaml")
    
    # Try to load tuned params, but don't crash if missing
    try:
        tuned_params = load_config("best_params.yaml")
    except FileNotFoundError:
        tuned_params = {}

    print("📂 Loading Data...")
    df_train = pd.merge(pd.read_csv(DATA_DIR / "x_train.csv", engine="pyarrow"), 
                        pd.read_csv(DATA_DIR / "y_train.csv", engine="pyarrow"))
    df_test = pd.read_csv(DATA_DIR / "x_test.csv", engine="pyarrow")

    target_freq = config['features']['target_freq']
    target_cm = config['features']['target_cm']
    id_col = config['features']['id_col']
    drop_col = [target_freq, target_cm, id_col] + config['features']['drop_col']

    df_train = initial_cleaning(df_train)
    df_test = initial_cleaning(df_test)

    X = df_train.drop(drop_col, axis=1)
    y_freq = df_train[target_freq]
    y_cm = df_train[target_cm]
    X_test = df_test.drop([id_col], axis=1)



    # 3. Model Selection 
    model_name = config['training']['model_name']
    print(f"⚙️  Selected Model: {model_name.upper()}")

    # Hierarchy: Tuned Params > Config Params > Defaults
    # Check if we have tuned params for THIS specific model
    if model_name in tuned_params:
        print(f"✨ Found tuned parameters for {model_name}!")
        final_params = tuned_params[model_name]
    else:
        print(f"⚠️  No tuned params found. Using defaults from config.")
        final_params = config['model_params'].get(model_name, {})


#=================# MODEL FREQ #=================#
    print(f"#===# MODEL 1 : FREQ #===#")

    # Build Pipeline
    preprocessor = get_preprocessor(X, config)
    model = get_model(model_name, final_params)
    
    pipeline_freq = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    # Validation (Using Config CV)
    cv_folds = config['training']['cv_folds']
    print(f"📊 Validating (CV={cv_folds})...")
    
    cv_scores_freq = cross_val_score(
        pipeline_freq, X, y_freq, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1
    )
    print(f"✅ RMSE: {-np.mean(cv_scores_freq):.4f} (+/- {np.std(cv_scores_freq):.4f})")

    # Training & Saving
    print("🚀 Retraining on full data...")
    pipeline_freq.fit(X, y_freq)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if config['training']['save_model']:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}_FREQ_{timestamp}.pkl"
        joblib.dump(pipeline_freq, model_path)
        print(f"💾 Model saved to {model_path}")

#=================# MODEL CM #=================#

    print(f"#===# MODEL 2 : CM #===#")

     # Build Pipeline
    mask_sinistres = df_train[target_freq] > 0
    X_cm = X[mask_sinistres]
    y_cm = df_train.loc[mask_sinistres, target_cm]

    preprocessor = get_preprocessor(X_cm, config)
    model = get_model(model_name, final_params)
    
    pipeline_cm = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    # Validation (Using Config CV)
    cv_folds = config['training']['cv_folds']
    print(f"📊 Validating (CV={cv_folds})...")
    
    cv_scores_cm = cross_val_score(
        pipeline_cm, X_cm, y_cm, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1
    )
    print(f"✅ RMSE: {-np.mean(cv_scores_cm):.4f} (+/- {np.std(cv_scores_cm):.4f})")

    # Training & Saving
    print("🚀 Retraining on full data...")
    pipeline_cm.fit(X_cm, y_cm)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if config['training']['save_model']:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}_CM_{timestamp}.pkl"
        joblib.dump(pipeline_cm, model_path)
        print(f"💾 Model saved to {model_path}")

    print(f"#===# COMBINING RESULTS #===#")

    preds_freq_train = pipeline_freq.predict(X)
    preds_cm_train = pipeline_cm.predict(X)
    preds_charge_train = preds_cm_train * preds_freq_train * df_train['ANNEE_ASSURANCE']
    rmse_train = root_mean_squared_error(preds_charge_train, df_train['CHARGE'])
    print(f"RMSE : {rmse_train}")
    
    
    print("\n Predicting Final CHARGE on Test Set...")
    

    preds_freq = pipeline_freq.predict(X_test)
    preds_cm = pipeline_cm.predict(X_test)
    preds_charge = preds_freq * preds_cm * df_test['ANNEE_ASSURANCE']
    preds_charge = np.clip(preds_charge, 0, np.inf)

if __name__ == "__main__":
    main()