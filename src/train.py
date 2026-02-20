import sys
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from src.config import load_config, DATA_DIR, OUTPUTS_DIR, MODELS_DIR
from src.features import create_features, get_preprocessor
from src.models import get_model
from src.utils import log_experiment

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
    df_train = pd.merge(pd.read_csv(DATA_DIR / "x_train.csv"), 
                        pd.read_csv(DATA_DIR / "y_train.csv"))
    df_test = pd.read_csv(DATA_DIR / "x_test.csv")

    # 2. Feature Engineering


    target_col = config['features']['target']
    id_col = config['features']['id_col']
    drop_col = config['features']['drop_col']

    X = df_train.drop([target_col, id_col, drop_col], axis=1)
    y = df_train[target_col]
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

    # 4. Build Pipeline
    preprocessor = get_preprocessor(X, config)
    model = get_model(model_name, final_params)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    # 5. Validation (Using Config CV)
    cv_folds = config['training']['cv_folds']
    print(f"📊 Validating (CV={cv_folds})...")
    
    cv_scores = cross_val_score(
        full_pipeline, X, y, 
        cv=cv_folds, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1
    )
    print(f"✅ RMSE: {-np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # 6. Training & Saving
    print("🚀 Retraining on full data...")
    full_pipeline.fit(X, y)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if config['training']['save_model']:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}_{timestamp}.pkl"
        joblib.dump(full_pipeline, model_path)
        print(f"💾 Model saved to {model_path}")

    joblib.dump(full_pipeline, MODELS_DIR / f"{model_name}_{timestamp}.pkl")

    # Log to CSV
    metrics = {"cv_rmse": -np.mean(cv_scores), "std_rmse": np.std(cv_scores)}
    log_experiment(OUTPUTS_DIR, model_name, final_params, metrics)
    print("✅ Experiment logged.")

    # 7. Submission
    preds = full_pipeline.predict(X_test)
    preds = np.clip(preds, 0, 100) 
    
    submission = pd.DataFrame({id_col: df_test[id_col], target_col: preds})
    OUTPUTS_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUTS_DIR / "submission.csv", index=False)
    print(f"📝 Submission saved to {OUTPUTS_DIR}")

if __name__ == "__main__":
    main()