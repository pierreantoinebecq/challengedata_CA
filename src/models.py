# src/models.py
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

def get_xgb_model(params=None):
    if params is None: params = {}
    return XGBRegressor(
        random_state=42, 
        n_jobs=-1,
        **params)

def get_ridge_model(params=None):
    if params is None: params = {}
    return Ridge(random_state=42, **params)

def get_model(model_name, params=None):
    """
    L'Usine à modèles.
    """
    if model_name == "xgboost":
        return get_xgb_model(params)
    elif model_name == "ridge":
        return get_ridge_model(params)
    else:
        raise ValueError(f"Erreur fatale : Le modèle '{model_name}' n'est pas codé dans models.py.")