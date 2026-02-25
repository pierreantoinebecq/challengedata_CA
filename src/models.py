from xgboost import XGBRegressor
from sklearn.linear_model import PoissonRegressor, GammaRegressor

def get_xgb_model(params=None):
    if params is None: params = {}
    return XGBRegressor(random_state=42, n_jobs=-1, **params)

def get_glm_poisson(params=None):
    if params is None: params = {}
    return PoissonRegressor(**params)

def get_glm_gamma(params=None):
    if params is None: params = {}
    return GammaRegressor(**params)

def get_model(model_name, params=None):
    if model_name == "xgboost":
        return get_xgb_model(params)
    elif model_name == "glm_poisson":
        return get_glm_poisson(params)
    elif model_name == "glm_gamma":
        return get_glm_gamma(params)
    else:
        raise ValueError(f"Erreur fatale : Le modèle '{model_name}' n'est pas codé.")