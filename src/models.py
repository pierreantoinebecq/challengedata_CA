# src/models.py
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

def get_xgb_model(params=None):
    if params is None: params = {}
    # On force des defaults solides ici (reproductibilité + vitesse)
    return XGBRegressor(n_jobs=-1, random_state=42, objective='reg:squarederror', **params)


def get_ridge_model(params=None):
    if params is None: params = {}
    return Ridge(random_state=42, **params)

# --- LA FONCTION MAGIQUE : LE DISPATCHER ---
def get_model(model_name, params=None):
    """
    C'est le serveur. Tu lui demandes un plat, il t'apporte le bon objet.
    """
    if model_name == "xgboost":
        return get_xgb_model(params)
    elif model_name == "ridge":
        return get_ridge_model(params)
    else:
        raise ValueError(f"Modèle '{model_name}' inconnu au bataillon !")