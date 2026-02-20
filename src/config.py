import yaml
from pathlib import Path

# 1. Define Project Root 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Define Standard Sub-directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 3. The Loader Function
def load_config(config_name="config.yaml"):
    """
    Loads a YAML file from the PROJECT_ROOT.
    Returns a dictionary.
    """
    config_path = PROJECT_ROOT / config_name
    
    if not config_path.exists():
        # Ideally, we create a default one or error out clearly
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(data, config_name="best_params.yaml"):
    """
    Saves a dictionary to a YAML file in PROJECT_ROOT.
    """
    config_path = PROJECT_ROOT / config_name
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)