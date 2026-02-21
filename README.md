# AssurPrime : Prédiction de Prime d'Assurance Incendie

Ce projet a été développé dans le cadre du Challenge Data Crédit Agricole Assurances. L'objectif est de prédire la charge sinistre incendie pour les contrats Multirisque Agricole en combinant deux modèles distincts : un modèle de Fréquence et un modèle de Coût Moyen (CM).
## 🚀 Architecture du Modèle

Le projet repose sur une approche actuarielle classique de décomposition de la prime pure :
Charge = Fréquence × Coût Moyen × Année Assurance

- Modèle de Fréquence : Régression XGBoost utilisant une loi de Poisson (count:poisson).

- Modèle de Sévérité (CM) : Régression XGBoost utilisant une loi Gamma (reg:gamma), entraînée uniquement sur les sinistres positifs.

## 📁 Structure du Projet

    pierreantoinebecq-challengedata_ca/
    ├── config.yaml             # Configuration globale (colonnes, preprocessing, modèles)
    ├── best_params.yaml        # Hyperparamètres optimisés par Optuna
    ├── notebooks/
    │   └── EDA.py              # Analyse exploratoire et recherche de features
    └── src/
        ├── config.py           # Utilitaire de chargement de la configuration
        ├── features.py         # Pipeline de nettoyage (regex, types) et preprocessing
        ├── models.py           # Factory pour l'instanciation des modèles (XGBoost, Ridge)
        ├── train.py            # Script principal d'entraînement et de prédiction
        └── tune.py             # Optimisation des hyperparamètres avec Optuna

## 🛠️ Installation

Python 3.9+

Installez les dépendances nécessaires :
     
     pip install pandas numpy xgboost scikit-learn optuna pyyaml joblib matplotlib seaborn missingno pyarrow

## ⚙️ Utilisation
### 1. Analyse Exploratoire (EDA)

Le script notebooks/EDA.py permet d'analyser la structure des données, de visualiser les valeurs manquantes et d'identifier les variables les plus importantes via une PCA et une Random Forest rapide.
### 2. Optimisation des Hyperparamètres

Pour lancer une recherche des meilleurs paramètres via Optuna (peut prendre plus d'une heure en fonction de votre machine) :

    python src/tune.py

Cela mettra à jour le fichier best_params.yaml avec les meilleurs réglages pour les modèles de fréquence et de coût moyen.
### 3. Entraînement et Soumission

Pour entraîner les modèles finaux et générer le fichier de soumission (outputs/submission.csv) :

    python src/train.py

### 🔍 Détails du Preprocessing

Le pipeline de données (src/features.py) inclut des traitements spécifiques aux données d'assurance :

- Nettoyage Regex : Extraction automatique des valeurs numériques depuis des chaînes de caractères (ex: [10k - 50k] devient 30000).

- Gestion des seuils : Conversion des catégories textuelles (ex: <= 10) en valeurs numériques.

- Imputation intelligente :

      - 0 pour les variables liées aux capitaux ou surfaces manquantes.

      - Médiane pour les variables temporelles.

      -  Target Encoding pour les variables catégorielles à haute cardinalité.

  - Optimisation mémoire : Conversion des types de données pour réduire l'empreinte RAM.

### 📊 Évaluation

La performance est mesurée par la RMSE (Root Mean Square Error) entre la charge réelle et la charge prédite sur l'ensemble de test.
