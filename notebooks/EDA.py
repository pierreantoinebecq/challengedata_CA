#%%
### 1. IMPORT LIBRARIES
from pathlib import Path 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import re

ROOT = Path(__file__).resolve().parent
# %%
### 2. READ DATA
print(f"Importing files from {ROOT}")
X_raw = pd.read_csv(ROOT / 'x_train.csv')
y_raw = pd.read_csv(ROOT / 'y_train.csv')
X_final = pd.read_csv(ROOT / 'x_test.csv')

def optimize_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            df[col] = df[col].astype('category')
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
    return df
 
X_raw = optimize_memory(X_raw)
y_raw = optimize_memory(y_raw)
X_final = optimize_memory(X_final)

# %%
### FIRST VIEW ###
X_eda = X_raw.copy()
print(X_eda.info())
msno.bar(X_eda)

# %%
### DATA CLEANING ###

# - ID provides no real info
to_drop = ['ID', 'ANNEE_ASSURANCE']

def size_extract(val): 
    #Ignore missing values
    if pd.isna(val) or str(val) == 'nan' or val == 'MISSING' or val == -1:
        return np.nan
    
    #Get the lower / upper bound of the range
    match = re.search(r"\[\s*([\d.kM]+)\s*-\s*([\d.kM+]+)\s*\]?", str(val))
    if not match: 
        return np.nan
    
    low_bound = match[1].strip()
    up_bound = match[2].strip()
    unit_list = {'k': 1000, 'M': 1000000}
    
    unit = low_bound[-1]
    if low_bound == '0':
        low_bound = 0 
    else: 
        low_bound = float(low_bound[:-1])
        low_bound = low_bound * unit_list[unit]

    if '+' in up_bound: 
        return low_bound*1.1 if low_bound else np.nan
    
    else: 
        unit = up_bound[-1]
        up_bound = float(up_bound[:-1])
        up_bound  = up_bound * unit_list[unit]

    avg = (low_bound + up_bound) / 2
    return avg

size_cols = ['TAILLE1', 'TAILLE2']
for col in size_cols : 
    X_eda[col]= X_eda[col].apply(size_extract)

#Transform the thresholds (ex: 01. <= 10)) into numerical values (10)
def get_threshold(val):
    if pd.isna(val) or str(val) == 'nan' or val == 'MISSING' or val == -1:
        return np.nan
    
    match = re.search(r"\.\s*(<=|>=|>|<)\s*([\d.]+)", str(val))

    if match:
        signe = match.group(1)
        nombre = float(match.group(2)) 
        return nombre
    
    else: 
        return np.nan

potential_cols = X_eda.select_dtypes(include=['object', 'category']).columns

# 2. On filtre celles qui contiennent nos signes
cols_to_convert = [
    col for col in potential_cols 
    if X_eda[col].astype(str).str.contains(r'<=|>=|>|<', na=False).any()
]

X_eda[cols_to_convert] = X_eda[cols_to_convert].map(get_threshold)

print(f"Colonnes identifiées : {cols_to_convert}")

# - Remove data with less than 20% of values - 
thresh = 0.2 * len(X_eda)
X_clean = X_eda.dropna(thresh=thresh, axis= 1)
outside_thresh = [c for c in X_eda.columns if c not in X_clean.columns]
print(f"Colonnes quasi-vides à supprimer: {len(outside_thresh)}")
to_drop.extend(outside_thresh)

# - Lists numerical and categorical columns - 
num_cols = X_eda.select_dtypes(include=['number']).columns
cat_cols = X_eda.select_dtypes(include=['category']).columns

# - Inpute values with 0 (Missing = there is 0)
keywords_zero = ['KAPITAL', 'SURFACE', 'NBBAT', 'EQUIPEMENT', 'CA', 'RISK', 'TAILLE', 'NBSIN']
cols_zero = [c for c in num_cols if any(k in c for k in keywords_zero)]
X_eda[cols_zero] = X_eda[cols_zero].fillna(0)

# - Inpute times with the median (Missing = Error in filling the data)
keywords_time = ['ANNEE', 'ANCIENNETE', 'DUREE']
cols_time = [c for c in num_cols if any(k in c for k in keywords_time)]
for c in cols_time:
    X_eda[c] = X_eda[c].fillna(X_eda[c].median())

# - Inpute numerical categories with -1 (Missing = no info)
keywords_code = ['ZONE', 'TYP', 'ID', 'CARACT', 'DEROG']
cols_code = [c for c in num_cols if any(k in c for k in keywords_code)]
X_eda[cols_code] = X_eda[cols_code].fillna(-1)

# - Inpute categorical categories with MISSING (Missing = no info)
X_eda[num_cols] = X_eda[num_cols].fillna(0) # Reste numérique
for col in cat_cols: # Reste catégoriel
    if X_eda[col].isnull().any():
        X_eda[col] = X_eda[col].cat.add_categories(['MISSING']).fillna('MISSING')


# - Removes colums with low variance (no informations)
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_eda[num_cols])
cols_var_keep = num_cols[selector.get_support()]
cols_var_drop = [c for c in num_cols if c not in cols_var_keep]
to_drop.extend(cols_var_drop)
print(f"Colonnes quasi-constantes à supprimer : {len(cols_var_drop)}")

# - Remove colums with high correlation (no additionnal information)
corr_matrix = X_eda[cols_var_keep].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
print(f"Colonnes très corrélées à supprimer : {len(high_corr)}")
to_drop.extend(high_corr)


print(f"{len(to_drop)} colonnes supprimées : ")
print(to_drop)

X_eda = X_eda.drop(to_drop, axis=1)


print(f"Dataset X_eda prêt. Shape: {X_eda.shape}")
# %%
### GLOBAL VISUALIZATION (PCA)


# 1. Préparation (PCA ne mange que du numérique)
X_num_eda = X_eda.select_dtypes(include=['number'])
# On récupère la target (Assure-toi que y_raw a bien une colonne cible, ex: 'CHARGE' ou 'FREQ')
target_col = y_raw.columns[0] if 'ID' not in y_raw.columns else y_raw.columns[1]
y_values = y_raw[target_col]

# 2. Scaling (OBLIGATOIRE pour PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num_eda)

# 3. Projection
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

# 4. Plot
plt.figure(figsize=(10, 8))
plt.scatter(components[:, 0], components[:, 1], c=y_values, cmap='viridis', alpha=0.5, s=2)
plt.colorbar(label='Target Value')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Carte Globale du Dataset (PCA)')
plt.show()

# Ratio de variance expliquée
print(f"Variance expliquée par ces 2 axes : {np.sum(pca.explained_variance_ratio_):.2%}")


# %%
### 4. QUICK FEATURE IMPORTANCE
from sklearn.ensemble import RandomForestRegressor 

rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_num_eda, y_values)

# Visualisation
feat_imp = pd.Series(rf.feature_importances_, index=X_num_eda.columns).sort_values(ascending=False).head(30)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh')
plt.title("Top 30 Variables les plus importantes")
plt.gca().invert_yaxis() # La plus importante en haut
plt.show()
# %%
#PROPORTION, MEN , LOG, IND, ALTITUDE, HAUTEUR, BDTOPP, NB_CASERNES, NBHTX, TAMPLIAB, TAMPLIM, TM, TMM, TMMAX, TMMIN, TN, TNAB, TNMAX, TX, TXAB, TXMIN, NBJFF, NBJFX, FFM,  FXI, FXY, FFM, NBJRR, RR, RRAB

    
