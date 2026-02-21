import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
import re

def initial_cleaning(df): 
    df_clean = df.copy()

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

    def get_threshold(val):
        """
        Transform the thresholds (ex: 01. <= 10)) into numerical values (10,0)
        """
        if pd.isna(val) or str(val) == 'nan' or val == 'MISSING' or val == -1:
            return np.nan
        
        match = re.search(r"\.\s*(<=|>=|>|<)\s*([\d.]+)", str(val))

        if match:
            signe = match.group(1)
            nombre = float(match.group(2)) 
            return nombre
        
        else: 
            return np.nan
    
    df_clean = optimize_memory(df_clean)

    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    size_cols = [c for c in num_cols if any(k in c for k in 'TAILLE')]
    df_clean[size_cols] = df_clean[size_cols].map(size_extract)


    cols_to_convert = [
        col for col in cat_cols 
        if df_clean[col].astype(str).str.contains(r'<=|>=|>|<', na=False).any()]
    df_clean[cols_to_convert] = df_clean[cols_to_convert].map(get_threshold)

    return df_clean

def get_preprocessor(X, config): 
    num_strat = config['preprocessing']['numerical']['imputer']
    cat_strat = config['preprocessing']['categorical']['imputer']
    scaler_type = config['preprocessing']['numerical']['scaler']
    encoder_type = config['preprocessing']['categorical']['encoding']

    if scaler_type == 'robust':
            scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=num_strat)),
        ('scaler', scaler)
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=cat_strat)),
        ('target_encode', TargetEncoder(target_type='continuous', cv=5, smooth="auto"))
        ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    return preprocessor