import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2023-12-31')]
    return df

def load_factors(file_path):
    with open(file_path, 'r') as f:
        factors = [line.strip() for line in f if line.strip()]
    return factors

def print_data_info(df):
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)

def calculate_factor_importance(df, factors, target='stock_exret'):
    available_factors = [f for f in factors if f in df.columns]
    print(f"\nNumber of available factors: {len(available_factors)}")
    
    X = df[available_factors]
    y = df[target]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    mi_scores = mutual_info_regression(X_scaled, y)
    factor_importance = pd.Series(mi_scores, index=available_factors).sort_values(ascending=False)
    return factor_importance

def main():
    df = load_data('data_cleaning/random_ten_sample.csv')
    factors = load_factors('factor analysis/factor_char_list.csv')
    
    print_data_info(df)
    
    factor_importance = calculate_factor_importance(df, factors)
    print("\nTop 20 most important factors:")
    print(factor_importance.head(20))
    factor_importance.to_csv('factor analysis/factor_importance.csv')

if __name__ == "__main__":
    main()