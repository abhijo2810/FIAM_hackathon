import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(file_path):
    """Load the stock dataset"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df

def load_factors(file_path):
    """Load factor list"""
    with open(file_path, 'r') as f:
        factors = [line.strip() for line in f if line.strip()]
    return factors

def analyze_factor_relevance(df, factors, target='stock_exret'):
    """Analyze factor relevance for the target variable"""
    available_factors = [f for f in factors if f in df.columns]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df[available_factors + [target]]),
                              columns=available_factors + [target],
                              index=df.index)
    
    # Calculate correlations
    correlations = df_imputed[available_factors].corrwith(df_imputed[target])
    
    # Calculate mutual information scores
    mi_scores = mutual_info_regression(df_imputed[available_factors], df_imputed[target])
    mi_scores = pd.Series(mi_scores, index=available_factors)
    
    return correlations, mi_scores

def monthly_factor_analysis(df, factors):
    """Perform factor analysis for each month"""
    monthly_correlations = []
    monthly_mi_scores = []
    
    for date, group in tqdm(df.groupby(df['date'].dt.to_period('M')), desc="Analyzing months"):
        correlations, mi_scores = analyze_factor_relevance(group, factors)
        monthly_correlations.append(correlations)
        monthly_mi_scores.append(mi_scores)
    
    return pd.DataFrame(monthly_correlations), pd.DataFrame(monthly_mi_scores)

def aggregate_factor_importance(monthly_correlations, monthly_mi_scores):
    """Aggregate factor importance over the entire period"""
    avg_correlations = monthly_correlations.abs().mean()
    avg_mi_scores = monthly_mi_scores.mean()
    
    combined_importance = (avg_correlations.rank() + avg_mi_scores.rank()) / 2
    return combined_importance.sort_values(ascending=False)

def plot_top_factors(factor_importance, title, filename):
    """Plot top 20 most important factors"""
    plt.figure(figsize=(12, 8))
    factor_importance.head(20).plot(kind='bar')
    plt.title(title)
    plt.xlabel('Factors')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # Load data and factors
    df = load_data('hackathon_sample_v2.csv')
    factors = load_factors('factor_char_list.csv')
    
    # Remove 'variable' from factors if it exists
    if 'variable' in factors:
        factors.remove('variable')
    
    # Perform monthly factor analysis
    monthly_correlations, monthly_mi_scores = monthly_factor_analysis(df, factors)
    
    # Aggregate factor importance
    factor_importance = aggregate_factor_importance(monthly_correlations, monthly_mi_scores)
    
    # Plot and save results
    plot_top_factors(factor_importance, 'Top 20 Factors for Monthly Stock Excess Returns (2000-2023)', 'top_factors.png')
    
    # Save factor importance to CSV
    factor_importance.to_csv('factor_importance_monthly_changes.csv')
    
    print("Top 20 factors influencing monthly stock excess returns:")
    print(factor_importance.head(20))

if __name__ == "__main__":
    main()