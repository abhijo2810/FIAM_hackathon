import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DropAllNaNFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_to_drop_ = X.columns[X.isnull().all()].tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

def load_factors(file_path):
    """Load factor list from a file"""
    with open(file_path, 'r') as f:
        factors = [line.strip() for line in f if line.strip()]
    return factors

def load_data(file_path):
    """Load the stock dataset"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_features(df, factors):
    """Prepare features for prediction"""
    available_factors = [f for f in factors if f in df.columns]
    missing_factors = set(factors) - set(available_factors)
    if missing_factors:
        logging.warning(f"The following factors are not in the dataset and will be excluded: {missing_factors}")
    return df[available_factors]

def predict_returns(pipeline, X):
    """Predict returns using the trained pipeline"""
    return pipeline.predict(X)

def construct_portfolio(df, predicted_returns, n_stocks=50):
    """Construct a long-short portfolio based on predicted returns"""
    df['predicted_return'] = predicted_returns
    df_sorted = df.sort_values('predicted_return', ascending=False)
    
    long_portfolio = df_sorted.head(n_stocks)
    short_portfolio = df_sorted.tail(n_stocks)
    
    portfolio = pd.concat([long_portfolio, short_portfolio])
    portfolio['weight'] = np.where(portfolio.index.isin(long_portfolio.index), 1/n_stocks, -1/n_stocks)
    
    return portfolio

def calculate_portfolio_return(portfolio):
    """Calculate the portfolio return"""
    return (portfolio['stock_exret'] * portfolio['weight']).sum()

def main():
    # Load data and factor list
    df = load_data('hackathon_sample_v2.csv')
    factors = load_factors('factor_char_list.csv')
    
    # Load the best model pipeline
    best_pipeline = joblib.load('best_model_pipeline.joblib')
    
    # Prepare features
    X = prepare_features(df, factors)
    
    # Predict returns using the pipeline
    predicted_returns = predict_returns(best_pipeline, X)
    
    # Construct portfolio
    portfolio = construct_portfolio(df, predicted_returns)
    
    # Calculate portfolio return
    portfolio_return = calculate_portfolio_return(portfolio)
    
    logging.info(f"Portfolio return: {portfolio_return:.4f}")
    
    # Save portfolio
    portfolio.to_csv('portfolio.csv', index=False)
    logging.info("Portfolio saved to 'portfolio.csv'")

if __name__ == "__main__":
    main()