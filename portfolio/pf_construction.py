import pandas as pd
import numpy as np
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2023-12-31')]
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    return df

def prepare_features(df, factors):
    available_factors = [f for f in factors if f in df.columns]
    print(f"Number of available factors: {len(available_factors)}")
    print(f"Factors not found in the dataset: {set(factors) - set(available_factors)}")
    
    if 'sentiment' in df.columns:
        available_factors.append('sentiment')
    else:
        print("Warning: 'sentiment' column not found in the dataset")
    
    return df[available_factors]

def predict_returns(model, X):
    return model.predict(X)

def construct_portfolio(df, predicted_returns, n_stocks=50):
    df['predicted_return'] = predicted_returns
    df_sorted = df.sort_values('predicted_return', ascending=False)
    
    long_portfolio = df_sorted.head(n_stocks)
    short_portfolio = df_sorted.tail(n_stocks)
    
    portfolio = pd.concat([long_portfolio, short_portfolio])
    portfolio['weight'] = np.where(portfolio.index.isin(long_portfolio.index), 1/n_stocks, -1/n_stocks)
    
    return portfolio

def main():
    df = load_data('sentiment/data_with_sentiment.csv')
    with open('factor analysis/factor_char_list.csv', 'r') as f:
        factors = [line.strip() for line in f if line.strip()]
    
    X = prepare_features(df, factors)
    
    try:
        best_model = joblib.load('model selection/best_model.joblib')
    except FileNotFoundError:
        print("Error: 'best_model.joblib' not found. Please run model selection first.")
        return
    
    predicted_returns = predict_returns(best_model, X)
    
    portfolio = construct_portfolio(df, predicted_returns)
    
    portfolio.to_csv('portfolio/portfolio.csv', index=False)
    print("Portfolio saved to 'portfolio.csv'")
    print(f"Portfolio shape: {portfolio.shape}")
    print(f"Columns in the portfolio: {portfolio.columns.tolist()}")

if __name__ == "__main__":
    main()