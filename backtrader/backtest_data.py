import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Load the factors
def load_factors(file_path):
    with open(file_path, 'r') as f:
        factors = [line.strip() for line in f if line.strip()]
    return factors

# Prepare features for the asset pricing model
def prepare_features(df, factors):
    X = df[factors]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=factors, index=df.index)

# Load the sentiment data
sentiment_df = pd.read_csv('sentiment/data_with_sentiment.csv')
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df.set_index('date', inplace=True)

# Load the original stock data
stock_df = load_data('data_cleaning/random_ten_sample.csv')
stock_df.set_index('date', inplace=True)

# Load factors
factors = load_factors('factor analysis/factor_char_list.csv')

# Prepare features for prediction
X = prepare_features(stock_df, factors)

# Load the trained model
model = joblib.load('model selection/best_model.joblib')

# Predict returns
predicted_returns = model.predict(X)

# Add predicted returns to the stock dataframe
stock_df['predicted_return'] = predicted_returns

# Merge stock data with sentiment data
combined_df = stock_df.merge(sentiment_df[['sentiment']], left_index=True, right_index=True, how='left')

# Fill NaN values in sentiment with 0 (neutral)
combined_df['sentiment'] = combined_df['sentiment'].fillna(0)

# Make sure we have all required columns
required_columns = ['close', 'sentiment', 'predicted_return', 'stock_exret']
for col in required_columns:
    if col not in combined_df.columns:
        print(f"Warning: {col} column is missing from the dataset")

# Add S&P 500 returns (you may need to adjust this based on your data)
market_data = pd.read_csv('data_cleaning/mkt_ind.csv')
market_data['date'] = pd.to_datetime(market_data['date'])
market_data.set_index('date', inplace=True)
combined_df = combined_df.merge(market_data[['sp_ret']], left_index=True, right_index=True, how='left')

# Sort the dataframe by date
combined_df.sort_index(inplace=True)

# Save the combined dataframe
combined_df.to_csv('backtrader/data_with_sentiment_and_predicted_returns.csv')

print("Data saved to 'data_with_sentiment_and_predicted_returns.csv'")
print("Dataframe shape:", combined_df.shape)
print("Columns:", combined_df.columns.tolist())
combined_df.head()