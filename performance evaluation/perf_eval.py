import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_explore_data(file_path, file_type):
    """Load data and explore its structure"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"{file_type} data shape: {df.shape}")
        logging.info(f"{file_type} columns: {df.columns.tolist()}")
        logging.info(f"{file_type} data types:\n{df.dtypes}")
        logging.info(f"{file_type} first few rows:\n{df.head()}")
        
        if file_type == 'Market' and 'year' in df.columns and 'month' in df.columns:
            df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
            df['date'] = df['date'] + pd.offsets.MonthEnd(0)
            df = df.drop(['year', 'month'], axis=1)
            logging.info("Created 'date' column from 'year' and 'month'")
            logging.info(f"Updated {file_type} columns: {df.columns.tolist()}")
            logging.info(f"Updated {file_type} first few rows:\n{df.head()}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading {file_type} data: {e}")
        return None

def calculate_portfolio_returns(portfolio):
    """Calculate monthly portfolio returns"""
    if 'date' not in portfolio.columns or 'stock_exret' not in portfolio.columns or 'weight' not in portfolio.columns:
        logging.error("Portfolio data is missing required columns (date, stock_exret, or weight)")
        return None
    
    portfolio['date'] = pd.to_datetime(portfolio['date'])
    monthly_returns = portfolio.groupby('date').apply(lambda x: (x['stock_exret'] * x['weight']).sum())
    monthly_returns.name = 'portfolio_return'
    return monthly_returns

def calculate_performance_metrics(portfolio_returns, market_returns, risk_free_rate):
    """Calculate various performance metrics"""
    if len(portfolio_returns) == 0:
        logging.error("Portfolio returns data is empty")
        return None

    excess_returns = portfolio_returns - risk_free_rate
    market_excess_returns = market_returns - risk_free_rate
    
    sharpe_ratio = np.sqrt(12) * excess_returns.mean() / excess_returns.std()
    beta, alpha, _, _, _ = stats.linregress(market_excess_returns, excess_returns)
    alpha *= 12  # Annualize alpha
    
    tracking_error = (excess_returns - market_excess_returns).std() * np.sqrt(12)
    information_ratio = (excess_returns - market_excess_returns).mean() * 12 / tracking_error
    
    cumulative_returns = (1 + excess_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'Sharpe Ratio': sharpe_ratio,
        'Alpha (annualized)': alpha,
        'Beta': beta,
        'Information Ratio': information_ratio,
        'Maximum Drawdown': max_drawdown
    }

def plot_cumulative_returns(portfolio_returns, market_returns, risk_free_rate):
    """Plot cumulative returns of portfolio vs market"""
    excess_returns = portfolio_returns - risk_free_rate
    market_excess_returns = market_returns - risk_free_rate
    
    cumulative_portfolio = (1 + excess_returns).cumprod()
    cumulative_market = (1 + market_excess_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_portfolio.index, cumulative_portfolio, label='Portfolio')
    plt.plot(cumulative_market.index, cumulative_market, label='S&P 500')
    plt.title('Cumulative Returns: Portfolio vs S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_returns.png')
    plt.close()

def main():
    # Load and explore portfolio and market data
    portfolio = load_and_explore_data('portfolio.csv', 'Portfolio')
    market_data = load_and_explore_data('mkt_ind.csv', 'Market')
    
    if portfolio is None or market_data is None:
        logging.error("Failed to load necessary data. Exiting.")
        return
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(portfolio)
    if portfolio_returns is None:
        return
    
    # Ensure necessary columns are present in market data
    required_columns = ['date', 'sp_ret', 'rf']
    if not all(col in market_data.columns for col in required_columns):
        logging.error(f"Market data is missing required columns. Expected: {required_columns}")
        return
    
    # Align dates
    logging.info(f"Portfolio returns date range: {portfolio_returns.index.min()} to {portfolio_returns.index.max()}")
    logging.info(f"Market data date range: {market_data['date'].min()} to {market_data['date'].max()}")
    
    aligned_data = pd.merge(portfolio_returns.reset_index(), market_data, on='date', how='inner')
    
    logging.info(f"Aligned data shape: {aligned_data.shape}")
    logging.info(f"Aligned data date range: {aligned_data['date'].min()} to {aligned_data['date'].max()}")
    
    if aligned_data.empty:
        logging.error("No overlapping dates between portfolio returns and market data")
        return
    
    aligned_data.set_index('date', inplace=True)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(aligned_data['portfolio_return'], aligned_data['sp_ret'], aligned_data['rf'])
    
    if metrics is not None:
        # Print performance metrics
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # Plot cumulative returns
        plot_cumulative_returns(aligned_data['portfolio_return'], aligned_data['sp_ret'], aligned_data['rf'])
        logging.info("Cumulative returns plot saved as 'cumulative_returns.png'")
    else:
        logging.error("Failed to calculate performance metrics")

if __name__ == "__main__":
    main()