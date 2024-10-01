import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_portfolio(file_path):
    df = pd.read_csv(file_path)
    print(f"Portfolio data shape: {df.shape}")
    print(f"Portfolio columns: {df.columns.tolist()}")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2023-12-31')]
    else:
        print("Warning: 'date' column not found in portfolio data")
    return df

def load_market_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Market data shape: {df.shape}")
    print(f"Market data columns: {df.columns.tolist()}")
    if 'date' not in df.columns:
        if 'year' in df.columns and 'month' in df.columns:
            print("Creating 'date' column from 'year' and 'month'")
            df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
        else:
            print("Error: Unable to create 'date' column. 'year' and 'month' columns not found.")
            return None
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2023-12-31')]
    return df

def calculate_portfolio_returns(portfolio):
    if 'weight' not in portfolio.columns or 'stock_exret' not in portfolio.columns:
        print("Error: 'weight' or 'stock_exret' column not found in portfolio data")
        return None
    returns = portfolio.groupby('date').apply(lambda x: (x['stock_exret'] * x['weight']).sum())
    returns.name = 'portfolio_return'
    return returns.reset_index()

def calculate_performance_metrics(portfolio_returns, market_returns, risk_free_rate):
    excess_returns = portfolio_returns - risk_free_rate
    market_excess_returns = market_returns - risk_free_rate
    
    sharpe_ratio = np.sqrt(12) * excess_returns.mean() / excess_returns.std()
    beta, alpha, _, _, _ = stats.linregress(market_excess_returns, excess_returns)
    alpha *= 12  # Annualize alpha
    tracking_error = (excess_returns - market_excess_returns).std() * np.sqrt(12)
    information_ratio = (excess_returns - market_excess_returns).mean() * 12 / tracking_error
    max_monthly_loss = portfolio_returns.min()
    cumulative_returns = (1 + excess_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'Sharpe Ratio': sharpe_ratio,
        'Alpha (annualized)': alpha,
        'Beta': beta,
        'Information Ratio': information_ratio,
        'Maximum one-month loss': max_monthly_loss,
        'Maximum Drawdown': max_drawdown
    }

def calculate_turnover(portfolio):
    monthly_holdings = portfolio.groupby('date')['weight'].apply(lambda x: x.abs().sum()) / 2
    turnover = monthly_holdings.diff().abs().mean() * 12  # Annualized turnover
    return turnover

def plot_cumulative_returns(portfolio_returns, market_returns, risk_free_rate):
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
    plt.savefig('performance evaluation/cumulative_returns.png')
    plt.close()

def plot_metric(metric_name, value):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[metric_name], y=[value])
    plt.title(f'{metric_name}: {value:.4f}')
    plt.savefig(f'{metric_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_all_metrics(metrics):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title('Performance Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('performance evaluation/all_metrics.png')
    plt.close()

def main():
    portfolio = load_portfolio('portfolio/portfolio.csv')
    market_data = load_market_data('data_cleaning/mkt_ind.csv')
    
    if market_data is None:
        print("Error: Failed to load market data")
        return
    
    portfolio_returns = calculate_portfolio_returns(portfolio)
    if portfolio_returns is None:
        print("Error: Failed to calculate portfolio returns")
        return
    
    print("Portfolio returns shape:", portfolio_returns.shape)
    print("Market data shape:", market_data.shape)
    
    aligned_data = pd.merge(portfolio_returns, market_data, on='date', how='inner')
    print("Aligned data shape:", aligned_data.shape)
    print("Aligned data columns:", aligned_data.columns.tolist())
    
    if 'sp_ret' not in aligned_data.columns or 'rf' not in aligned_data.columns:
        print("Error: Required columns not found in market data")
        print("Available columns:", aligned_data.columns.tolist())
        return
    
    metrics = calculate_performance_metrics(aligned_data['portfolio_return'], aligned_data['sp_ret'], aligned_data['rf'])
    turnover = calculate_turnover(portfolio)
    metrics['Turnover (annualized)'] = turnover
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        plot_metric(metric, value)
    
    plot_all_metrics(metrics)
    plot_cumulative_returns(aligned_data['portfolio_return'], aligned_data['sp_ret'], aligned_data['rf'])

if __name__ == "__main__":
    main()