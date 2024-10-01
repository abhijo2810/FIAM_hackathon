import pandas as pd
import numpy as np
import backtrader as bt
import quantstats as qs
import matplotlib.pyplot as plt
from datetime import datetime

class AssetPricingSentimentStrategy(bt.Strategy):
    params = (
        ('sentiment_threshold', 0),
        ('return_threshold', 0),
    )

    def __init__(self):
        self.sentiment = self.datas[0].sentiment
        self.predicted_return = self.datas[0].predicted_return
        self.close = self.datas[0].close

    def next(self):
        if not self.position:
            if (self.sentiment[0] > self.params.sentiment_threshold and 
                self.predicted_return[0] > self.params.return_threshold):
                self.buy()
        elif (self.sentiment[0] < -self.params.sentiment_threshold or 
              self.predicted_return[0] < -self.params.return_threshold):
            self.close()

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

class AssetPricingSentimentData(bt.feeds.PandasData):
    lines = ('sentiment', 'predicted_return',)
    params = (('sentiment', -1), ('predicted_return', -1),)

def run_backtest(data):
    cerebro = bt.Cerebro()
    
    data_feed = AssetPricingSentimentData(dataname=data)
    cerebro.adddata(data_feed)
    
    cerebro.addstrategy(AssetPricingSentimentStrategy)
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    return results[0]

def plot_results(cerebro):
    plt.figure(figsize=(12, 8))
    cerebro.plot(style='candlestick')
    plt.savefig('backtest_results.png')
    plt.close()

def run_quantstats_analysis(returns, benchmark_returns):
    qs.extend_pandas()
    
    qs.reports.html(returns, benchmark_returns, output='quantstats_report.html')
    
    qs.plots.returns(returns, save='returns.png')
    qs.plots.yearly_returns(returns, save='yearly_returns.png')
    qs.plots.monthly_heatmap(returns, save='monthly_heatmap.png')
    qs.plots.drawdown(returns, save='drawdown.png')
    qs.plots.drawdowns_periods(returns, save='drawdown_periods.png')
    qs.plots.rolling_beta(returns, benchmark_returns, save='rolling_beta.png')
    qs.plots.rolling_volatility(returns, save='rolling_volatility.png')
    qs.plots.rolling_sharpe(returns, save='rolling_sharpe.png')

def main():
    # Load data
    data = load_data('data_with_sentiment_and_predicted_returns.csv')
    
    # Run backtest
    results = run_backtest(data)
    
    # Plot results
    plot_results(results.cerebro)
    
    # Extract returns
    portfolio_value = results.cerebro.broker.getvalue()
    returns = pd.Series(results._trades).pct_change()
    
    # Load benchmark data (assuming S&P 500 returns are in the same file)
    benchmark_returns = data['sp_ret']
    
    # Run Quantstats analysis
    run_quantstats_analysis(returns, benchmark_returns)

if __name__ == "__main__":
    main()