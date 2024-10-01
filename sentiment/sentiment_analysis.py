import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import time

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def get_cik_from_ticker(ticker):
    """Get CIK number for a given ticker"""
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&Find=Search&owner=exclude&action=getcompany"
    response = requests.get(url, headers={'User-Agent': 'Your Name yourname@example.com'})
    soup = BeautifulSoup(response.content, 'html.parser')
    cik_re = re.compile(r'CIK=(\d{10})')
    result = cik_re.search(str(soup))
    return result.group(1) if result else None

def get_filings(cik, form_type, start_date, end_date):
    """Get SEC filings for a given CIK and form type"""
    base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        'action': 'getcompany',
        'CIK': cik,
        'type': form_type,
        'dateb': end_date.replace('-', ''),
        'datea': start_date.replace('-', ''),
        'owner': 'exclude',
        'output': 'atom'
    }
    response = requests.get(base_url, params=params, headers={'User-Agent': 'Your Name yourname@example.com'})
    soup = BeautifulSoup(response.content, 'lxml-xml')
    entries = soup.find_all('entry')
    
    filings = []
    for entry in entries:
        filing_date = entry.find('filing-date').text
        filing_url = entry.find('filing-href').text
        filings.append((filing_date, filing_url))
    
    return filings

def get_filing_text(url):
    """Get the text content of a filing"""
    response = requests.get(url, headers={'User-Agent': 'Your Name yourname@example.com'})
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text

def analyze_sentiment(text):
    """Analyze the sentiment of given text using FinBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = probabilities[0][1].item() - probabilities[0][0].item()  # Positive - Negative
    return sentiment_score

def get_sentiment_for_ticker(ticker, start_date, end_date):
    """Get sentiment scores for a ticker's filings"""
    cik = get_cik_from_ticker(ticker)
    if not cik:
        print(f"CIK not found for ticker {ticker}")
        return []

    sentiments = []
    for form_type in ['8-K', '10-K']:
        filings = get_filings(cik, form_type, start_date, end_date)
        for filing_date, filing_url in filings:
            text = get_filing_text(filing_url)
            sentiment = analyze_sentiment(text)
            sentiments.append({
                'stock_ticker': ticker,
                'date': filing_date,
                'form_type': form_type,
                'sentiment': sentiment
            })
        time.sleep(1)  # Be respectful to SEC servers

    return sentiments

def main():
    # Load your stock data
    df = pd.read_csv('data_cleaning/random_ten_sample.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2023-12-31')]

    # Get unique tickers
    tickers = df['stock_ticker'].unique()

    all_sentiments = []

    for ticker in tqdm(tickers, desc="Processing tickers"):
        ticker_sentiments = get_sentiment_for_ticker(ticker, '2000-01-01', '2023-12-31')
        all_sentiments.extend(ticker_sentiments)

    sentiment_df = pd.DataFrame(all_sentiments)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    # Merge sentiment data with original data
    result_df = pd.merge(df, sentiment_df, on=['stock_ticker', 'date'], how='left')

    # Fill NaN sentiments with 0 (neutral)
    result_df['sentiment'] = result_df['sentiment'].fillna(0)

    result_df.to_csv('sentiment/data_with_sentiment.csv', index=False)
    print("Data with sentiment scores saved to 'data_with_sentiment.csv'")

if __name__ == "__main__":
    main()