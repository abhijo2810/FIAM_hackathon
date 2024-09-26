import pandas as pd
import numpy as np
from secedgar import FilingType, CompanyFilings
from secedgar.client import NetworkClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import tempfile
import os
from collections import defaultdict
import yfinance as yf
from sec_api import MappingApi
from sec_api import QueryApi

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mehassan/finbert-finetuned")
model = AutoModelForSequenceClassification.from_pretrained("mehassan/finbert-finetuned")

def load_stock_data(file_path):
    """Load the stock dataset with error handling"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded CSV file from {file_path}")
        logging.info(f"Columns in the CSV: {df.columns.tolist()}")
        
        if 'date' not in df.columns:
            logging.error("No 'date' column found in the CSV file")
            return None
        
        logging.info(f"Sample of date values: {df['date'].head().tolist()}")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        null_dates = df['date'].isnull().sum()
        if null_dates > 0:
            logging.warning(f"{null_dates} rows have invalid date formats")
        
        logging.info(f"Date range in the data: {df['date'].min()} to {df['date'].max()}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading stock data: {str(e)}")
        return None

# Set your SEC API key here
SEC_API_KEY = 'c4aecc42414d4602d1f48ada754e63907f506c2c77fae0c1a99191bc209f99d6'  # Replace with your actual SEC API key

def get_sec_filings(ticker, start_date, end_date):
    """Fetch SEC 8-K filings using sec-api and organize by month"""
    monthly_filings = defaultdict(list)
    
    query_api = QueryApi(api_key=SEC_API_KEY)

    try:
        query = {
            "query": {
                "query_string": {
                    "query": f"ticker:{ticker} AND formType:8-K"
                }
            },
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
            "size": 100  # Adjust size as needed
        }
        
        response = query_api.get_filings(query)

        for filing in response['filings']:
            filing_date = datetime.strptime(filing['filedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
            month_key = filing_date.strftime('%Y-%m')

            # Extract event date and filing type (customize extraction logic)
            event_date = extract_event_date(filing['description'])  # Adjust as necessary
            filing_type = extract_filing_type(filing['description'])  # Adjust as necessary
            
            monthly_filings[month_key].append({
                'filing_date': filing_date,
                'event_date': event_date,
                'content': filing['description'],
                'type': filing_type
            })

        return monthly_filings
    except Exception as e:
        logging.error(f"Error fetching 8-K filings for {ticker}: {e}")
        return {}

def extract_event_date(content):
    """Extract the event date from the 8-K content"""
    # This is a placeholder. You'd need to implement actual extraction logic.
    return None

def extract_filing_type(content):
    """Extract the specific type of 8-K filing"""
    # This is a placeholder. You'd need to implement actual extraction logic.
    return "Unknown"

def analyze_sentiment(text):
    """Analyze the sentiment of given text using FinBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_prob = probabilities[0][2].item()
    negative_prob = probabilities[0][0].item()
    return positive_prob - negative_prob  # Returns a score between -1 and 1

def get_stock_returns(ticker, date, days=5):
    """Get stock returns for a given number of days after the filing"""
    try:
        end_date = date + timedelta(days=days)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=date, end=end_date)
        if not hist.empty:
            return (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
    except Exception as e:
        logging.error(f"Error fetching stock returns for {ticker}: {e}")
    return None

def process_company_sentiment(ticker, start_date, end_date):
    """Process sentiment for a company's 8-K filings over time"""
    # Resolve company details using the Mapping API
    try:
        company_info = mapping_api.resolve("ticker", ticker)
        if company_info:
            company_details = company_info[0]  # Get the first matching company
            logging.info(f"Company details for {ticker}: {company_details}")
        else:
            logging.warning(f"No company details found for ticker: {ticker}")
            return []
    except Exception as e:
        logging.error(f"Error resolving company details for {ticker}: {e}")
        return []
    
    monthly_filings = get_sec_filings(ticker, start_date, end_date)
    
    monthly_sentiments = []
    for month, filings in monthly_filings.items():
        if filings:
            for filing in filings:
                sentiment = analyze_sentiment(filing['content'])
                returns = get_stock_returns(ticker, filing['filing_date'])
                
                monthly_sentiments.append({
                    'date': month,
                    'stock_ticker': ticker,
                    'company_name': company_details['name'],  # Add company name
                    'sentiment': sentiment,
                    'filing_type': filing['type'],
                    'event_date': filing['event_date'],
                    'filing_date': filing['filing_date'],
                    'returns_5d': returns,
                    'time_to_file': (filing['filing_date'] - filing['event_date']).days if filing['event_date'] else None
                })
    
    return monthly_sentiments

def main():
    # Load stock data
    stock_file_path = input("Enter the path to your stock data CSV file: ").strip()
    stock_df = load_stock_data(stock_file_path)
    
    # Get unique tickers
    tickers = stock_df['stock_ticker'].unique()
    
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    all_sentiments = []
    
    for ticker in tqdm(tickers, desc="Processing companies"):
        company_sentiments = process_company_sentiment(ticker, start_date, end_date)
        all_sentiments.extend(company_sentiments)
    
    # Create DataFrame
    sentiment_df = pd.DataFrame(all_sentiments)

    print(sentiment_df.head())
    print(sentiment_df.columns)

    if sentiment_df.empty:
        logging.warning("Sentiment DataFrame is empty. Skipping merge.")
    else:
        result_df = pd.merge(stock_df, sentiment_df, on=['stock_ticker', 'date'], how='left')


    # Merge with original stock data
    result_df = pd.merge(stock_df, sentiment_df, on=['stock_ticker', 'date'], how='left')
    
    # Save to CSV
    output_file = 'stock_sentiment_8k_analysis_optimized.csv'
    result_df.to_csv(output_file, index=False)
    logging.info(f"Data saved to {output_file}")
    
    # Display some statistics
    logging.info("\nDataset Summary:")
    logging.info(result_df.info())
    
    # Display a sample of the data
    logging.info("\nSample of processed data:")
    logging.info(result_df.head())

if __name__ == "__main__":
    main()