import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
import talib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fetch Stock Data using yfinance
def fetch_stock_data(tickers, start='', end=''):
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start, end=end)
    return stock_data

# Fetch News Articles using NewsAPI
API_KEY = "" #news api key here
BASE_URL = "https://newsapi.org/v2/everything"

def fetch_news(stock_name, from_date, to_date):
    query = f"{stock_name} stock"
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'sortBy': 'publishedAt',
        'apiKey': API_KEY,
    }
    
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        articles = response.json()['articles']
        return articles
    else:
        print(f"Error fetching news: {response.status_code}")
        return []

# Sentiment Analysis using Hugging Face Transformers
sentiment_analyzer = pipeline('sentiment-analysis')

def get_sentiment(news_articles):
    sentiments = []
    for article in news_articles:
        sentiment = sentiment_analyzer(article['description'])[0]
        sentiments.append(sentiment['label'])  # "POSITIVE", "NEGATIVE", or "NEUTRAL"
    return sentiments

# Compute Technical Indicators using TA-Lib
def compute_indicators(stock_data):
    for ticker in stock_data:
        stock_data[ticker]['SMA_50'] = stock_data[ticker]['Close'].rolling(window=50).mean()
        stock_data[ticker]['RSI'] = talib.RSI(stock_data[ticker]['Close'], timeperiod=14)
        stock_data[ticker]['MACD'], stock_data[ticker]['MACD_signal'], _ = talib.MACD(stock_data[ticker]['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return stock_data

# Aggregating Sentiment for the Day (Positive/Negative/Neutral Count)
def aggregate_sentiment(sentiments):
    return {"positive": sentiments.count("POSITIVE"), "negative": sentiments.count("NEGATIVE"), "neutral": sentiments.count("NEUTRAL")}

# Prepare Features for the Model
def prepare_features(stock_data, sentiment_summary):
    features = []
    target = []
    
    for ticker in stock_data:
        data = stock_data[ticker].dropna()
        
        # Add features: technical indicators + sentiment
        for index, row in data.iterrows():
            features.append([
                row['SMA_50'], row['RSI'], row['MACD'], row['MACD_signal'],
                sentiment_summary['positive'], sentiment_summary['negative'], sentiment_summary['neutral']
            ])
            
            # Target: Buy, Hold, or Sell based on price change
            if row['Close'] > row['Open']:  # If closing price is higher than open
                target.append(1)  # Buy
            else:
                target.append(0)  # Hold (can expand for Sell)
    
    return pd.DataFrame(features), target

# Train a Random Forest classifier to predict Buy/Hold
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Get Stock Recommendations based on the trained model
def recommend_stocks(model, stock_data, sentiment_summary):
    recommendations = []
    
    for ticker in stock_data:
        data = stock_data[ticker].dropna()
        for index, row in data.iterrows():
            features = [
                row['SMA_50'], row['RSI'], row['MACD'], row['MACD_signal'],
                sentiment_summary['positive'], sentiment_summary['negative'], sentiment_summary['neutral']
            ]
            prediction = model.predict([features])[0]
            if prediction == 1:  # Buy recommendation
                recommendations.append(ticker)
                
    return recommendations

# Main Function to Fetch Data, Process, Train, and Recommend
def main():
    # Example NSE500 tickers (replace with actual NSE500 tickers)
    nse500_tickers = [for tickers in pd.read_csv("ind_nifty500list.csv")["Symbol"] ticker+".NS"]#extract all tickers

    # 1. Fetch Stock Data
    stock_data = fetch_stock_data(nse500_tickers)
    
    # 2. Fetch News Data for the last 24 hours (or any desired time range)
    from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    news_articles = []
    for ticker in nse500_tickers:
        news_articles += fetch_news(ticker, from_date, to_date)
    
    # 3. Perform Sentiment Analysis on News Articles
    sentiments = get_sentiment(news_articles)
    
    # 4. Aggregate Sentiment for the day
    sentiment_summary = aggregate_sentiment(sentiments)
    print("Sentiment Summary:", sentiment_summary)
    
    # 5. Compute Technical Indicators for Stock Data
    stock_data = compute_indicators(stock_data)
    
    # 6. Prepare Features for Model
    X, y = prepare_features(stock_data, sentiment_summary)
    
    # 7. Split Data and Train the Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    
    # 8. Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    # 9. Get Stock Recommendations for the Day
    recommended_stocks = recommend_stocks(model, stock_data, sentiment_summary)
    print("Recommended Stocks to Buy:", recommended_stocks)

# Run the main function
if __name__ == "__main__":
    main()
