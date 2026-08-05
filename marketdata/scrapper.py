import yfinance as yf

def fetch_stock_ohlcv(ticker='AAPL', period='1mo', interval='1d'):
    data = yf.download(tickers=ticker, period=period, interval=interval)
    # Standardizes columns: Open, High, Low, Close, Volume
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(fetch_stock_ohlcv('RELIANCE.NS'))
