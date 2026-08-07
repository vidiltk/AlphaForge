import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

def get_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches OHLCV data for a given ticker and date range using yfinance.
    Dates format: YYYY-MM-DD
    """
    print(f"\nFetching OHLCV data for {ticker} from {start_date} to {end_date}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        print("No OHLCV data returned. Please check the ticker symbol or dates.")
        return pd.DataFrame()
        
    # Keep only required columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = df.index.strftime('%Y-%m-%d')
    return df

def scrape_yahoo_fundamentals(ticker: str) -> dict:
    """
    Scrapes the fundamental summary table metrics from Yahoo Finance via HTML.
    """
    print(f"Scraping Yahoo Finance HTML fundamentals for {ticker}...")
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch Yahoo Finance page: {e}")
        return {}
        
    soup = BeautifulSoup(response.text, "html.parser")
    fundamentals = {}
    target_fields = [
        "Previous Close", "Open", "Bid", "Ask", "Day's Range", 
        "52 Week Range", "Volume", "Avg. Volume", "Market Cap", 
        "Beta", "PE Ratio", "EPS", "Earnings Date", 
        "Forward Dividend & Yield", "Ex-Dividend Date", "1y Target Est"
    ]

    rows = soup.find_all(["tr", "li"])
    for row in rows:
        text_parts = [t.strip() for t in row.stripped_strings if t.strip()]
        if len(text_parts) >= 2:
            label = text_parts[0]
            value = " ".join(text_parts[1:])
            for field in target_fields:
                if field.lower() in label.lower() and field not in fundamentals:
                    fundamentals[field] = value
                    break
    return fundamentals

def get_deep_fundamentals(ticker: str) -> dict:
    """
    Fetches deeper company fundamentals (margins, sector, ROE, etc.) using yfinance.
    """
    print(f"Fetching deep company fundamentals for {ticker} via yfinance API...")
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Extracting a curated list of important company fundamentals
    deep_fundamentals = {
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Full Time Employees": info.get("fullTimeEmployees", "N/A"),
        "Trailing P/E": info.get("trailingPE", "N/A"),
        "Forward P/E": info.get("forwardPE", "N/A"),
        "Price to Book (P/B)": info.get("priceToBook", "N/A"),
        "Profit Margin": f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A",
        "Operating Margin": f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get('operatingMargins') else "N/A",
        "Return on Equity (ROE)": f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A",
        "Revenue Growth": f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get('revenueGrowth') else "N/A",
        "Debt to Equity": info.get("debtToEquity", "N/A"),
        "Total Cash": info.get("totalCash", "N/A"),
        "Total Debt": info.get("totalDebt", "N/A"),
    }
    return deep_fundamentals

# ====================================================
# Main Program
# ====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Comprehensive Yahoo Finance Data Scraper")
    print("=" * 60)
    
    # User Inputs
    ticker = input("Enter Stock Ticker (e.g. AAPL, NVDA, RELIANCE.NS): ").strip().upper()
    start_date = input("Enter Start Date (YYYY-MM-DD): ").strip()
    end_date = input("Enter End Date (YYYY-MM-DD): ").strip()
    
    # Fetch OHLCV
    ohlcv_df = get_ohlcv(ticker, start_date, end_date)
    if not ohlcv_df.empty:
        print("\nOHLCV Data (Last 5 Rows)")
        print("-" * 60)
        print(ohlcv_df.tail())
        
    # Fetch Scraped Fundamentals (Summary Table)
    print("\n" + "=" * 60)
    html_fundamentals = scrape_yahoo_fundamentals(ticker)
    print("\nSummary Metrics (Scraped)")
    print("-" * 60)
    if html_fundamentals:
        for key, value in html_fundamentals.items():
            print(f"{key:<30}: {value}")
    else:
        print("No HTML fundamental data found.")

    # Fetch Deep Fundamentals (yfinance API)
    print("\n" + "=" * 60)
    deep_fundamentals = get_deep_fundamentals(ticker)
    print("\nDeep Company Fundamentals (API)")
    print("-" * 60)
    for key, value in deep_fundamentals.items():
        print(f"{key:<30}: {value}")
