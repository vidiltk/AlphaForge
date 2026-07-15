"""
Live data access layer. Every function here hits yfinance directly (or the
local DB cache of a previous live pull) — nothing in this module ever
returns fabricated or sample data. If yfinance has no data for a symbol,
functions raise instead of silently returning something fake.
"""
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from django.utils import timezone

from .models import Ticker, PriceBar


class TickerNotFound(Exception):
    pass


def get_or_create_ticker(symbol: str) -> Ticker:
    symbol = symbol.strip().upper()
    ticker, _ = Ticker.objects.get_or_create(symbol=symbol)
    return ticker


def refresh_fundamentals(symbol: str) -> Ticker:
    """Pull live .info snapshot from yfinance and store it on the Ticker row."""
    try:
        yf_ticker = yf.Ticker(symbol)
        info = yf_ticker.info
    except Exception as exc:
        raise TickerNotFound(f"couldn't reach Yahoo Finance for '{symbol}': {exc}") from exc
    if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None \
            and not info.get('shortName'):
        raise TickerNotFound(f"yfinance returned no data for '{symbol}'")

    ticker = get_or_create_ticker(symbol)
    ticker.short_name = info.get('shortName', '') or ''
    ticker.sector = info.get('sector', '') or ''
    ticker.industry = info.get('industry', '') or ''
    ticker.currency = info.get('currency', '') or ''
    ticker.market_cap = info.get('marketCap')
    ticker.pe_ratio = info.get('trailingPE')
    ticker.forward_pe = info.get('forwardPE')
    ticker.eps = info.get('trailingEps')
    ticker.dividend_yield = info.get('dividendYield')
    ticker.fifty_two_week_high = info.get('fiftyTwoWeekHigh')
    ticker.fifty_two_week_low = info.get('fiftyTwoWeekLow')
    ticker.beta = info.get('beta')
    ticker.long_business_summary = info.get('longBusinessSummary', '') or ''
    ticker.fundamentals_updated_at = timezone.now()
    ticker.save()
    return ticker


def fetch_price_history(symbol: str, start: date = None, end: date = None,
                         period: str = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Return an OHLCV DataFrame indexed by date for `symbol`.

    Pass either (start, end) for an explicit historical range, or `period`
    (e.g. "1y", "5y", "max") the way yfinance expects it. Data pulled live
    from yfinance and upserted into PriceBar so repeat requests for the
    same range don't re-hit the API unless force_refresh=True.
    """
    ticker = get_or_create_ticker(symbol)

    if not force_refresh and start and end:
        cached = PriceBar.objects.filter(ticker=ticker, date__gte=start, date__lte=end)
        expected_trading_days = max((end - start).days * 5 // 7 - 5, 0)
        if cached.count() >= expected_trading_days:
            return _bars_to_df(cached)

    try:
        yf_ticker = yf.Ticker(symbol)
        if period:
            df = yf_ticker.history(period=period, auto_adjust=False)
        else:
            df = yf_ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=False)
    except Exception as exc:
        raise TickerNotFound(f"couldn't reach Yahoo Finance for '{symbol}': {exc}") from exc

    if df.empty:
        raise TickerNotFound(f"yfinance returned no price history for '{symbol}'")

    _upsert_bars(ticker, df)

    if start and end:
        bars = PriceBar.objects.filter(ticker=ticker, date__gte=start, date__lte=end)
        return _bars_to_df(bars)
    return df.rename(columns={'Adj Close': 'adj_close', 'Open': 'open', 'High': 'high',
                               'Low': 'low', 'Close': 'close', 'Volume': 'volume'})


def _upsert_bars(ticker: Ticker, df: pd.DataFrame):
    objs = []
    for idx, row in df.iterrows():
        objs.append(PriceBar(
            ticker=ticker,
            date=idx.date(),
            open=row['Open'], high=row['High'], low=row['Low'],
            close=row['Close'], adj_close=row.get('Adj Close', row['Close']),
            volume=int(row['Volume']) if not pd.isna(row['Volume']) else 0,
        ))
    PriceBar.objects.bulk_create(
        objs,
        update_conflicts=True,
        update_fields=['open', 'high', 'low', 'close', 'adj_close', 'volume'],
        unique_fields=['ticker', 'date'],
    )


def _bars_to_df(queryset) -> pd.DataFrame:
    records = list(queryset.order_by('date').values(
        'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'
    ))
    df = pd.DataFrame.from_records(records, index='date')
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    return df
