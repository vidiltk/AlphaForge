from django.db import models


class Ticker(models.Model):
    """A single tradable symbol, e.g. AAPL, RELIANCE.NS, ^NSEI."""
    symbol = models.CharField(max_length=20, unique=True, db_index=True)
    short_name = models.CharField(max_length=255, blank=True)
    sector = models.CharField(max_length=100, blank=True)
    industry = models.CharField(max_length=100, blank=True)
    currency = models.CharField(max_length=10, blank=True)

    # Fundamentals snapshot (from yfinance .info) — always overwritten with
    # the latest live pull. Never seeded manually / no demo values.
    market_cap = models.BigIntegerField(null=True, blank=True)
    pe_ratio = models.FloatField(null=True, blank=True)
    forward_pe = models.FloatField(null=True, blank=True)
    eps = models.FloatField(null=True, blank=True)
    dividend_yield = models.FloatField(null=True, blank=True)
    fifty_two_week_high = models.FloatField(null=True, blank=True)
    fifty_two_week_low = models.FloatField(null=True, blank=True)
    beta = models.FloatField(null=True, blank=True)
    long_business_summary = models.TextField(blank=True)
    fundamentals_updated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.symbol


class PriceBar(models.Model):
    """One OHLCV bar for a ticker. Populated only from live yfinance pulls."""
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE, related_name='bars')
    date = models.DateField(db_index=True)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    adj_close = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('ticker', 'date')
        ordering = ['date']

    def __str__(self):
        return f"{self.ticker.symbol} {self.date}"
