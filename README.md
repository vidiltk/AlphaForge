# Quant Dashboard — Mentee Framework

Django skeleton for the ticker → indicators → weighted signal → backtest →
dashboard pipeline. All price and fundamental data is pulled **live** from
yfinance (cached in the DB after first pull) — there is no seeded/demo data
anywhere in this repo.

## Apps and what each mentee owns

- **marketdata/** — the scraper. `services.py` wraps yfinance
  (`fetch_price_history`, `refresh_fundamentals`). Extend this for more
  fundamentals fields, other data vendors, or smarter caching.
- **signals/** — indicators. Subclass `signals.base.BaseIndicator`,
  implement `compute(df) -> pd.Series` returning values in [-1, 1], and
  register with `@IndicatorRegistry.register`. `indicators.py` has three
  working examples (SMA crossover, RSI, MACD) plus `combine_signals()`
  which turns `[(indicator, weight), ...]` into one composite signal.
- **backtest/** — `engine.py`'s `BaseBacktester` is a vectorized
  long/short backtester with transaction costs and standard metrics
  (CAGR, Sharpe, max drawdown, win rate). Subclass it to add slippage
  models, position sizing, stop losses, etc. Contract:
  `run(df, signal) -> BacktestResult`.
- **mlstrategies/** — `base.py`'s `BaseMLStrategy` mirrors the indicator
  contract but for ML models: `fit(df)` + `predict(df) -> pd.Series` in
  [-1, 1], so ML strategies plug into the same `BaseBacktester` untouched.
- **dashboard/** — glues it together. `views.run_backtest` is the one
  endpoint that fetches live data, builds the composite signal, runs the
  backtest, and returns backtest results + fundamentals as JSON. The
  template is a bare stub — this is the main thing to build out
  (ticker input, multi-select indicators with weight sliders, equity
  curve chart, fundamentals panel).

## Setup

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

## Extension pattern (adding a new indicator)

```python
# signals/indicators.py
@IndicatorRegistry.register
class BollingerBandSignal(BaseIndicator):
    name = "bollinger"

    def __init__(self, window: int = 20, n_std: float = 2.0, **params):
        super().__init__(window=window, n_std=n_std, **params)
        self.window, self.n_std = window, n_std

    def compute(self, df):
        mid = df['close'].rolling(self.window).mean()
        std = df['close'].rolling(self.window).std()
        z = (df['close'] - mid) / (self.n_std * std)
        return -z.clip(-1, 1)  # price near upper band -> bearish score
```

No other file needs to change — it shows up automatically in
`IndicatorRegistry.all()` and can be requested from the dashboard by name.
