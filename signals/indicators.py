"""
Starter indicators so the framework is runnable end-to-end on day one.
Mentees will add more of these following the same pattern — subclass
BaseIndicator, implement compute(), register with @IndicatorRegistry.register.
"""
import pandas as pd

from .base import BaseIndicator, IndicatorRegistry


@IndicatorRegistry.register
class SMACrossover(BaseIndicator):
    """+1 when fast SMA > slow SMA (uptrend), -1 when below."""
    name = "sma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, **params):
        super().__init__(fast=fast, slow=slow, **params)
        self.fast = fast
        self.slow = slow

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = df['close'].rolling(self.fast).mean()
        slow_ma = df['close'].rolling(self.slow).mean()
        spread = (fast_ma - slow_ma) / slow_ma
        # scale so a ~5% spread already saturates to +/-1
        return (spread / 0.05).clip(-1, 1)


@IndicatorRegistry.register
class RSI(BaseIndicator):
    """Classic RSI mapped from [0,100] to [-1,1], inverted so oversold=+1."""
    name = "rsi"

    def __init__(self, window: int = 14, **params):
        super().__init__(window=window, **params)
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(self.window).mean()
        loss = (-delta.clip(upper=0)).rolling(self.window).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        # RSI 30 (oversold) -> +1 (buy signal), RSI 70 (overbought) -> -1
        return ((50 - rsi) / 20).clip(-1, 1)


@IndicatorRegistry.register
class MACDSignal(BaseIndicator):
    """+1/-1 scaled by MACD histogram normalized by price."""
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **params):
        super().__init__(fast=fast, slow=slow, signal=signal, **params)
        self.fast, self.slow, self.signal = fast, slow, signal

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        hist = (macd_line - signal_line) / df['close']
        return (hist / 0.02).clip(-1, 1)


def combine_signals(df: pd.DataFrame, weighted_indicators: list[tuple[BaseIndicator, float]]) -> pd.Series:
    """
    weighted_indicators: list of (indicator_instance, weight) tuples.
    Weights are normalized to sum to 1 automatically. Returns the composite
    signal Series in [-1, 1] that the backtester consumes.
    """
    total_weight = sum(abs(w) for _, w in weighted_indicators) or 1.0
    composite = pd.Series(0.0, index=df.index)
    for indicator, weight in weighted_indicators:
        composite = composite.add(indicator(df) * (weight / total_weight), fill_value=0)
    return composite.clip(-1, 1)
