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

@IndicatorRegistry.register
class BollingerBandSignal(BaseIndicator):
    """Price relative to Bollinger Bands."""

    name = "bollinger"

    def __init__(self, window: int = 20, n_std: float = 2.0, **params):
        super().__init__(window=window, n_std=n_std, **params)
        self.window = window
        self.n_std = n_std

    def compute(self, df: pd.DataFrame) -> pd.Series:

        close = df["close"]

        mid = close.rolling(self.window).mean()

        std = close.rolling(self.window).std()

        z = (close - mid) / (self.n_std * std)

        return (-z).clip(-1, 1)

@IndicatorRegistry.register
class ATRSignal(BaseIndicator):
    """Trend weighted by ATR."""

    name = "atr"

    def __init__(self, window: int = 14, **params):
        super().__init__(window=window, **params)
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:

        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/self.window, adjust=False).mean()

        ema20 = close.ewm(span=20, adjust=False).mean()

        ema50 = close.ewm(span=50, adjust=False).mean()

        trend = np.sign(ema20 - ema50)

        strength = atr / atr.rolling(50).mean()

        signal = trend * strength

        return signal.clip(-1, 1)

@IndicatorRegistry.register
class KeltnerChannelSignal(BaseIndicator):

    name = "keltner"

    def __init__(self, window: int = 20, atr_window: int = 14, **params):
        super().__init__(window=window, atr_window=atr_window, **params)
        self.window = window
        self.atr_window = atr_window

    def compute(self, df: pd.DataFrame) -> pd.Series:

        high = df["high"]
        low = df["low"]
        close = df["close"]

        ema = close.ewm(span=self.window, adjust=False).mean()

        prev_close = close.shift()

        tr = pd.concat([
            high-low,
            (high-prev_close).abs(),
            (low-prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/self.atr_window, adjust=False).mean()

        signal = (close - ema)/(2*atr)

        return signal.clip(-1,1)

@IndicatorRegistry.register
class OBVSignal(BaseIndicator):

    name = "obv"

    def __init__(self, **params):
        super().__init__(**params)

    def compute(self, df: pd.DataFrame) -> pd.Series:

        close = df["close"]

        volume = df["volume"]

        direction = np.sign(close.diff()).fillna(0)

        obv = (direction * volume).cumsum()

        fast = obv.ewm(span=5, adjust=False).mean()

        slow = obv.ewm(span=20, adjust=False).mean()

        signal = (fast - slow) / (obv.abs().rolling(20).mean() + 1e-9)

        return signal.clip(-1,1)

@IndicatorRegistry.register
class ADXSignal(BaseIndicator):

    name = "adx"

    def __init__(self, window: int = 14, **params):
        super().__init__(window=window, **params)
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:

        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()

        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)

        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        prev_close = close.shift()

        tr = pd.concat([
            high-low,
            (high-prev_close).abs(),
            (low-prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/self.window, adjust=False).mean()

        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/self.window, adjust=False).mean() / atr

        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/self.window, adjust=False).mean() / atr

        dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di)

        adx = dx.ewm(alpha=1/self.window, adjust=False).mean()

        direction = (plus_di-minus_di)/(plus_di+minus_di)

        signal = direction * (adx/50)

        return signal.clip(-1,1)


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
