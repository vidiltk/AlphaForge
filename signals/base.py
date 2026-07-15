"""
Every indicator a mentee builds subclasses BaseIndicator and implements
compute(). This is the contract the rest of the system (combiner,
backtester) relies on — keep it stable.
"""
from abc import ABC, abstractmethod

import pandas as pd


class BaseIndicator(ABC):
    """
    Subclass this for every new indicator.

    Contract:
      - __init__ takes whatever parameters the indicator needs
        (e.g. window length) with sensible defaults.
      - compute(df) receives a live OHLCV DataFrame (columns:
        open, high, low, close, adj_close, volume; DatetimeIndex)
        and returns a pd.Series, same index as df, with values in
        [-1, 1]:
            +1  = strongest bullish signal
             0  = neutral / no signal
            -1  = strongest bearish signal
        Continuous values in between are fine and encouraged
        (e.g. a normalized RSI-based score) — the combiner will
        respect the strength, not just the sign.
    """

    name: str = "base_indicator"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def validate_output(self, signal: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Called automatically after compute() — clips to [-1, 1] and
        aligns the index. Mentees don't need to call this themselves."""
        if not signal.index.equals(df.index):
            signal = signal.reindex(df.index)
        return signal.clip(-1, 1).fillna(0)

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.validate_output(self.compute(df), df)


class IndicatorRegistry:
    """Central place indicators register themselves so the dashboard can
    list them without hardcoding imports everywhere."""
    _registry = {}

    @classmethod
    def register(cls, indicator_cls):
        cls._registry[indicator_cls.name] = indicator_cls
        return indicator_cls

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]

    @classmethod
    def all(cls):
        return dict(cls._registry)
