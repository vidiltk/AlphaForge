"""
Parallel contract to signals.base.BaseIndicator, but for ML-driven
strategies where the user tunes hyperparameters instead of picking
indicator weights. compute() still must return a signal Series in
[-1, 1] so it plugs straight into the same BaseBacktester.
"""
from abc import ABC, abstractmethod

import pandas as pd


class BaseMLStrategy(ABC):
    """
    Subclass this for every ML strategy.

    Contract:
      - __init__ takes hyperparameters exposed to the user
        (e.g. lookback window, n_estimators, threshold) with defaults.
      - fit(df) trains on live historical OHLCV data (walk-forward
        splitting is the mentee's responsibility inside fit/predict —
        the base class does not hide lookahead bugs for you).
      - predict(df) returns a pd.Series in [-1, 1], same index as df.
    """

    name: str = "base_ml_strategy"

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = None

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.Series:
        self.fit(df)
        signal = self.predict(df)
        return signal.reindex(df.index).clip(-1, 1).fillna(0)


class MLStrategyRegistry:
    _registry = {}

    @classmethod
    def register(cls, strategy_cls):
        cls._registry[strategy_cls.name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]

    @classmethod
    def all(cls):
        return dict(cls._registry)
