"""
Regime switching framework for AlphaForge.
Detects market regimes and routes predictions to regime-specific models.
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings

from .base import BaseMLStrategy, MLStrategyRegistry

warnings.filterwarnings("ignore")


# ─── Regime Detectors ───────────────────────────────────────────────

class BaseRegimeDetector(ABC):
    """Abstract base for regime detection."""

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return pd.Series[str] with same index as df, each value a regime label."""
        raise NotImplementedError

    @property
    @abstractmethod
    def regime_names(self) -> list[str]:
        """Return list of possible regime labels this detector can produce."""
        raise NotImplementedError


class ATRRegimeDetector(BaseRegimeDetector):
    """
    ATR-based volatility regime detection.
    ATR(14) > rolling ATR mean(20) → high_vol, else low_vol.
    """

    def __init__(self, atr_period=14, ma_period=20):
        self.atr_period = atr_period
        self.ma_period = ma_period

    @property
    def regime_names(self):
        return ["high_vol", "low_vol"]

    def detect(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_val = tr.ewm(span=self.atr_period, adjust=False).mean()
        atr_ma = atr_val.rolling(self.ma_period).mean()

        regime = pd.Series("low_vol", index=df.index)
        regime[atr_val > atr_ma] = "high_vol"
        # Back-fill the warm-up window
        regime.iloc[: self.ma_period] = "low_vol"
        return regime


class ADXRegimeDetector(BaseRegimeDetector):
    """
    ADX-based trend strength detection with hysteresis.
    ADX > 25 → trending, ADX < 20 → range_bound.
    ADX between 20-25 keeps previous regime to avoid whipsaw.
    """

    def __init__(self, window=14):
        self.window = window

    @property
    def regime_names(self):
        return ["trending", "range_bound"]

    def detect(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        alpha = 1 / self.window
        atr_val = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = (
            100
            * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
            / atr_val
        )
        minus_di = (
            100
            * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
            / atr_val
        )

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        # Hysteresis loop
        regimes = []
        prev_regime = "range_bound"
        for val in adx.values:
            if np.isnan(val):
                regimes.append(prev_regime)
            elif val > 25:
                prev_regime = "trending"
                regimes.append("trending")
            elif val < 20:
                prev_regime = "range_bound"
                regimes.append("range_bound")
            else:
                regimes.append(prev_regime)

        return pd.Series(regimes, index=df.index)


class HMMRegimeDetector(BaseRegimeDetector):
    """
    Hidden Markov Model regime detection trained on returns.
    Maps hidden states to bull / bear / sideways / high_vol based on
    state means and variances.  Falls back to ATR if training fails
    or hmmlearn is unavailable.
    """

    def __init__(self, n_states=3):
        self.n_states = min(max(n_states, 2), 4)
        self._fallback = ATRRegimeDetector()

    @property
    def regime_names(self):
        return ["bull", "bear", "sideways", "high_vol"]

    def detect(self, df: pd.DataFrame) -> pd.Series:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            return self._fallback.detect(df)

        returns = df["close"].pct_change().dropna()
        if len(returns) < 50:
            return self._fallback.detect(df)

        try:
            X = returns.values.reshape(-1, 1)
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(X)
            hidden_states = model.predict(X)

            state_means = model.means_.flatten()
            state_vars = np.array(
                [c[0][0] for c in model.covars_]
                if model.covars_.ndim == 3
                else model.covars_.flatten()
            )

            labels = ["sideways"] * self.n_states
            sorted_by_mean = np.argsort(state_means)

            if self.n_states == 2:
                labels[sorted_by_mean[0]] = "bear"
                labels[sorted_by_mean[1]] = "bull"
            elif self.n_states == 3:
                labels[sorted_by_mean[0]] = "bear"
                labels[sorted_by_mean[1]] = "sideways"
                labels[sorted_by_mean[2]] = "bull"
            else:  # 4
                highest_var = int(np.argmax(state_vars))
                remaining = [i for i in sorted_by_mean if i != highest_var]
                labels[highest_var] = "high_vol"
                labels[remaining[0]] = "bear"
                labels[remaining[1]] = "sideways"
                labels[remaining[2]] = "bull"

            regime_series = pd.Series("sideways", index=df.index)
            for i, idx in enumerate(returns.index):
                regime_series.loc[idx] = labels[hidden_states[i]]
            # Forward-fill the very first row (no return available)
            regime_series.iloc[0] = (
                regime_series.iloc[1] if len(regime_series) > 1 else "sideways"
            )
            return regime_series

        except Exception:
            return self._fallback.detect(df)


REGIME_DETECTORS = {
    "atr": ATRRegimeDetector,
    "adx": ADXRegimeDetector,
    "hmm": HMMRegimeDetector,
}


# ─── Regime Switching Strategy ──────────────────────────────────────

class RegimeSwitchingStrategy:
    """
    Orchestrates regime-aware ML prediction.
    Detects regimes → trains per-regime models → routes predictions.
    """

    def __init__(self, detector_name: str, model_map: dict):
        """
        Args:
            detector_name: One of 'atr', 'adx', 'hmm'.
            model_map: Maps regime label → ML model registry name,
                       e.g. {"high_vol": "xgboost", "low_vol": "lightgbm"}.
        """
        detector_cls = REGIME_DETECTORS.get(detector_name, ATRRegimeDetector)
        self.detector = detector_cls()
        self.model_map = model_map
        self.regime_models: dict = {}  # regime_label → fitted strategy instance
        self.is_fitted = False

    # ── public entry-point ──────────────────────────────────────────

    def run(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple:
        """
        Returns
        -------
        signal : pd.Series   – trading signal in [-1, 1] for the test period.
        regime_info : dict    – timeline, usage stats, per-regime metrics.
        """
        # 1. Detect regimes on training data
        train_regimes = self.detector.detect(train_df)

        # 2. Train one model per regime
        for regime_label, model_name in self.model_map.items():
            regime_mask = train_regimes == regime_label
            regime_data = train_df[regime_mask]

            # Fall back to full training set when regime slice is too thin
            if len(regime_data) < 50:
                regime_data = train_df

            try:
                strategy_cls = MLStrategyRegistry.get(model_name)
            except KeyError:
                strategy_cls = MLStrategyRegistry.get("ensemble")

            instance = strategy_cls()
            instance.fit(regime_data)
            self.regime_models[regime_label] = instance

        self.is_fitted = True

        # 3. Detect regimes on the *full* history so test-time detection
        #    uses only past information (no look-ahead: the detector
        #    processes each bar using only data up to that bar).
        full_df = pd.concat([train_df, test_df])
        full_regimes = self.detector.detect(full_df)
        test_regimes = full_regimes.reindex(test_df.index)

        # 4. Route each bar to its regime-specific model
        signal = pd.Series(0.0, index=test_df.index)

        for regime_label, model_instance in self.regime_models.items():
            regime_mask = test_regimes == regime_label
            if not regime_mask.any():
                continue
            regime_test = test_df[regime_mask]
            regime_signal = model_instance.predict(regime_test)
            signal.loc[regime_mask] = (
                regime_signal.reindex(test_df.index[regime_mask]).fillna(0.0)
            )

        signal = signal.clip(-1, 1).fillna(0)

        # 5. Build analytics payload
        regime_info = self._compute_regime_info(test_df, test_regimes, signal)

        return signal, regime_info

    # ── private helpers ─────────────────────────────────────────────

    def _compute_regime_info(self, test_df, regimes, signal):
        timeline = [
            {
                "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                "regime": regime,
            }
            for idx, regime in regimes.items()
        ]

        # Usage %
        usage_counts = regimes.value_counts().to_dict()
        total = max(len(regimes), 1)
        usage_pct = {k: round(v / total * 100, 1) for k, v in usage_counts.items()}

        # Switches
        switches = max(int((regimes != regimes.shift(1)).sum()) - 1, 0)

        # Per-regime metrics
        returns = test_df["close"].pct_change().fillna(0.0)
        per_regime = {}

        for label in regimes.unique():
            mask = regimes == label
            r = returns[mask]
            s = signal[mask]
            strat_rets = s.shift(1).fillna(0.0) * r

            n_bars = int(mask.sum())
            total_ret = float((1 + strat_rets).prod() - 1) * 100 if n_bars else 0.0
            trades = strat_rets[strat_rets != 0]
            wins = int((trades > 0).sum())
            n_trades = len(trades)
            win_rate = round(wins / n_trades * 100, 1) if n_trades else 0.0
            sharpe = (
                round(float(strat_rets.mean() / strat_rets.std() * np.sqrt(252)), 2)
                if len(strat_rets) > 1 and strat_rets.std() > 0
                else 0.0
            )
            avg_ret = round(float(strat_rets.mean()) * 100, 4) if len(strat_rets) else 0.0

            per_regime[label] = {
                "bars": n_bars,
                "time_pct": round(n_bars / total * 100, 1),
                "returns_pct": round(total_ret, 2),
                "win_rate_pct": win_rate,
                "sharpe": sharpe,
                "trades": n_trades,
                "avg_trade_return_pct": avg_ret,
            }

        return {
            "timeline": timeline,
            "usage": usage_pct,
            "switches": switches,
            "per_regime_metrics": per_regime,
        }
