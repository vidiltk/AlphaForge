"""
Vectorized backtest engine. Mentees extending this framework will mostly
subclass BaseBacktester to add execution realism (slippage, position
sizing rules, stop losses) while keeping the same input/output contract.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "equity_curve": [
                {"date": d.strftime("%Y-%m-%d"), "equity": v}
                for d, v in self.equity_curve.items()
            ],
            "metrics": self.metrics,
        }


class BaseBacktester:
    """
    Contract:
      run(df, signal) -> BacktestResult

      df:     live OHLCV DataFrame (DatetimeIndex, columns include 'close')
      signal: pd.Series in [-1, 1], same index as df, from combine_signals()

    The default implementation here is a simple long/short vectorized
    backtest: position on day t = signal on day t-1 (no lookahead),
    scaled by signal strength, applied to next-day returns.
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 fee_bps: float = 5.0, allow_short: bool = True):
        self.initial_capital = initial_capital
        self.fee_bps = fee_bps  # round-trip cost per unit of position change, in basis points
        self.allow_short = allow_short

    def run(self, df: pd.DataFrame, signal: pd.Series) -> BacktestResult:
        signal = signal.reindex(df.index).fillna(0)
        position = signal.shift(1).fillna(0)  # trade on next bar, no lookahead
        if not self.allow_short:
            position = position.clip(lower=0)

        asset_returns = df['close'].pct_change().fillna(0)
        turnover = position.diff().abs().fillna(0)
        cost = turnover * (self.fee_bps / 10_000)

        strategy_returns = position * asset_returns - cost
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()

        metrics = self.compute_metrics(strategy_returns, equity_curve)
        return BacktestResult(
            equity_curve=equity_curve,
            returns=strategy_returns,
            positions=position,
            metrics=metrics,
        )

    def compute_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> dict:
        n_years = max(len(returns) / 252, 1e-9)
        total_return = equity_curve.iloc[-1] / self.initial_capital - 1
        cagr = (equity_curve.iloc[-1] / self.initial_capital) ** (1 / n_years) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1
        max_drawdown = drawdown.min()
        win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0.0

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "annualized_vol_pct": round(vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
        }
