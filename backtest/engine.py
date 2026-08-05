"""
Strategy-agnostic backtesting engine.

The engine consumes market bars plus a signal in [-1, 1]. It never knows
whether that signal came from RSI, MACD, Bollinger Bands, ML, or any other
strategy. Signals are shifted by one bar before execution to prevent
look-ahead bias: a signal observed after today's close can only trade on the
next available bar.
"""
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


AllocationMode = Literal["full", "fixed", "conviction"]


class BacktestError(ValueError):
    """Raised when inputs cannot produce a valid backtest."""


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 0.0
    allow_short: bool = True
    allocation_mode: AllocationMode = "conviction"
    fixed_allocation: float = 1.0
    risk_per_trade: float | None = None
    benchmark_column: str | None = None


@dataclass
class RiskConfig:
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    max_holding_period: int | None = None
    cooldown_period: int = 0
    max_drawdown_pct: float | None = None
    max_consecutive_losses: int | None = None


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position: str
    holding_days: int
    gross_return_pct: float
    net_return_pct: float
    transaction_costs: float
    slippage: float
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    metrics: dict = field(default_factory=dict)
    trade_log: list[Trade] = field(default_factory=list)
    drawdown: pd.Series = field(default_factory=pd.Series)
    benchmark_comparison: dict = field(default_factory=dict)
    cash: pd.Series = field(default_factory=pd.Series)
    portfolio_value: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> dict:
        daily_returns = self.daily_returns if not self.daily_returns.empty else self.returns
        portfolio_value = self.portfolio_value if not self.portfolio_value.empty else self.equity_curve
        return {
            "equity_curve": _series_to_records(self.equity_curve, "equity"),
            "positions": _series_to_records(self.positions, "position"),
            "daily_returns": _series_to_records(daily_returns, "return"),
            "portfolio_value": _series_to_records(portfolio_value, "value"),
            "cash": _series_to_records(self.cash, "cash"),
            "drawdown": _series_to_records(self.drawdown, "drawdown"),
            "trade_log": [asdict(trade) for trade in self.trade_log],
            "metrics": self.metrics,
            "benchmark_comparison": self.benchmark_comparison,
        }


class BaseBacktester:
    """
    Backward-compatible strategy-agnostic backtester.

    Existing callers may continue to use:
        BaseBacktester(initial_capital=100000, fee_bps=5).run(df, signal)

    New callers can pass a richer BacktestConfig/RiskConfig or supply a
    DataFrame with a `signal`/`Signal` column and omit the second argument.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_bps: float = 5.0,
        allow_short: bool = True,
        slippage_bps: float = 0.0,
        allocation_mode: AllocationMode = "conviction",
        fixed_allocation: float = 1.0,
        risk_per_trade: float | None = None,
        risk_config: RiskConfig | None = None,
        config: BacktestConfig | None = None,
    ):
        self.config = config or BacktestConfig(
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            allow_short=allow_short,
            allocation_mode=allocation_mode,
            fixed_allocation=fixed_allocation,
            risk_per_trade=risk_per_trade,
        )
        self.risk = risk_config or RiskConfig()
        self._validate_config()

    @property
    def initial_capital(self) -> float:
        return self.config.initial_capital

    @property
    def fee_bps(self) -> float:
        return self.config.fee_bps

    @property
    def allow_short(self) -> bool:
        return self.config.allow_short

    def run(self, df: pd.DataFrame, signal: pd.Series | None = None) -> BacktestResult:
        bars, raw_signal = self._prepare_inputs(df, signal)
        execution_signal = raw_signal.shift(1).fillna(0.0)
        if not self.config.allow_short:
            execution_signal = execution_signal.clip(lower=0.0)

        target_exposure = self._target_exposure(execution_signal)
        simulation = self._simulate(bars, target_exposure)
        daily_returns = simulation["portfolio_value"].pct_change().fillna(0.0)
        drawdown = _drawdown(simulation["portfolio_value"])
        benchmark = self._benchmark_comparison(bars, daily_returns)
        metrics = self.compute_metrics(
            daily_returns,
            simulation["portfolio_value"],
            simulation["positions"],
            simulation["trades"],
            simulation["turnover"],
        )

        return BacktestResult(
            equity_curve=simulation["portfolio_value"],
            returns=daily_returns,
            positions=simulation["positions"],
            metrics=metrics,
            trade_log=simulation["trades"],
            drawdown=drawdown,
            benchmark_comparison=benchmark,
            cash=simulation["cash"],
            portfolio_value=simulation["portfolio_value"],
            daily_returns=daily_returns,
        )

    def _simulate(self, bars: pd.DataFrame, target_exposure: pd.Series) -> dict:
        cash = self.config.initial_capital
        units = 0.0
        current_side = 0
        entry_date = None
        entry_price = 0.0
        entry_value = 0.0
        entry_costs = 0.0
        entry_slippage = 0.0
        highest_price = -np.inf
        lowest_price = np.inf
        holding_days = 0
        cooldown = 0
        consecutive_losses = 0
        kill_switch = False
        peak_value = self.config.initial_capital

        cash_values = []
        portfolio_values = []
        position_values = []
        turnover_values = []
        trades: list[Trade] = []

        for timestamp, row in bars.iterrows():
            open_price = float(row["open"])
            close_price = float(row["close"])
            portfolio_before_trade = cash + units * open_price
            peak_value = max(peak_value, portfolio_before_trade)
            current_drawdown = portfolio_before_trade / peak_value - 1 if peak_value else 0.0

            desired_exposure = 0.0 if kill_switch else float(target_exposure.loc[timestamp])
            desired_side = int(np.sign(desired_exposure))
            exit_reason = self._risk_exit_reason(
                side=current_side,
                open_price=open_price,
                entry_price=entry_price,
                highest_price=highest_price,
                lowest_price=lowest_price,
                holding_days=holding_days,
                current_drawdown=current_drawdown,
            )

            if self.risk.max_drawdown_pct is not None and current_drawdown <= -abs(self.risk.max_drawdown_pct):
                exit_reason = exit_reason or "max_drawdown"
                kill_switch = True
                desired_exposure = 0.0
                desired_side = 0

            if self.risk.max_consecutive_losses is not None and consecutive_losses >= self.risk.max_consecutive_losses:
                exit_reason = exit_reason or "max_consecutive_losses"
                kill_switch = True
                desired_exposure = 0.0
                desired_side = 0

            if exit_reason:
                desired_exposure = 0.0
                desired_side = 0

            if cooldown > 0 and current_side == 0:
                desired_exposure = 0.0
                desired_side = 0
                cooldown -= 1

            if current_side != 0 and desired_side != current_side:
                trade, cash, units = self._close_trade(
                    timestamp=timestamp,
                    price=open_price,
                    cash=cash,
                    units=units,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    entry_value=entry_value,
                    entry_costs=entry_costs,
                    entry_slippage=entry_slippage,
                    exit_reason=exit_reason or ("signal_flip" if desired_side else "signal_exit"),
                )
                trades.append(trade)
                consecutive_losses = consecutive_losses + 1 if trade.net_return_pct < 0 else 0
                current_side = 0
                entry_date = None
                entry_costs = 0.0
                entry_slippage = 0.0
                holding_days = 0
                highest_price = -np.inf
                lowest_price = np.inf
                if self.risk.cooldown_period > 0:
                    cooldown = self.risk.cooldown_period
                    if desired_side != 0:
                        desired_exposure = 0.0
                        desired_side = 0
                        cooldown -= 1

            turnover = 0.0
            if desired_side != 0:
                target_value = self._target_position_value(cash + units * open_price, desired_exposure)
                if abs(units) <= 1e-12:
                    target_value = self._reserve_execution_costs(target_value)
                target_units = target_value / open_price
                delta_units = target_units - units
                if abs(delta_units) > 1e-12:
                    trade_value = abs(delta_units * open_price)
                    fee, slippage = self._execution_costs(trade_value)
                    cash -= delta_units * open_price + fee + slippage
                    units = target_units
                    turnover = trade_value / max(cash + units * open_price, 1e-12)
                    if current_side == 0:
                        current_side = desired_side
                        entry_date = timestamp
                        entry_price = open_price
                        entry_value = abs(units * open_price)
                        entry_costs = fee
                        entry_slippage = slippage
                        highest_price = open_price
                        lowest_price = open_price
            elif current_side == 0 and abs(units) > 1e-12:
                cash += units * open_price
                units = 0.0

            if current_side != 0:
                holding_days += 1
                highest_price = max(highest_price, close_price)
                lowest_price = min(lowest_price, close_price)

            portfolio_value = cash + units * close_price
            cash_values.append(cash)
            portfolio_values.append(portfolio_value)
            position_values.append(units * close_price / portfolio_value if portfolio_value else 0.0)
            turnover_values.append(turnover)

        if current_side != 0:
            timestamp = bars.index[-1]
            trade, cash, units = self._close_trade(
                timestamp=timestamp,
                price=float(bars.iloc[-1]["close"]),
                cash=cash,
                units=units,
                entry_date=entry_date,
                entry_price=entry_price,
                entry_value=entry_value,
                entry_costs=entry_costs,
                entry_slippage=entry_slippage,
                exit_reason="end_of_data",
            )
            trades.append(trade)
            portfolio_values[-1] = cash
            cash_values[-1] = cash
            position_values[-1] = 0.0

        return {
            "cash": pd.Series(cash_values, index=bars.index, dtype="float64"),
            "portfolio_value": pd.Series(portfolio_values, index=bars.index, dtype="float64"),
            "positions": pd.Series(position_values, index=bars.index, dtype="float64"),
            "turnover": pd.Series(turnover_values, index=bars.index, dtype="float64"),
            "trades": trades,
        }

    def _close_trade(
        self,
        timestamp,
        price: float,
        cash: float,
        units: float,
        entry_date,
        entry_price: float,
        entry_value: float,
        entry_costs: float,
        entry_slippage: float,
        exit_reason: str,
    ) -> tuple[Trade, float, float]:
        exit_value = units * price
        fee, slippage = self._execution_costs(abs(exit_value))
        cash += exit_value - fee - slippage
        side = "long" if units > 0 else "short"
        direction = 1 if units > 0 else -1
        gross_return = direction * (price / entry_price - 1) if entry_price else 0.0
        total_costs = entry_costs + fee
        total_slippage = entry_slippage + slippage
        net_pnl = direction * abs(units) * (price - entry_price) - total_costs - total_slippage
        net_return = net_pnl / entry_value if entry_value else 0.0
        holding_days = max((timestamp - entry_date).days, 0) if entry_date is not None else 0
        trade = Trade(
            entry_date=_date_string(entry_date),
            exit_date=_date_string(timestamp),
            entry_price=_safe_round(entry_price, 6),
            exit_price=_safe_round(price, 6),
            position=side,
            holding_days=holding_days,
            gross_return_pct=_pct(gross_return),
            net_return_pct=_pct(net_return),
            transaction_costs=_safe_round(total_costs),
            slippage=_safe_round(total_slippage),
            exit_reason=exit_reason,
        )
        return trade, cash, 0.0

    def _risk_exit_reason(
        self,
        side: int,
        open_price: float,
        entry_price: float,
        highest_price: float,
        lowest_price: float,
        holding_days: int,
        current_drawdown: float,
    ) -> str | None:
        if side == 0 or entry_price <= 0:
            return None
        trade_return = side * (open_price / entry_price - 1)
        if self.risk.stop_loss_pct is not None and trade_return <= -abs(self.risk.stop_loss_pct):
            return "stop_loss"
        if self.risk.take_profit_pct is not None and trade_return >= abs(self.risk.take_profit_pct):
            return "take_profit"
        if self.risk.trailing_stop_pct is not None:
            if side > 0 and highest_price > 0 and open_price / highest_price - 1 <= -abs(self.risk.trailing_stop_pct):
                return "trailing_stop"
            if side < 0 and lowest_price < np.inf and lowest_price / open_price - 1 <= -abs(self.risk.trailing_stop_pct):
                return "trailing_stop"
        if self.risk.max_holding_period is not None and holding_days >= self.risk.max_holding_period:
            return "max_holding_period"
        if self.risk.max_drawdown_pct is not None and current_drawdown <= -abs(self.risk.max_drawdown_pct):
            return "max_drawdown"
        return None

    def _prepare_inputs(self, df: pd.DataFrame, signal: pd.Series | None) -> tuple[pd.DataFrame, pd.Series]:
        if df.empty:
            raise BacktestError("price data is empty")

        bars = df.copy()
        bars.columns = [str(column).strip().lower().replace(" ", "_") for column in bars.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(bars.columns)
        if missing:
            raise BacktestError(f"missing required columns: {sorted(missing)}")

        if signal is None:
            if "signal" not in bars.columns:
                raise BacktestError("signal must be supplied or present as a DataFrame column")
            signal = bars["signal"]

        if not isinstance(bars.index, pd.DatetimeIndex):
            date_column = "date" if "date" in bars.columns else None
            if date_column is None:
                raise BacktestError("DataFrame must have a DatetimeIndex or Date column")
            bars.index = pd.to_datetime(bars[date_column])

        bars = bars.sort_index()
        bars = bars.loc[~bars.index.duplicated(keep="last")]
        numeric_columns = ["open", "high", "low", "close", "volume"]
        bars[numeric_columns] = bars[numeric_columns].apply(pd.to_numeric, errors="coerce")
        bars = bars.dropna(subset=["open", "close"])
        bars = bars[(bars["open"] > 0) & (bars["close"] > 0)]
        if bars.empty:
            raise BacktestError("no valid price rows after cleaning")

        signal = pd.Series(signal, index=df.index if len(signal) == len(df.index) else signal.index)
        signal.index = pd.to_datetime(signal.index)
        signal = signal.reindex(bars.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
        return bars, signal

    def _target_exposure(self, signal: pd.Series) -> pd.Series:
        if self.config.allocation_mode == "full":
            exposure = np.sign(signal).astype(float)
        elif self.config.allocation_mode == "fixed":
            exposure = np.sign(signal).astype(float) * self.config.fixed_allocation
        elif self.config.allocation_mode == "conviction":
            exposure = signal.abs().clip(0.0, 1.0) * np.sign(signal)
        else:
            raise BacktestError(f"unknown allocation_mode: {self.config.allocation_mode}")
        if not self.config.allow_short:
            exposure = exposure.clip(lower=0.0)
        return exposure.clip(-1.0, 1.0)

    def _target_position_value(self, portfolio_value: float, exposure: float) -> float:
        allocation = abs(exposure)
        if self.config.risk_per_trade is not None:
            allocation = min(allocation, abs(self.config.risk_per_trade))
        return portfolio_value * np.sign(exposure) * allocation

    def _execution_costs(self, traded_value: float) -> tuple[float, float]:
        fee = traded_value * self.config.fee_bps / 10_000
        slippage = traded_value * self.config.slippage_bps / 10_000
        return fee, slippage

    def _reserve_execution_costs(self, target_value: float) -> float:
        cost_rate = (self.config.fee_bps + self.config.slippage_bps) / 10_000
        return target_value / (1 + cost_rate) if cost_rate > 0 else target_value

    def _benchmark_comparison(self, bars: pd.DataFrame, strategy_returns: pd.Series) -> dict:
        benchmark_source = self.config.benchmark_column
        benchmark_returns = None
        if benchmark_source and benchmark_source in bars.columns:
            benchmark_returns = bars[benchmark_source].pct_change().fillna(0.0)
        else:
            benchmark_returns = bars["close"].pct_change().fillna(0.0)

        benchmark_equity = self.config.initial_capital * (1 + benchmark_returns).cumprod()
        strategy_equity = self.config.initial_capital * (1 + strategy_returns).cumprod()
        return {
            "benchmark_total_return_pct": _pct(benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1),
            "strategy_total_return_pct": _pct(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1),
            "excess_return_pct": _pct(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]),
            "benchmark_max_drawdown_pct": _pct(_drawdown(benchmark_equity).min()),
            "benchmark_cagr_pct": _pct(_cagr(benchmark_equity)),
        }

    def compute_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        positions: pd.Series | None = None,
        trades: list[Trade] | None = None,
        turnover: pd.Series | None = None,
    ) -> dict:
        returns = returns.fillna(0.0)
        trades = trades or []
        positions = positions if positions is not None else pd.Series(0.0, index=returns.index)
        turnover = turnover if turnover is not None else positions.diff().abs().fillna(0.0)
        drawdown = _drawdown(equity_curve)
        downside = returns[returns < 0]
        annual_vol = returns.std() * np.sqrt(252)
        downside_vol = downside.std() * np.sqrt(252) if len(downside) else 0.0
        annual_return = returns.mean() * 252
        trade_returns = pd.Series([trade.net_return_pct / 100 for trade in trades], dtype="float64")
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())

        return {
            "total_return_pct": _pct(equity_curve.iloc[-1] / self.config.initial_capital - 1),
            "cagr_pct": _pct(_cagr(equity_curve)),
            "annual_return_pct": _pct(annual_return),
            "annualized_vol_pct": _pct(annual_vol),
            "sharpe_ratio": _safe_round(annual_return / annual_vol if annual_vol else 0.0),
            "sortino_ratio": _safe_round(annual_return / downside_vol if downside_vol else 0.0),
            "calmar_ratio": _safe_round(_cagr(equity_curve) / abs(drawdown.min()) if drawdown.min() else 0.0),
            "max_drawdown_pct": _pct(drawdown.min()),
            "average_drawdown_pct": _pct(drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0),
            "win_rate_pct": _pct(len(wins) / len(trade_returns) if len(trade_returns) else 0.0),
            "loss_rate_pct": _pct(len(losses) / len(trade_returns) if len(trade_returns) else 0.0),
            "profit_factor": _safe_round(gross_profit / gross_loss if gross_loss else 0.0),
            "expectancy_pct": _pct(trade_returns.mean() if len(trade_returns) else 0.0),
            "average_trade_return_pct": _pct(trade_returns.mean() if len(trade_returns) else 0.0),
            "average_win_pct": _pct(wins.mean() if len(wins) else 0.0),
            "average_loss_pct": _pct(losses.mean() if len(losses) else 0.0),
            "largest_win_pct": _pct(wins.max() if len(wins) else 0.0),
            "largest_loss_pct": _pct(losses.min() if len(losses) else 0.0),
            "average_holding_period": _safe_round(np.mean([trade.holding_days for trade in trades]) if trades else 0.0),
            "exposure_pct": _pct((positions.abs() > 1e-12).mean() if len(positions) else 0.0),
            "turnover_pct": _pct(turnover.sum() if len(turnover) else 0.0),
            "number_of_trades": len(trades),
            "skewness": _safe_round(returns.skew() if len(returns) > 2 else 0.0),
            "kurtosis": _safe_round(returns.kurtosis() if len(returns) > 3 else 0.0),
        }

    def _validate_config(self) -> None:
        if self.config.initial_capital <= 0:
            raise BacktestError("initial_capital must be positive")
        if self.config.fee_bps < 0 or self.config.slippage_bps < 0:
            raise BacktestError("fee_bps and slippage_bps must be non-negative")
        if not 0 <= self.config.fixed_allocation <= 1:
            raise BacktestError("fixed_allocation must be between 0 and 1")
        if self.config.risk_per_trade is not None and not 0 < self.config.risk_per_trade <= 1:
            raise BacktestError("risk_per_trade must be in (0, 1]")


def _drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax().replace(0, np.nan)
    return (equity_curve / running_max - 1).fillna(0.0)


def _cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty or equity_curve.iloc[0] <= 0:
        return 0.0
    n_years = max(len(equity_curve) / 252, 1e-9)
    ending_ratio = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(ending_ratio ** (1 / n_years) - 1) if ending_ratio > 0 else -1.0


def _pct(value: float) -> float:
    return _safe_round(value * 100)


def _safe_round(value: float, digits: int = 4) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(round(float(value), digits))


def _series_to_records(series: pd.Series, value_name: str) -> list[dict]:
    if series.empty:
        return []
    return [
        {"date": _date_string(index), value_name: _safe_round(value, 6)}
        for index, value in series.items()
    ]


def _date_string(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)
