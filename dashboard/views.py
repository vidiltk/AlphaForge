import json
from datetime import date, timedelta

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from marketdata.services import fetch_price_history, refresh_fundamentals, TickerNotFound
from signals.base import IndicatorRegistry
from signals.indicators import combine_signals
from backtest.engine import BaseBacktester
from mlstrategies.base import MLStrategyRegistry
import mlstrategies.models  # to trigger registration
import pandas as pd


def index(request):
    """Renders the form: ticker input, indicator multi-select + weights, date range."""
    context = {"indicators": list(IndicatorRegistry.all().keys())}
    return render(request, "dashboard/index.html", context)


@csrf_exempt
def run_backtest(request):
    """
    POST JSON body — supports both normal and regime-switching modes.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        symbol = payload["symbol"].strip().upper()
    except (json.JSONDecodeError, KeyError, AttributeError):
        return JsonResponse({"error": "malformed request: 'symbol' is required"}, status=400)

    start = date.fromisoformat(payload.get("start", str(date.today() - timedelta(days=5 * 365))))
    end = date.fromisoformat(payload.get("end", str(date.today())))

    fetch_start = start - timedelta(days=3 * 365)

    try:
        df = fetch_price_history(symbol, start=fetch_start, end=end)
        ticker = refresh_fundamentals(symbol)
    except TickerNotFound as exc:
        return JsonResponse({"error": str(exc)}, status=404)

    strategy_type = payload.get("strategy_type", "indicator")
    start_dt = pd.to_datetime(start)
    regime_info = None

    if strategy_type == "ml":
        train_df = df[df.index < start_dt]
        test_df = df[df.index >= start_dt]

        if test_df.empty:
            return JsonResponse({"error": "No price data in the selected test period."}, status=400)

        regime_config = payload.get("regime_switching")

        if regime_config and regime_config.get("enabled"):
            # ── Regime Switching Mode ──
            from mlstrategies.regime import RegimeSwitchingStrategy

            detector_name = regime_config.get("method", "atr")
            model_map = regime_config.get("model_map", {})

            if not model_map:
                return JsonResponse(
                    {"error": "regime switching enabled but no model_map provided"},
                    status=400,
                )

            strategy = RegimeSwitchingStrategy(detector_name, model_map)
            signal, regime_info = strategy.run(train_df, test_df)
            backtest_df = test_df
        else:
            # ── Normal Single-Model Mode (unchanged) ──
            ml_model = payload.get("ml_model")
            if not ml_model:
                return JsonResponse({"error": "ml_model is required for ML strategy"}, status=400)
            try:
                strategy_cls = MLStrategyRegistry.get(ml_model)
            except KeyError:
                return JsonResponse(
                    {"error": f"unknown ML model '{ml_model}'. "
                              f"available: {list(MLStrategyRegistry.all().keys())}"},
                    status=400,
                )
            instance = strategy_cls()
            signal = instance.run(train_df, test_df)
            backtest_df = test_df
    else:
        weighted = []
        for item in payload.get("indicators", []):
            try:
                indicator_cls = IndicatorRegistry.get(item["name"])
            except KeyError:
                return JsonResponse(
                    {"error": f"unknown indicator '{item.get('name')}'. "
                              f"available: {list(IndicatorRegistry.all().keys())}"},
                    status=400,
                )
            instance = indicator_cls(**item.get("params", {}))
            weighted.append((instance, item.get("weight", 1.0)))

        if not weighted:
            return JsonResponse({"error": "select at least one indicator"}, status=400)

        full_signal = combine_signals(df, weighted)
        backtest_df = df[df.index >= start_dt]
        if backtest_df.empty:
            return JsonResponse({"error": "No price data in the selected test period."}, status=400)

        raw_signal = full_signal.reindex(backtest_df.index).fillna(0.0)
        import numpy as np
        threshold = 0.3
        signal_arr = np.where(raw_signal > threshold, 1.0, np.where(raw_signal < -threshold, -1.0, 0.0))
        signal = pd.Series(signal_arr, index=raw_signal.index)

    backtester = BaseBacktester(
        initial_capital=payload.get("initial_capital", 100_000.0),
        fee_bps=payload.get("fee_bps", 10.0),
    )
    result = backtester.run(backtest_df, signal)

    response_data = {
        "symbol": symbol,
        "backtest": result.to_dict(),
        "fundamentals": {
            "short_name": ticker.short_name,
            "sector": ticker.sector,
            "industry": ticker.industry,
            "market_cap": ticker.market_cap,
            "pe_ratio": ticker.pe_ratio,
            "forward_pe": ticker.forward_pe,
            "eps": ticker.eps,
            "dividend_yield": ticker.dividend_yield,
            "fifty_two_week_high": ticker.fifty_two_week_high,
            "fifty_two_week_low": ticker.fifty_two_week_low,
            "beta": ticker.beta,
        },
    }

    if regime_info is not None:
        response_data["regime_info"] = regime_info

    return JsonResponse(response_data)
