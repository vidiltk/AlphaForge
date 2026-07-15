import json
from datetime import date, timedelta

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from marketdata.services import fetch_price_history, refresh_fundamentals, TickerNotFound
from signals.base import IndicatorRegistry
from signals.indicators import combine_signals
from backtest.engine import BaseBacktester


def index(request):
    """Renders the form: ticker input, indicator multi-select + weights, date range."""
    context = {"indicators": list(IndicatorRegistry.all().keys())}
    return render(request, "dashboard/index.html", context)


@csrf_exempt  # NOTE: fine for local dev; before any real deployment, either
# add {% csrf_token %} + fetch the cookie from the frontend, or switch this
# to DRF with proper session/token auth.
def run_backtest(request):
    """
    POST JSON body:
    {
        "symbol": "AAPL",
        "start": "2020-01-01",
        "end": "2025-01-01",
        "indicators": [{"name": "sma_crossover", "weight": 0.5, "params": {"fast": 10, "slow": 30}},
                        {"name": "rsi", "weight": 0.5, "params": {"window": 14}}],
        "initial_capital": 100000,
        "fee_bps": 5
    }
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

    try:
        df = fetch_price_history(symbol, start=start, end=end)
        ticker = refresh_fundamentals(symbol)
    except TickerNotFound as exc:
        return JsonResponse({"error": str(exc)}, status=404)

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

    signal = combine_signals(df, weighted)

    backtester = BaseBacktester(
        initial_capital=payload.get("initial_capital", 100_000.0),
        fee_bps=payload.get("fee_bps", 5.0),
    )
    result = backtester.run(df, signal)

    return JsonResponse({
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
    })
