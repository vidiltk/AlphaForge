from django.contrib import admin
from .models import Ticker, PriceBar


@admin.register(Ticker)
class TickerAdmin(admin.ModelAdmin):
    list_display = ("symbol", "short_name", "sector", "pe_ratio", "fundamentals_updated_at")
    search_fields = ("symbol", "short_name")


@admin.register(PriceBar)
class PriceBarAdmin(admin.ModelAdmin):
    list_display = ("ticker", "date", "open", "close", "volume")
    list_filter = ("ticker",)
