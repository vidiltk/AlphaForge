from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/run-backtest/", views.run_backtest, name="run_backtest"),
]
