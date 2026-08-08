import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

from .base import BaseMLStrategy, MLStrategyRegistry

warnings.filterwarnings("ignore", category=UserWarning)

# --- Feature Engineering ---

def atr(df, period=14):
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['prev_close']).abs()
    tr3 = (df['low'] - df['prev_close']).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(span=period, adjust=False).mean()
    return df['ATR']

def compute_keltner_bands(df, period=20, multiplier=1):
    df = df.copy()
    df['KC_EMA'] = df['close'].ewm(span=period, adjust=False).mean()
    df['ATR'] = atr(df, period=period)
    df['KC_Upper'] = df['KC_EMA'] + multiplier * df['ATR']
    df['KC_Lower'] = df['KC_EMA'] - multiplier * df['ATR']
    
    signal = [0] * len(df)
    last_sig = 0
    close_vals = df['close'].values
    kc_up = df['KC_Upper'].values
    kc_dn = df['KC_Lower'].values
    for i in range(len(df)):
        if close_vals[i] > kc_up[i]: sig = 1
        elif close_vals[i] < kc_dn[i]: sig = -1
        else: sig = 0
        if sig == last_sig: sig = 0
        else:
            if sig != 0: last_sig = sig
        signal[i] = sig
    df['KC_signal'] = signal
    return df

def add_indicators(df, atr_window=15, multiplier=1):
    df = df.copy()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df = compute_keltner_bands(df, period=atr_window, multiplier=multiplier)
    
    # Matching exact notebook features
    df["ATR_signal"] = 0.0
    df["ATR_Upper"] = 0.0
    df["ATR_Lower"] = 0.0
    return df

# --- Signal Processing ---

def boost_prob(p, factor=3):
    return np.tanh((p - 0.5) * factor)

def compute_vol_scale(raw_signal, asset_rets, lookback=40, target_vol=0.25, min_floor=1e-3, smooth_window=10):
    raw = pd.Series(raw_signal, index=asset_rets.index).fillna(0.0)
    strat_rets = raw * asset_rets
    roll_vol = strat_rets.rolling(lookback).std() * np.sqrt(252)
    roll_vol = roll_vol.bfill().clip(lower=min_floor)
    raw_scale = target_vol / roll_vol
    smoothed_scale = raw_scale.rolling(smooth_window).mean().fillna(0.0)
    return smoothed_scale.shift(1).fillna(0.0)

FEAT_COLS = ["ATR_signal", "KC_signal", "sma_50", "sma_200", "ema_50", "ema_200", "KC_Upper", "KC_Lower", "ATR_Upper", "ATR_Lower"]

# --- Base Wrapper ---

class MLModelWrapper(BaseMLStrategy):
    def __init__(self, model, **hyperparams):
        super().__init__(**hyperparams)
        self.model = model
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df):
        df_feats = add_indicators(df)
        # 1 for up, 0 for down
        df_feats['target'] = (df_feats['close'].pct_change().shift(-1) > 0).astype(int)
        
        df_clean = df_feats.dropna()
        if df_clean.empty:
            return
            
        X = df_clean[FEAT_COLS]
        y = df_clean['target'].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def get_probs(self, X_scaled):
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, df):
        if not self.is_fitted:
            return pd.Series(0.0, index=df.index)
            
        df_feats = add_indicators(df)
        X = df_feats[FEAT_COLS].fillna(0.0)
        
        X_scaled = self.scaler.transform(X)
        probs = self.get_probs(X_scaled)
        
        boosted = boost_prob(probs, factor=3)
        boosted_series = pd.Series(boosted, index=df.index)
        boosted_series[np.abs(boosted_series) < 0.20] = 0.0 
        
        ret = df['close'].pct_change().fillna(0.0)
        scale = compute_vol_scale(boosted_series, ret, lookback=40, target_vol=0.25)
        targets = boosted_series * scale
        
        return targets.fillna(0.0)

# --- Strategies ---

@MLStrategyRegistry.register
class LogisticRegStrategy(MLModelWrapper):
    name = "logistic"
    def __init__(self):
        super().__init__(model=LogisticRegression(max_iter=500, penalty="l2", solver='lbfgs', random_state=42))

@MLStrategyRegistry.register
class XGBoostStrategy(MLModelWrapper):
    name = "xgboost"
    def __init__(self):
        super().__init__(model=XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss'))

@MLStrategyRegistry.register
class LightGBMStrategy(MLModelWrapper):
    name = "lightgbm"
    def __init__(self):
        super().__init__(model=LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1))


@MLStrategyRegistry.register
class EnsembleStrategy(BaseMLStrategy):
    name = "ensemble"
    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)
        self.models = {
            "logreg": LogisticRegression(max_iter=500, penalty="l2", solver='lbfgs', random_state=42),
            "xgb": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "lgb": LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
        }
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df):
        df_feats = add_indicators(df)
        df_feats['target'] = (df_feats['close'].pct_change().shift(-1) > 0).astype(int)
        
        df_clean = df_feats.dropna()
        if df_clean.empty:
            return
            
        X = df_clean[FEAT_COLS]
        y = df_clean['target'].values
        
        X_scaled = self.scaler.fit_transform(X)
        for m in self.models.values():
            m.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, df):
        if not self.is_fitted:
            return pd.Series(0.0, index=df.index)
            
        df_feats = add_indicators(df)
        X = df_feats[FEAT_COLS].fillna(0.0)
        
        X_scaled = self.scaler.transform(X)
        
        probs = np.zeros(len(X_scaled))
        for m in self.models.values():
            probs += m.predict_proba(X_scaled)[:, 1]
        probs /= len(self.models)
        
        boosted = boost_prob(probs, factor=3)
        boosted_series = pd.Series(boosted, index=df.index)
        boosted_series[np.abs(boosted_series) < 0.20] = 0.0 
        
        ret = df['close'].pct_change().fillna(0.0)
        scale = compute_vol_scale(boosted_series, ret, lookback=40, target_vol=0.25)
        targets = boosted_series * scale
        
        return targets.fillna(0.0)
