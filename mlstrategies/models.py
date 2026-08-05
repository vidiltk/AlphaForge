import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

# Suppress LightGBM warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

class MLModelWrapper:
    def __init__(self, model, is_classifier=False):
        self.model = model
        self.scaler = StandardScaler()
        self.is_classifier = is_classifier
        self.is_fitted = False

    def _get_clean_df(self, df):
        df_clean = df.copy().dropna()
        valid_features = [col for col in df_clean.columns if col not in ['target', 'signal']]
        return df_clean, valid_features

    def fit(self, df):
        df_clean, valid_features = self._get_clean_df(df)
        if df_clean.empty:
            return
        
        # Predict next period's return
        df_clean['target'] = df_clean['Close'].pct_change().shift(-1)
        df_clean = df_clean.dropna()
        
        if self.is_classifier:
            # XGBoost and LightGBM expect classes starting from 0 (0 for down, 1 for up)
            y = np.where(df_clean['target'] > 0, 1, 0)
        else:
            y = df_clean['target'].values
            
        X = df_clean[valid_features]
        
        # 80-20 Chronological split
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y[:split_idx]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

    def predict(self, df):
        df_clean, valid_features = self._get_clean_df(df)
        if df_clean.empty or not self.is_fitted:
            return pd.Series(0, index=df.index)
            
        X_scaled = self.scaler.transform(df_clean[valid_features])
        signals = np.zeros(len(df_clean))
        
        if self.is_classifier:
            probabilities = self.model.predict_proba(X_scaled)
            try:
                class_1_index = list(self.model.classes_).index(1)
                prob_going_up = probabilities[:, class_1_index]
                
                # 0.65 threshold logic
                confidence_threshold = 0.65
                signals[prob_going_up >= confidence_threshold] = 1
                signals[prob_going_up <= (1 - confidence_threshold)] = -1
                
            except ValueError:
                pass
                
        else:
            # Regression threshold logic
            raw_predictions = self.model.predict(X_scaled)
            threshold = 0.0005
            signals[raw_predictions > threshold] = 1
            signals[raw_predictions < -threshold] = -1
            
        out_series = pd.Series(0, index=df.index)
        out_series.loc[df_clean.index] = signals
        return out_series


# --- 3 Regression Strategies ---

class LinearRegStrategy(MLModelWrapper):
    def __init__(self):
        super().__init__(model=LinearRegression(), is_classifier=False)

class RidgeRegStrategy(MLModelWrapper):
    def __init__(self):
        super().__init__(model=Ridge(random_state=42), is_classifier=False)

class LassoRegStrategy(MLModelWrapper):
    def __init__(self):
        super().__init__(model=Lasso(alpha=0.001, random_state=42), is_classifier=False)


# --- 3 Classification Strategies ---

class RandomForestStrategy(MLModelWrapper):
    def __init__(self):
        super().__init__(model=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), is_classifier=True)

class XGBoostStrategy(MLModelWrapper):
    def __init__(self):
        super().__init__(model=XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, random_state=42), is_classifier=True)

class LightGBMStrategy(MLModelWrapper):
    def __init__(self):
        super().__init__(model=LGBMClassifier(max_depth=5, random_state=42, verbose=-1), is_classifier=True)
