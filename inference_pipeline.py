"""
Price Forecasting Inference Pipeline
Robust production-ready version (safe execution)
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class PriceForecastingPipeline:
    def __init__(self, model_dir="."):
        self.model_dir = model_dir
        self.models = {}
        self.label_encoders = {}

        self.feature_columns = [
            'price_lag_1', 'price_lag_2', 'price_lag_3',
            'rolling_mean_2', 'rolling_mean_3',
            'category_avg_price', 'category_avg_price_week',
            'discount', 'price_change_frequency',
            'price_momentum', 'price_vs_category', 'trend_indicator',
            'main_category_encoded', 'sub_category_encoded',
            'week_number', 'month', 'day_of_week'
        ]

        self.historical_data = None
        self.threshold = 1.0

    # ─────────────────────────────
    # Load Models
    # ─────────────────────────────
    def load_models(self):
        print("[1] Loading trained models...")

        model_files = {
            'xgb': 'xgb_model.pkl',
            'lgb': 'lgb_model.pkl',
            'cat': 'cat_model.pkl'
        }

        for name, file in model_files.items():
            try:
                with open(f"{self.model_dir}/{file}", 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"    Loaded: {file}")
            except FileNotFoundError:
                print(f"    Warning: {file} not found")

        if not self.models:
            raise ValueError("No models loaded.")

        return self

    # ─────────────────────────────
    # Load Historical Data
    # ─────────────────────────────
    def load_historical_data(self):
        print("[2] Loading historical data...")

        state_path = f"{self.model_dir}/price_state.csv"

        if os.path.exists(state_path):
            self.historical_data = pd.read_csv(state_path)
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'], errors='coerce')
            print(f"    Loaded state: {len(self.historical_data)} rows")
        else:
            self.historical_data = pd.DataFrame()
            print("    No historical data found")

        return self

    # ─────────────────────────────
    # Preprocessing
    # ─────────────────────────────
    def preprocess_data(self, new_data_path):
        print("[3] Preprocessing...")

        path = os.path.join(self.model_dir, new_data_path)

        if not os.path.exists(path):
            print("    File not found!")
            return pd.DataFrame()

        new_df = pd.read_csv(path)

        # ── basic cleaning ──
        new_df['date'] = pd.to_datetime(new_df.get('date'), errors='coerce')
        new_df['price'] = pd.to_numeric(new_df.get('price'), errors='coerce')
        new_df['discount'] = pd.to_numeric(new_df.get('discount'), errors='coerce').fillna(0)

        for col in ['main_category', 'sub_category', 'product_name']:
            if col not in new_df.columns:
                new_df[col] = ""

        # ── SAFE product_name_clean ──
        new_df['product_name_clean'] = new_df['product_name'].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).lower().strip())
        )

        # ── combine history ──
        state_path = f"{self.model_dir}/price_state.csv"

        if os.path.exists(state_path):
            hist = pd.read_csv(state_path)
            combined_df = pd.concat([hist, new_df], ignore_index=True)
        else:
            combined_df = new_df.copy()

        # ── safety cleaning ──
        combined_df = combined_df.dropna(subset=['price', 'date'])
        combined_df = combined_df.sort_values(['product_name_clean', 'date']).reset_index(drop=True)

        # ── LAGS ──
        def create_lags(g):
            g = g.sort_values('date')
            for lag in [1, 2, 3]:
                g[f'price_lag_{lag}'] = g['price'].shift(lag)

            g['rolling_mean_2'] = g['price'].shift(1).rolling(2, min_periods=1).mean()
            g['rolling_mean_3'] = g['price'].shift(1).rolling(3, min_periods=1).mean()

            return g

        combined_df = combined_df.groupby('product_name_clean', group_keys=False).apply(create_lags)

        # ── ENCODING ──
        combined_df['main_category'] = combined_df['main_category'].fillna("")
        combined_df['sub_category'] = combined_df['sub_category'].fillna("")

        if 'label_encoders.pkl' in os.listdir(self.model_dir):
            with open(f"{self.model_dir}/label_encoders.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)
        else:
            self.label_encoders['main'] = LabelEncoder()
            self.label_encoders['sub'] = LabelEncoder()

            combined_df['main_category_encoded'] = self.label_encoders['main'].fit_transform(
                combined_df['main_category']
            )
            combined_df['sub_category_encoded'] = self.label_encoders['sub'].fit_transform(
                combined_df['sub_category']
            )

        # ── time features ──
        combined_df['week_number'] = combined_df['date'].dt.isocalendar().week.astype(int)
        combined_df['month'] = combined_df['date'].dt.month
        combined_df['day_of_week'] = combined_df['date'].dt.dayofweek

        # ── safe features ──
        combined_df['category_avg_price'] = combined_df.groupby('main_category')['price'].transform('mean')
        combined_df['category_avg_price_week'] = combined_df.groupby(
            ['main_category', 'week_number']
        )['price'].transform('mean')

        combined_df['price_change_frequency'] = combined_df.groupby('product_name_clean')['price'] \
            .transform(lambda x: (x.diff() != 0).rolling(3, min_periods=1).sum())

        combined_df['price_momentum'] = combined_df['price_lag_1'] - combined_df['price_lag_2']
        combined_df['price_vs_category'] = combined_df['price_lag_1'] - combined_df['category_avg_price']
        combined_df['trend_indicator'] = combined_df['price_lag_1'] - combined_df['rolling_mean_3']

        # ❌ FIX IMPORTANT: no filtering to single date
        processed = combined_df.copy()

        # ── SAFE FEATURE FIX ──
        for col in self.feature_columns:
            if col not in processed.columns:
                processed[col] = 0

        processed[self.feature_columns] = processed[self.feature_columns].fillna(0)

        # update state
        self._update_state(new_df)

        print(f"    Processed rows: {len(processed)}")

        return processed

    # ─────────────────────────────
    # Prediction
    # ─────────────────────────────
    def predict(self, df):
        print("[4] Predicting...")

        X = df[self.feature_columns].fillna(0)

        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)

        return preds

    # ─────────────────────────────
    # Ensemble
    # ─────────────────────────────
    def ensemble(self, preds):
        print("[5] Ensemble...")

        return np.mean(pd.DataFrame(preds), axis=1).values

    # ─────────────────────────────
    # Decision
    # ─────────────────────────────
    def generate_decision(self, df, pred):
        print("[6] Decision...")

        result = df.copy()
        result['current_price'] = result['price']
        result['predicted_price'] = pred

        result['price_change'] = result['predicted_price'] - result['current_price']
        result['change_percentage'] = (result['price_change'] / result['current_price']) * 100

        result['trend_label'] = 'STABLE'
        result.loc[result['price_change'] > self.threshold, 'trend_label'] = 'UP'
        result.loc[result['price_change'] < -self.threshold, 'trend_label'] = 'DOWN'

        return result[[
            'product_name_clean',
            'date',
            'current_price',
            'predicted_price',
            'price_change',
            'change_percentage',
            'trend_label'
        ]]

    # ─────────────────────────────
    # State
    # ─────────────────────────────
    def _update_state(self, df):
        state_path = f"{self.model_dir}/price_state.csv"

        df[['product_name_clean', 'date', 'price',
            'main_category', 'sub_category', 'discount']].to_csv(
            state_path, index=False
        )

    # ─────────────────────────────
    # Run pipeline
    # ─────────────────────────────
    def run_inference(self, new_data_path):
        self.load_models()
        self.load_historical_data()

        df = self.preprocess_data(new_data_path)
        preds = self.predict(df)
        ensemble = self.ensemble(preds)
        return self.generate_decision(df, ensemble)


def main():
    pipeline = PriceForecastingPipeline()
    result = pipeline.run_inference("carrefour_products-2026-04-28.csv")

    print(result.head())


if __name__ == "__main__":
    main()
