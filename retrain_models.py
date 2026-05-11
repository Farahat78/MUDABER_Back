"""
Model Retraining Module
Continuously updates models with new data
"""

import os
import pandas as pd
import numpy as np
import pickle
import re
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')


class ModelRetrainer:
    def __init__(self, model_dir="."):
        self.model_dir = model_dir
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

    def update_and_load_training_data(self, latest_data_path=None):
        """Load historical data, merge with new scraped data, and save back"""
        print("[1] Accumulating and loading training data...")
        
        training_dir = os.path.join(self.model_dir, "data", "training")
        os.makedirs(training_dir, exist_ok=True)
        full_data_path = os.path.join(training_dir, "full_training_data.csv")
        temp_data_path = os.path.join(training_dir, "full_training_data_temp.csv")

        # 1. Load historical dataset
        if os.path.exists(full_data_path):
            print(f"    Loading historical dataset: {full_data_path}")
            df_hist = pd.read_csv(full_data_path, encoding='utf-8-sig')
            hist_rows = len(df_hist)
        else:
            print("    No historical dataset found. Initializing new one.")
            df_hist = pd.DataFrame()
            hist_rows = 0

        # 2. Load latest scraped dataset
        # IMPORTANT: We trust the 'date' column already written by the scraper on the day it ran.
        # Each weekly scraping session produces a CSV with that week's date embedded in every row.
        # We do NOT override this date - doing so would cause all sessions to collide on today's date.
        if latest_data_path and os.path.exists(latest_data_path):
            print(f"    Loading new data: {latest_data_path}")
            df_new = pd.read_csv(latest_data_path, encoding='utf-8-sig')
            
            # If the 'date' column is missing or all-null, extract date from the filename as fallback
            if 'date' not in df_new.columns or df_new['date'].isna().all():
                import re as _re
                fname = os.path.basename(latest_data_path)
                match = _re.search(r'(\d{4}-\d{2}-\d{2})', fname)
                if match:
                    fallback_date = match.group(1)
                    df_new['date'] = fallback_date
                    print(f"    'date' column was missing - extracted from filename: {fallback_date}")
                else:
                    df_new['date'] = str(datetime.now().date())
                    print(f"    'date' column was missing - using today as fallback")
            
            new_rows = len(df_new)
            scraped_date = pd.to_datetime(df_new['date'], errors='coerce').dt.date.iloc[0]
            print(f"    Loaded {new_rows:,} new rows | Scraping date: {scraped_date}")
        else:
            print(f"    Warning: New data path not found or not provided: {latest_data_path}")
            df_new = pd.DataFrame()
            new_rows = 0

        if df_hist.empty and df_new.empty:
            raise ValueError("No data available for training.")

        # 3. Validate schema before merge
        if not df_hist.empty and not df_new.empty:
            required_cols = set(df_hist.columns) - {'product_name_clean'}
            new_cols = set(df_new.columns)
            missing_cols = required_cols - new_cols
            if missing_cols:
                raise ValueError(f"Schema validation failed. New data is missing required columns: {missing_cols}")

        # 4. Merge datasets
        df_combined = pd.concat([df_hist, df_new], ignore_index=True)
        
        # Ensure product_name_clean exists for deduplication
        if 'product_name_clean' not in df_combined.columns:
            df_combined['product_name_clean'] = df_combined['product_name'].apply(
                lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()) if pd.notna(x) else ""
            )

        # 5. Remove duplicates
        # Rule: ONE record per product per calendar day.
        # Same product scraped on a new day = new row (preserved).
        # Same product scraped multiple times on same day = keep latest (deduplicated).
        df_combined['date'] = pd.to_datetime(df_combined['date'], errors='coerce').dt.date
        rows_before = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['product_name_clean', 'date'], keep='last')
        dups_removed = rows_before - len(df_combined)

        # 6. Sort data
        df_combined = df_combined.sort_values(['product_name_clean', 'date']).reset_index(drop=True)

        # 7. Log statistics
        print("    --- Data Accumulation Stats ---")
        print(f"    Old: {hist_rows:,} rows")
        print(f"    New: {new_rows:,} rows")
        print(f"    After merge: {len(df_combined):,} rows")
        print(f"    Duplicates removed: {dups_removed:,}")
        
        min_date = df_combined['date'].min()
        max_date = df_combined['date'].max()
        unique_products = df_combined['product_name_clean'].nunique()
        print(f"    Integrity: {unique_products:,} unique products from {min_date} to {max_date}")
        print("    -------------------------------")

        # Check if we should retrain
        actual_new_rows = len(df_combined) - hist_rows
        
        threshold = 10
        try:
            from config_loader import Config
            config = Config(os.path.join(self.model_dir, "config.yaml"))
            threshold = config.get("training.retrain_threshold", 10)
        except:
            pass

        if not df_hist.empty and actual_new_rows < threshold:
            print(f"    SKIPPED: Only {actual_new_rows} new unique rows added. Threshold is {threshold}.")
            self._insufficient_new_data = True
        else:
            self._insufficient_new_data = False
            if not df_hist.empty:
                print(f"    TRIGGERED: {actual_new_rows} new rows meet threshold {threshold}.")

        # 8. Safe File Writing
        print("    Saving updated dataset safely...")
        df_combined.to_csv(temp_data_path, index=False, encoding='utf-8-sig')
        
        if os.path.exists(full_data_path):
            os.remove(full_data_path)
        os.rename(temp_data_path, full_data_path)

        return df_combined

    def preprocess_for_training(self, df):
        """Apply full preprocessing pipeline for training"""
        print("[2] Preprocessing data for training...")

        # Clean data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0)
        df['main_category'] = df['main_category'].fillna('')
        df['sub_category'] = df['sub_category'].fillna('')
        df['product_name'] = df['product_name'].fillna('')
        df = df.dropna(subset=['price', 'date'])
        df = df.drop_duplicates(subset=['product_name', 'date'])

        # Check if we have enough data
        date_count = df['date'].nunique()
        if date_count < 2:
            print("    Warning: Less than 2 unique dates. Cannot create training data.")
            print("    Need at least 2 dates per product to create lag features and target.")
            return pd.DataFrame()

        # Normalize names
        df['product_name_clean'] = df['product_name'].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()) if pd.notna(x) else ""
        )

        # Sort for lag computation
        df = df.sort_values(['product_name_clean', 'date']).reset_index(drop=True)

        # Create lag features
        def create_lags(g):
            g = g.sort_values('date')
            for lag in [1, 2, 3]:
                g[f'price_lag_{lag}'] = g['price'].shift(lag).ffill().fillna(g['price'])
            g['rolling_mean_2'] = g['price'].shift(1).rolling(2, min_periods=1).mean().ffill().fillna(g['price'])
            g['rolling_mean_3'] = g['price'].shift(1).rolling(3, min_periods=1).mean().ffill().fillna(g['price'])
            return g

        df = df.groupby('product_name_clean', group_keys=False).apply(create_lags)
        df = df.dropna(subset=['price_lag_1', 'price_lag_2', 'price_lag_3'])

        # Encode categories
        self.label_encoders['main'] = LabelEncoder()
        self.label_encoders['sub'] = LabelEncoder()
        df['main_category_encoded'] = self.label_encoders['main'].fit_transform(
            df['main_category'].astype(str)
        )
        df['sub_category_encoded'] = self.label_encoders['sub'].fit_transform(
            df['sub_category'].astype(str)
        )

        # Add time features
        df['week_number'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek

        # Create target (next price)
        df = df.sort_values(['product_name_clean', 'date']).reset_index(drop=True)
        df['price_next'] = df.groupby('product_name_clean')['price'].shift(-1)
        df = df.dropna(subset=['price_next'])

        if len(df) == 0:
            print("    Warning: No valid training samples after creating target.")
            return pd.DataFrame()

        # Add safe engineered features
        df['category_avg_price'] = df.groupby('main_category_encoded')['price'].transform('mean')
        df['category_avg_price_week'] = df.groupby(
            ['main_category_encoded', 'week_number']
        )['price'].transform('mean')

        df['price_change_frequency'] = df.groupby('product_name_clean')['price'].transform(
            lambda x: (x.diff() != 0).rolling(3, min_periods=1).sum()
        ).fillna(0)

        df['price_momentum'] = df['price_lag_1'] - df['price_lag_2']
        df['price_vs_category'] = df['price_lag_1'] - df['category_avg_price']
        df['trend_indicator'] = df['price_lag_1'] - df['rolling_mean_3']

        # Remove products with only one date
        product_counts = df.groupby('product_name_clean')['date'].nunique()
        products_multi_date = product_counts[product_counts >= 2].index
        df = df[df['product_name_clean'].isin(products_multi_date)]

        print(f"    Final shape: {df.shape}")
        print(f"    Products: {df['product_name_clean'].nunique()}")
        if len(df) > 0:
            print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        return df

    def prepare_training_data(self, df):
        """Prepare X and y for training"""
        X = df[self.feature_columns]
        y = df['price_next']
        return X, y

    def train_models(self, X, y):
        """Train all three models"""
        print("[3] Training models...")

        if len(X) == 0 or len(y) == 0:
            print("    Error: No training data available.")
            return None

        models = {}

        # XGBoost
        print("    Training XGBoost...")
        xgb_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X, y)
        models['xgb'] = xgb_model

        # LightGBM
        print("    Training LightGBM...")
        lgb_model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X, y)
        models['lgb'] = lgb_model

        # CatBoost
        print("    Training CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False
        )
        cat_model.fit(X, y)
        models['cat'] = cat_model

        print("    All models trained successfully")
        return models

    def evaluate_models(self, models, X, y):
        """Evaluate model performance"""
        print("[4] Evaluating models...")

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        results = {}
        for name, model in models.items():
            pred = model.predict(X)
            results[name] = {
                'MAE': mean_absolute_error(y, pred),
                'RMSE': np.sqrt(mean_squared_error(y, pred)),
                'R2': r2_score(y, pred)
            }
            print(f"    {name.upper()}: MAE={results[name]['MAE']:.2f}, RMSE={results[name]['RMSE']:.2f}, R2={results[name]['R2']:.4f}")

        return results

    def save_models(self, models, metrics: dict = None):
        """Save updated models to disk with versioning"""
        print("[5] Saving updated models...")

        # Use versioning module
        from model_versioning import ModelVersionManager
        version_manager = ModelVersionManager(self.model_dir)

        # Save with versioning
        version_tag = version_manager.save_versioned_models(models, metrics)

        # Also save label encoders for consistency
        with open(f"{self.model_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print("    Saved: label_encoders.pkl")

        return version_tag

    def save_updated_dataset(self, df, output_path="final_price_forecasting_dataset.csv"):
        """Save the updated dataset"""
        print(f"[6] Saving updated dataset to {output_path}...")

        features = self.feature_columns + ['price_next', 'product_name_clean', 'date']
        df_final = df[features].reset_index(drop=True)
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"    Saved: {len(df_final)} rows")

    def retrain(self, latest_data_path=None, output_dataset="final_price_forecasting_dataset.csv"):
        """Run complete retraining pipeline"""
        print("="*60)
        print("PRICE FORECASTING - MODEL RETRAINING")
        print("="*60 + "\n")

        # Load data
        df = self.update_and_load_training_data(latest_data_path)

        if getattr(self, '_insufficient_new_data', False):
            print("\nSkipping retraining: Not enough new data added since last run (didn't meet required threshold).")
            return None

        # Preprocess
        df = self.preprocess_for_training(df)

        if len(df) == 0:
            print("\nWarning: Not enough data to retrain models.")
            print("Need at least 2 dates per product with price history.")
            print("Skipping retraining...")
            return None

        # Prepare training data
        X, y = self.prepare_training_data(df)

        # Train models
        models = self.train_models(X, y)

        if models is None:
            print("Retraining failed.")
            return None

        # Evaluate
        self.evaluate_models(models, X, y)

        # Save models
        self.save_models(models)

        # Save updated dataset
        self.save_updated_dataset(df, output_dataset)

        print("\n" + "="*60)
        print("RETRAINING COMPLETE")
        print("="*60)

        return models


def main():
    retrainer = ModelRetrainer()
    retrainer.retrain()


if __name__ == '__main__':
    main()
