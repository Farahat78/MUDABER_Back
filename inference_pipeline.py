"""
Price Forecasting Inference Pipeline
Production-ready inference system with ensemble predictions and continuous learning
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

    def load_models(self):
        """Load all trained models from disk"""
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
            raise ValueError("No models loaded. Check model directory.")
        return self

    def load_historical_data(self, csv_files=None):
        """Load historical training data for lag feature computation"""
        print("[2] Loading historical data...")

        # Try to load from state file first (production approach)
        state_path = f"{self.model_dir}/price_state.csv"
        if os.path.exists(state_path):
            self.historical_data = pd.read_csv(state_path, encoding='utf-8-sig')
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'], errors='coerce')
            print(f"    Loaded from state: {len(self.historical_data)} rows")
            return self

        # Fall back to CSV files if no state file
        if csv_files is None:
            csv_files = [
                'carrefour_products-2026-04-24.csv',
                'carrefour_products-2026-04-17.csv',
                'carrefour_products-2026-04-11.csv',
                'carrefour_products-3-25-2026.csv',
                'carrefour_products-3-15-2026.csv',
                'carrefour_products-3-5-2026.csv',
                'carrefour_products-3-10-2026.csv'
            ]

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f"{self.model_dir}/{f}", encoding='utf-8-sig')
                dfs.append(df)
            except FileNotFoundError:
                print(f"    Warning: {f} not found, skipping...")

        if not dfs:
            print("    No historical data found. Will use current data for lags.")
            self.historical_data = pd.DataFrame()
        else:
            self.historical_data = pd.concat(dfs, ignore_index=True)
            print(f"    Loaded: {len(self.historical_data)} rows")

        return self

    def preprocess_data(self, new_data_path):
        """Preprocess new scraped data with feature engineering"""
        print("[3] Preprocessing new data...")

        # Handle both absolute and relative paths
        if not os.path.isabs(new_data_path):
            new_data_path = os.path.join(self.model_dir, new_data_path)

        if not os.path.exists(new_data_path):
            print(f"    Error: File not found: {new_data_path}")
            return pd.DataFrame()

        # Load new data
        new_df = pd.read_csv(new_data_path, encoding='utf-8-sig')
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df['price'] = pd.to_numeric(new_df['price'], errors='coerce')
        new_df['discount'] = pd.to_numeric(new_df['discount'], errors='coerce').fillna(0)
        new_df['main_category'] = new_df['main_category'].fillna('')
        new_df['sub_category'] = new_df['sub_category'].fillna('')
        new_df['product_name'] = new_df['product_name'].fillna('')

        # Normalize product names
        new_df['product_name_clean'] = new_df['product_name'].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()) if pd.notna(x) else ""
        )

        # Load existing state for lag computation
        state_path = f"{self.model_dir}/price_state.csv"
        if os.path.exists(state_path):
            historical = pd.read_csv(state_path, encoding='utf-8-sig')
            historical['date'] = pd.to_datetime(historical['date'], errors='coerce')
            combined_df = pd.concat([historical, new_df], ignore_index=True)
        else:
            combined_df = new_df.copy()

        combined_df = combined_df.drop_duplicates(subset=['product_name_clean', 'date'])

        # Clean combined data
        combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
        combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')
        combined_df['discount'] = pd.to_numeric(combined_df['discount'], errors='coerce').fillna(0)
        combined_df = combined_df.dropna(subset=['price', 'date'])
        combined_df = combined_df.sort_values(['product_name_clean', 'date']).reset_index(drop=True)

        # Create lag features
        def create_lags(g):
            g = g.sort_values('date').copy()
            for lag in [1, 2, 3]:
                g[f'price_lag_{lag}'] = g['price'].shift(lag)
            g['rolling_mean_2'] = g['price'].shift(1).rolling(2, min_periods=2).mean()
            g['rolling_mean_3'] = g['price'].shift(1).rolling(3, min_periods=3).mean()
            return g

        combined_df = combined_df.groupby('product_name_clean', group_keys=False).apply(create_lags).reset_index(drop=True)

        # Load or create label encoders
        encoder_path = f"{self.model_dir}/label_encoders.pkl"
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            # Transform using existing encoders
            combined_df['main_category_encoded'] = self._safe_transform(
                self.label_encoders['main'], combined_df['main_category'].astype(str)
            )
            combined_df['sub_category_encoded'] = self._safe_transform(
                self.label_encoders['sub'], combined_df['sub_category'].astype(str)
            )
        else:
            self.label_encoders['main'] = LabelEncoder()
            self.label_encoders['sub'] = LabelEncoder()
            combined_df['main_category_encoded'] = self.label_encoders['main'].fit_transform(
                combined_df['main_category'].astype(str)
            )
            combined_df['sub_category_encoded'] = self.label_encoders['sub'].fit_transform(
                combined_df['sub_category'].astype(str)
            )
            # Save encoders
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoders, f)

        # Add time features
        combined_df['week_number'] = combined_df['date'].dt.isocalendar().week.astype(int)
        combined_df['month'] = combined_df['date'].dt.month
        combined_df['day_of_week'] = combined_df['date'].dt.dayofweek

        # Add safe engineered features
        # Reset index to avoid conflicts
        combined_df = combined_df.reset_index(drop=True)

        combined_df['category_avg_price'] = combined_df.groupby('main_category_encoded')['price'].transform('mean')
        combined_df['category_avg_price_week'] = combined_df.groupby(
            ['main_category_encoded', 'week_number']
        )['price'].transform('mean')

        # Fix: use transform with proper groupby
        def calc_change_freq(x):
            return (x.diff() != 0).rolling(3, min_periods=1).sum()

        combined_df['price_change_frequency'] = combined_df.groupby('product_name_clean')['price'].transform(calc_change_freq).fillna(0)

        combined_df['price_momentum'] = combined_df['price_lag_1'] - combined_df['price_lag_2']
        combined_df['price_vs_category'] = combined_df['price_lag_1'] - combined_df['category_avg_price']
        combined_df['trend_indicator'] = combined_df['price_lag_1'] - combined_df['rolling_mean_3']

        # Get only new data rows
        processed_new_data = combined_df[combined_df['date'] == new_df['date'].max()].copy()

        # Update state for next run BEFORE filling (so we save all current data)
        self._update_state(new_df)

        # Fill missing lag features with current price (for first run)
        # This is not ideal but allows the pipeline to run
        required_features = ['price_lag_1', 'price_lag_2', 'price_lag_3',
                           'rolling_mean_2', 'rolling_mean_3']

        for feat in required_features:
            if feat in processed_new_data.columns:
                mask = processed_new_data[feat].isna()
                if mask.any():
                    print(f"    Filling {mask.sum()} missing values for {feat}")
                    # Use current price as fallback
                    processed_new_data.loc[mask, feat] = processed_new_data.loc[mask, 'price']

        print(f"    Processed: {len(processed_new_data)} products")
        return processed_new_data

    def _safe_transform(self, encoder, values):
        """Safely transform labels, handling unseen categories"""
        unique_vals = values.unique()
        for val in unique_vals:
            if val not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, val)
        return encoder.transform(values)

    def _update_state(self, new_data):
        """Update price state file for next inference"""
        state_path = f"{self.model_dir}/price_state.csv"
        state_df = new_data[['product_name_clean', 'date', 'price', 'main_category',
                             'sub_category', 'discount']].copy()
        state_df.to_csv(state_path, index=False, encoding='utf-8-sig')

    def predict(self, processed_data):
        """Generate predictions using all models"""
        print("[4] Generating predictions...")

        predictions = {}
        X = processed_data[self.feature_columns]

        for name, model in self.models.items():
            predictions[name] = model.predict(X)
            print(f"    {name.upper()} predictions generated")

        return predictions

    def ensemble(self, predictions):
        """Apply ensemble averaging"""
        print("[5] Computing ensemble average...")

        pred_df = pd.DataFrame(predictions)
        ensemble_pred = pred_df.mean(axis=1)

        return ensemble_pred.values

    def generate_decision(self, processed_data, ensemble_pred, threshold=None):
        """Generate final business output with trend analysis"""
        print("[6] Generating business decisions...")

        if threshold is None:
            threshold = self.threshold

        result_df = processed_data[['product_name_clean', 'date', 'price']].copy()
        result_df['current_price'] = result_df['price']
        result_df['predicted_price'] = ensemble_pred
        result_df['price_change'] = result_df['predicted_price'] - result_df['current_price']
        result_df['change_percentage'] = (result_df['price_change'] / result_df['current_price']) * 100

        # Apply trend logic
        result_df['trend_label'] = 'STABLE'
        result_df.loc[result_df['predicted_price'] > result_df['current_price'] + threshold, 'trend_label'] = 'UP'
        result_df.loc[result_df['predicted_price'] < result_df['current_price'] - threshold, 'trend_label'] = 'DOWN'

        # Select final columns
        output_df = result_df[[
            'product_name_clean', 'date', 'current_price',
            'predicted_price', 'price_change', 'change_percentage', 'trend_label'
        ]].copy()

        print(f"    Decisions generated for {len(output_df)} products")
        return output_df

    def save_predictions(self, output_df, output_path=None):
        """Save predictions to CSV with date versioning"""
        if output_path is None:
            # Generate date-versioned filename
            from datetime import datetime
            predictions_dir = os.path.join(self.model_dir, "data", "predictions")
            os.makedirs(predictions_dir, exist_ok=True)

            date_str = datetime.now().strftime('%Y-%m-%d')
            output_path = os.path.join(predictions_dir, f"predictions_{date_str}.csv")

        print(f"[7] Saving predictions to {output_path}...")
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"    Saved: {len(output_df)} rows")

        # Also save as latest
        latest_path = os.path.join(self.model_dir, "predictions_latest.csv")
        output_df.to_csv(latest_path, index=False, encoding='utf-8-sig')
        print(f"    Also saved as: {latest_path}")

        return self

    def run_inference(self, new_data_path, output_path="predictions_output.csv"):
        """Run complete inference pipeline"""
        print("="*60)
        print("PRICE FORECASTING - INFERENCE PIPELINE")
        print("="*60 + "\n")

        self.load_models()
        self.load_historical_data()
        processed_data = self.preprocess_data(new_data_path)
        predictions = self.predict(processed_data)
        ensemble_pred = self.ensemble(predictions)
        output_df = self.generate_decision(processed_data, ensemble_pred)
        self.save_predictions(output_df, output_path)

        print("\n" + "="*60)
        print("INFERENCE COMPLETE")
        print("="*60)

        return output_df


def main():
    pipeline = PriceForecastingPipeline()
    output = pipeline.run_inference(
        new_data_path="carrefour_products-2026-04-28.csv",
        output_path="predictions_output.csv"
    )
    print("\nPreview:")
    print(output.head(10))


if __name__ == '__main__':
    main()
