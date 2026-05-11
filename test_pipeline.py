"""
test_pipeline.py
─────────────────────────────────────────
Full system test for the prediction pipeline.
Tests:
  1. Imports & dependencies
  2. Model loading
  3. preprocess_data() with real data
  4. predict() + ensemble()
  5. generate_decision()
  6. Full end-to-end run via run_inference()
  7. Edge cases (missing cols, NaN dates, empty state)
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
SCRAPED_DIR = os.path.join(DATA_DIR, "scraped")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(name, fn):
    try:
        msg = fn()
        status = PASS
        detail = msg or ""
    except AssertionError as e:
        status = FAIL
        detail = str(e)
    except Exception as e:
        status = FAIL
        detail = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  →  {detail}" if detail else ""))


# ─── find a real scraped CSV to use ────────────────────────────────────────────
def find_input_csv():
    # try scraped dir first
    import glob
    files = sorted(glob.glob(os.path.join(SCRAPED_DIR, "*.csv")), key=os.path.getmtime)
    if files:
        return files[-1]
    # fallback to data/latest_products.csv
    lp = os.path.join(DATA_DIR, "latest_products.csv")
    if os.path.exists(lp):
        return lp
    return None

INPUT_CSV = find_input_csv()

print("\n" + "="*60)
print("  PREDICTION PIPELINE — SYSTEM TEST")
print("="*60)
print(f"  Models dir : {MODELS_DIR}")
print(f"  Input CSV  : {INPUT_CSV}")
print("="*60 + "\n")

# ── TEST 1: Imports ─────────────────────────────────────────────────────────────
print("[1] Testing imports...")

def test_imports():
    import pandas, numpy, pickle, sklearn, re
    return f"pandas={pandas.__version__}  numpy={numpy.__version__}"

check("Core imports (pandas, numpy, sklearn)", test_imports)

def test_inference_import():
    from inference_pipeline import PriceForecastingPipeline
    return "PriceForecastingPipeline imported"

check("Import inference_pipeline.PriceForecastingPipeline", test_inference_import)


# ── TEST 2: Model Loading ────────────────────────────────────────────────────────
print("\n[2] Testing model loading...")

def test_model_files():
    missing = [f for f in ["xgb_model.pkl","lgb_model.pkl","cat_model.pkl"]
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    assert not missing, f"Missing model files: {missing}"
    return "All 3 model files present"

check("Model files exist in models/", test_model_files)

def test_load_models():
    from inference_pipeline import PriceForecastingPipeline
    p = PriceForecastingPipeline(model_dir=MODELS_DIR)
    p.load_models()
    assert len(p.models) == 3, f"Expected 3 models, got {len(p.models)}"
    return f"Loaded: {list(p.models.keys())}"

check("load_models() loads xgb, lgb, cat", test_load_models)


# ── TEST 3: Preprocessing ────────────────────────────────────────────────────────
print("\n[3] Testing preprocess_data()...")

def test_input_csv_exists():
    assert INPUT_CSV and os.path.exists(INPUT_CSV), \
        "No scraped CSV found. Run scraper first."
    df = pd.read_csv(INPUT_CSV)
    return f"{len(df)} rows, cols: {list(df.columns)}"

check("Input CSV readable", test_input_csv_exists)

pipeline_instance = None

def test_preprocess():
    global pipeline_instance
    from inference_pipeline import PriceForecastingPipeline
    p = PriceForecastingPipeline(model_dir=MODELS_DIR)
    p.load_models()
    df = p.preprocess_data(INPUT_CSV)
    assert len(df) > 0, "preprocess_data returned empty DataFrame"
    assert 'product_name_clean' in df.columns, "Missing product_name_clean"
    assert 'date' in df.columns, "Missing date"
    assert 'price' in df.columns, "Missing price"
    # check all feature columns exist
    missing_feats = [c for c in p.feature_columns if c not in df.columns]
    assert not missing_feats, f"Missing feature columns: {missing_feats}"
    pipeline_instance = p
    return f"{len(df)} rows processed, all {len(p.feature_columns)} features present"

check("preprocess_data() returns valid DataFrame", test_preprocess)

def test_no_nan_features():
    global pipeline_instance
    from inference_pipeline import PriceForecastingPipeline
    p = PriceForecastingPipeline(model_dir=MODELS_DIR)
    p.load_models()
    df = p.preprocess_data(INPUT_CSV)
    feat_df = df[p.feature_columns]
    nan_counts = feat_df.isna().sum()
    total_nan = nan_counts.sum()
    assert total_nan == 0, f"NaN values in features: {nan_counts[nan_counts>0].to_dict()}"
    return "No NaN in any feature column"

check("No NaN values in feature columns", test_no_nan_features)

def test_product_name_clean_no_nan():
    from inference_pipeline import PriceForecastingPipeline
    p = PriceForecastingPipeline(model_dir=MODELS_DIR)
    p.load_models()
    df = p.preprocess_data(INPUT_CSV)
    nan_count = df['product_name_clean'].isna().sum()
    assert nan_count == 0, f"{nan_count} NaN in product_name_clean"
    empty_count = (df['product_name_clean'].str.strip() == '').sum()
    return f"product_name_clean: {len(df)} rows, {nan_count} NaN, {empty_count} empty"

check("product_name_clean has no NaN", test_product_name_clean_no_nan)


# ── TEST 4: Predict + Ensemble ───────────────────────────────────────────────────
print("\n[4] Testing predict() and ensemble()...")

processed_df = None

def test_predict():
    global processed_df
    from inference_pipeline import PriceForecastingPipeline
    p = PriceForecastingPipeline(model_dir=MODELS_DIR)
    p.load_models()
    df = p.preprocess_data(INPUT_CSV)
    preds = p.predict(df)
    assert len(preds) == 3, f"Expected 3 model preds, got {len(preds)}"
    for name, arr in preds.items():
        assert len(arr) == len(df), f"{name}: pred length mismatch"
        assert not np.isnan(arr).any(), f"{name}: NaN in predictions"
    processed_df = df
    return f"3 models predicted {len(df)} rows each"

check("predict() returns valid arrays for all 3 models", test_predict)

def test_ensemble():
    from inference_pipeline import PriceForecastingPipeline
    p = PriceForecastingPipeline(model_dir=MODELS_DIR)
    p.load_models()
    df = p.preprocess_data(INPUT_CSV)
    preds = p.predict(df)
    ensemble = p.ensemble(preds)
    assert len(ensemble) == len(df), "Ensemble length mismatch"
    assert not np.isnan(ensemble).any(), "NaN in ensemble predictions"
    return f"Ensemble: {len(ensemble)} predictions, mean={ensemble.mean():.2f}"

check("ensemble() produces valid averaged predictions", test_ensemble)


# ── TEST 5: generate_decision ────────────────────────────────────────────────────
print("\n[5] Testing generate_decision()...")

def test_generate_decision():
    from inference_pipeline import PriceForecastingPipeline
    p = PriceForecastingPipeline(model_dir=MODELS_DIR)
    p.load_models()
    df = p.preprocess_data(INPUT_CSV)
    preds = p.predict(df)
    ensemble = p.ensemble(preds)
    result = p.generate_decision(df, ensemble)
    required_cols = ['product_name_clean','date','current_price',
                     'predicted_price','price_change','change_percentage','trend_label']
    missing = [c for c in required_cols if c not in result.columns]
    assert not missing, f"Missing output columns: {missing}"
    assert len(result) > 0, "Empty decision output"
    labels = set(result['trend_label'].unique())
    assert labels.issubset({'UP','DOWN','STABLE'}), f"Unexpected labels: {labels}"
    return (f"{len(result)} rows | "
            f"UP={( result['trend_label']=='UP').sum()} "
            f"DOWN={(result['trend_label']=='DOWN').sum()} "
            f"STABLE={(result['trend_label']=='STABLE').sum()}")

check("generate_decision() returns correct columns and labels", test_generate_decision)


# ── TEST 6: Edge Cases ───────────────────────────────────────────────────────────
print("\n[6] Testing edge cases...")

def test_no_state_file():
    """Test that pipeline works even without price_state.csv"""
    import tempfile, shutil
    from inference_pipeline import PriceForecastingPipeline
    # create temp model dir without price_state.csv
    tmp = tempfile.mkdtemp()
    try:
        for f in ["xgb_model.pkl","lgb_model.pkl","cat_model.pkl","label_encoders.pkl"]:
            src = os.path.join(MODELS_DIR, f)
            if os.path.exists(src):
                shutil.copy2(src, tmp)
        p = PriceForecastingPipeline(model_dir=tmp)
        p.load_models()
        df = p.preprocess_data(INPUT_CSV)
        assert len(df) > 0, "Empty result with no state file"
        return f"{len(df)} rows processed (no state file)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

check("preprocess_data() works WITHOUT price_state.csv", test_no_state_file)

def test_nan_date_rows_ignored():
    """NaN date rows should be silently dropped"""
    import tempfile, shutil
    from inference_pipeline import PriceForecastingPipeline
    tmp_csv = os.path.join(DATA_DIR, "_test_nan_dates.csv")
    try:
        df = pd.read_csv(INPUT_CSV)
        # inject some NaN dates
        df.loc[df.index[:5], 'date'] = None
        df.to_csv(tmp_csv, index=False)
        tmp_models = tempfile.mkdtemp()
        for f in ["xgb_model.pkl","lgb_model.pkl","cat_model.pkl","label_encoders.pkl"]:
            src = os.path.join(MODELS_DIR, f)
            if os.path.exists(src): shutil.copy2(src, tmp_models)
        p = PriceForecastingPipeline(model_dir=tmp_models)
        p.load_models()
        result = p.preprocess_data(tmp_csv)
        assert len(result) > 0, "All rows dropped — even valid ones"
        return f"{len(result)} rows survive after dropping NaN-date rows"
    finally:
        if os.path.exists(tmp_csv): os.remove(tmp_csv)
        shutil.rmtree(tmp_models, ignore_errors=True)

check("NaN date rows are silently dropped", test_nan_date_rows_ignored)


# ── TEST 7: Full end-to-end ──────────────────────────────────────────────────────
print("\n[7] Full end-to-end via run_inference()...")

def test_full_end_to_end():
    from inference_pipeline import PriceForecastingPipeline
    import tempfile, shutil
    tmp_models = tempfile.mkdtemp()
    try:
        for f in ["xgb_model.pkl","lgb_model.pkl","cat_model.pkl","label_encoders.pkl"]:
            src = os.path.join(MODELS_DIR, f)
            if os.path.exists(src): shutil.copy2(src, tmp_models)
        # copy price_state too
        state = os.path.join(MODELS_DIR, "price_state.csv")
        if os.path.exists(state): shutil.copy2(state, tmp_models)
        p = PriceForecastingPipeline(model_dir=tmp_models)
        result = p.run_inference(INPUT_CSV)
        assert len(result) > 0, "run_inference returned empty DataFrame"
        assert 'trend_label' in result.columns
        return f"run_inference() → {len(result)} predictions"
    finally:
        shutil.rmtree(tmp_models, ignore_errors=True)

check("run_inference() completes without error", test_full_end_to_end)


# ── SUMMARY ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TEST SUMMARY")
print("="*60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
for status, name, detail in results:
    print(f"  {status}  {name}")

print(f"\n  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
print("="*60)

if failed > 0:
    print("\n  ❌ SYSTEM TEST FAILED — DO NOT PUSH")
    sys.exit(1)
else:
    print("\n  ✅ ALL TESTS PASSED — SAFE TO PUSH")
    sys.exit(0)
