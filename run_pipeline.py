"""
run_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Single entry point for the Prediction Pipeline backend.

Usage:
    python run_pipeline.py                      # Full run: scrape + inference
    python run_pipeline.py --skip-scraping      # Skip scraping, use latest file
    python run_pipeline.py --retrain            # Force model retraining after inference
    python run_pipeline.py --inference-only     # No retraining logic
    python run_pipeline.py --push               # Push data files to GitHub after run
    python run_pipeline.py --skip-scraping --push  # Combine flags

OUTPUTS (canonical files read by Streamlit):
    data/latest_products.csv    ← latest scraped products
    data/predictions.csv        ← latest price predictions
"""

import os
import sys
import argparse
import logging
import shutil
from typing import Optional
from datetime import datetime

# ── Base dir & logging ────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(log_dir, f"pipeline_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(base_dir, "data")
SCRAPED_DIR    = os.path.join(DATA_DIR, "scraped")
PRED_DIR       = os.path.join(DATA_DIR, "predictions")
TRAINING_DIR   = os.path.join(DATA_DIR, "training")
MODELS_DIR     = os.path.join(base_dir, "models")
MONTH_STATE    = os.path.join(DATA_DIR, "last_retrain_month.txt")

# ── Canonical output paths (read by Streamlit) ────────────────────────────────
CANONICAL_PRODUCTS    = os.path.join(DATA_DIR, "latest_products.csv")
CANONICAL_PREDICTIONS = os.path.join(DATA_DIR, "predictions.csv")

for _d in [DATA_DIR, SCRAPED_DIR, PRED_DIR, TRAINING_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Scraping
# ─────────────────────────────────────────────────────────────────────────────

def run_scraping() -> Optional[str]:
    """Run the Playwright scraper and return the output CSV path."""
    logger.info("=" * 60)
    logger.info("STEP 1 — SCRAPING")
    logger.info("=" * 60)
    try:
        import scraper
        scraper.main()
        output_path = scraper.OUTPUT_FILE

        if output_path and os.path.exists(output_path):
            import pandas as pd
            try:
                df = pd.read_csv(output_path)
                if len(df) == 0:
                    logger.error("Scraped file is empty (no products). Aborting update to avoid wiping latest_products.csv.")
                    return None
            except Exception as e:
                logger.error(f"Failed to read scraped file: {e}")
                return None

            # Copy to canonical latest_products.csv only if it has data
            shutil.copy2(output_path, CANONICAL_PRODUCTS)
            logger.info(f"Scraping done: {output_path} ({len(df)} products)")
            logger.info(f"Canonical products updated: {CANONICAL_PRODUCTS}")
            return output_path
        else:
            logger.error("Scraper ran but output file not found.")
            return None

    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(input_path: str, output_path: Optional[str] = None) -> bool:
    """Run price-forecasting inference and save predictions."""
    logger.info("=" * 60)
    logger.info("STEP 2 — INFERENCE")
    logger.info("=" * 60)
    try:
        from inference_pipeline import PriceForecastingPipeline

        # Use models/ directory for all model files
        pipeline = PriceForecastingPipeline(model_dir=MODELS_DIR)

        pipeline.load_models()
        processed = pipeline.preprocess_data(input_path)
        if len(processed) == 0:
            logger.error("No valid data for inference.")
            return False

        predictions    = pipeline.predict(processed)
        ensemble_pred  = pipeline.ensemble(predictions)
        output_df      = pipeline.generate_decision(processed, ensemble_pred)

        # Date-versioned save
        date_str  = datetime.now().strftime("%Y-%m-%d")
        versioned = os.path.join(PRED_DIR, f"predictions_{date_str}.csv")
        output_df.to_csv(versioned, index=False, encoding="utf-8-sig")
        logger.info(f"Versioned predictions: {versioned}")

        # Canonical predictions (overwrite latest)
        output_df.to_csv(CANONICAL_PREDICTIONS, index=False, encoding="utf-8-sig")
        logger.info(f"Canonical predictions updated: {CANONICAL_PREDICTIONS}")

        return True

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Monthly Retraining
# ─────────────────────────────────────────────────────────────────────────────

def _read_last_retrain_month() -> Optional[str]:
    if os.path.exists(MONTH_STATE):
        try:
            return open(MONTH_STATE, "r", encoding="utf-8").read().strip() or None
        except Exception:
            return None
    return None


def _write_last_retrain_month(month_str: str):
    with open(MONTH_STATE, "w", encoding="utf-8") as f:
        f.write(month_str)


def check_and_run_monthly_retrain():
    """After every successful scrape, check if monthly retraining is due."""
    now               = datetime.now()
    current_month_str = now.strftime("%Y-%m")
    last_month_str    = _read_last_retrain_month()

    logger.info("=" * 60)
    logger.info("MONTHLY RETRAIN CHECK")
    logger.info(f"  Current: {current_month_str} | Last: {last_month_str or 'Never'}")
    logger.info("=" * 60)

    if last_month_str is None:
        logger.info("  First run — recording month, no retrain yet.")
        _write_last_retrain_month(current_month_str)
        return

    if current_month_str == last_month_str:
        logger.info("  Same month — no retrain needed.")
        return

    # Month changed — retrain on previous month's data
    import glob, re as _re
    prev_year  = int(last_month_str[:4])
    prev_month = int(last_month_str[5:7])
    prefix     = f"{prev_year:04d}-{prev_month:02d}-"

    month_files = sorted([
        f for f in glob.glob(os.path.join(SCRAPED_DIR, "carrefour_products_*.csv"))
        if _re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f)) and
           _re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f)).group(1).startswith(prefix)
    ])

    if not month_files:
        logger.warning(f"  No scraped files for {last_month_str}. Skipping retrain.")
        _write_last_retrain_month(current_month_str)
        return

    logger.info(f"  Found {len(month_files)} files for {last_month_str} — merging...")
    import pandas as pd
    dfs = []
    for f in month_files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
        except Exception as e:
            logger.warning(f"  Skipping {f}: {e}")
    if not dfs:
        _write_last_retrain_month(current_month_str)
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged_path = os.path.join(TRAINING_DIR, f"monthly_merge_{last_month_str}.csv")
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    logger.info(f"  Merged {len(merged):,} rows → {merged_path}")

    run_retraining(merged_path)
    _write_last_retrain_month(current_month_str)


def run_retraining(input_file: str) -> bool:
    """Run model retraining on the given dataset."""
    logger.info("=" * 60)
    logger.info("RETRAINING")
    logger.info("=" * 60)
    try:
        from retrain_models import ModelRetrainer
        retrainer = ModelRetrainer(base_dir)
        models = retrainer.retrain(latest_data_path=input_file)
        if models is None:
            logger.warning("Retraining skipped (insufficient data).")
            return False
        logger.info("Retraining completed.")
        return True
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Push to GitHub
# ─────────────────────────────────────────────────────────────────────────────

def push_to_github():
    """Commit and push canonical data files to GitHub."""
    logger.info("=" * 60)
    logger.info("STEP 4 — PUSH TO GITHUB")
    logger.info("=" * 60)
    try:
        from push_to_github import push_data_files
        push_data_files(base_dir)
    except Exception as e:
        logger.error(f"GitHub push failed: {e}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prediction Pipeline — Single Entry Point")
    parser.add_argument("--input",           "-i", type=str,  help="Input CSV (scraped data)")
    parser.add_argument("--skip-scraping",   action="store_true", help="Skip scraping step")
    parser.add_argument("--retrain",         "-r", action="store_true", help="Force retraining after inference")
    parser.add_argument("--inference-only",  action="store_true", help="Skip all retraining logic")
    parser.add_argument("--push",            action="store_true", help="Push data files to GitHub after run")
    parser.add_argument("--config",          default="config.yaml", help="Config file path")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE — STARTED")
    logger.info(f"  skip_scraping  : {args.skip_scraping}")
    logger.info(f"  retrain        : {args.retrain}")
    logger.info(f"  inference_only : {args.inference_only}")
    logger.info(f"  push           : {args.push}")
    logger.info("=" * 60)

    # ── Step 1: Scraping ──────────────────────────────────────────────────────
    input_file = args.input

    if not args.skip_scraping and input_file is None:
        scraped_output = run_scraping()
        if scraped_output:
            input_file = scraped_output
            if not args.inference_only:
                check_and_run_monthly_retrain()
        else:
            logger.error("Scraping failed. Attempting to use latest existing file.")

    if not input_file or not os.path.exists(str(input_file)):
        # Find latest scraped file as fallback
        import glob
        files = sorted(
            glob.glob(os.path.join(SCRAPED_DIR, "carrefour_products_*.csv")),
            key=os.path.getmtime
        )
        if files:
            input_file = files[-1]
            logger.info(f"Using latest scraped file: {input_file}")
        elif os.path.exists(CANONICAL_PRODUCTS):
            input_file = CANONICAL_PRODUCTS
            logger.info(f"Using canonical products file: {input_file}")
        else:
            logger.error("No input file found. Cannot run inference.")
            sys.exit(1)

    # ── Step 2: Inference ─────────────────────────────────────────────────────
    success = run_inference(input_file)
    if not success:
        logger.error("Inference failed.")
        sys.exit(1)

    # ── Step 3: Manual retrain flag ────────────────────────────────────────────
    if args.retrain:
        run_retraining(input_file)

    # ── Step 4: Push to GitHub ─────────────────────────────────────────────────
    if args.push:
        push_to_github()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"  Products  : {CANONICAL_PRODUCTS}")
    logger.info(f"  Predictions: {CANONICAL_PREDICTIONS}")
    logger.info("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
