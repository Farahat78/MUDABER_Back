# 🔵 Prediction Pipeline — Backend

Fully independent ML + scraping pipeline for the Smart Salary System.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Run the full pipeline
python run_pipeline.py

# Skip scraping (use last scraped file)
python run_pipeline.py --skip-scraping

# Full run + push data to GitHub
python run_pipeline.py --push

# Force model retraining
python run_pipeline.py --retrain
```

## 📤 Outputs (Data Contract)

| File | Description | Read by |
|------|-------------|---------|
| `data/latest_products.csv` | Latest scraped Carrefour products | Shopping page |
| `data/predictions.csv` | Price trend predictions (XGB + LGB + CatBoost ensemble) | Shopping page |

These two files are the **only connection** between this pipeline and the Streamlit frontend.
After each run, push them to GitHub: the Streamlit Cloud app reads from GitHub raw URLs.

## 🏗️ Architecture

```
run_pipeline.py           ← Single entry point
scraper.py                ← Playwright scraper (Carrefour Egypt)
inference_pipeline.py     ← XGB/LGB/CatBoost ensemble inference
retrain_models.py         ← Monthly model retraining
push_to_github.py         ← Git commit + push helper
config.yaml               ← Pipeline configuration
models/
  xgb_model.pkl
  lgb_model.pkl
  cat_model.pkl
  label_encoders.pkl
data/
  latest_products.csv     ← ✅ Committed to GitHub
  predictions.csv         ← ✅ Committed to GitHub
  scraped/                ← Raw daily scrapes (not committed)
  training/               ← Accumulated training data (not committed)
  predictions/            ← Date-versioned predictions (not committed)
```

## 🔄 Scheduling (Windows Task Scheduler)

Run weekly (e.g., every Friday at 02:00):
```
Action: Start a program
Program: python
Arguments: C:\path\to\prediction-pipeline\run_pipeline.py --push
Start in: C:\path\to\prediction-pipeline\
```

## 🚨 Rules

- ❌ No Streamlit imports here
- ❌ No API calls to the frontend
- ✅ Only writes to `data/latest_products.csv` and `data/predictions.csv`
- ✅ Pushes to GitHub after each successful run (with `--push`)
# MUDABER_Back
