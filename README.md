# SENTINEL-National-Disease-Outbreak-Detection-Response-System

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data files in the same folder as the scripts
```
your_project_folder/
  ├── IHME-GBD_2023_DATA-dde85905-1.csv   ← IHME Kenya dataset
  ├── Training_extended.csv                ← Extended symptom training data
  ├── Testing_extended.csv                 ← Extended symptom testing data
  ├── requirements.txt
  ├── step1_data_cleaning.py
  ├── step2_random_forest.py
  ├── step3_prophet_forecasting.py
  ├── step4_alert_engine.py
  └── step5_export_dashboard_data.py
```

### 3. Run scripts in order
```bash
python step1_data_cleaning.py       # ~10 seconds
python step2_random_forest.py       # ~30 seconds
python step3_prophet_forecasting.py # ~10–15 minutes
python step4_alert_engine.py        # ~1 minute
python step5_export_dashboard_data.py # ~10 seconds
```

## What each step does

| Step | Script | Output |
|------|--------|--------|
| 1 | step1_data_cleaning.py | Cleaned CSV files in clean_data/ |
| 2 | step2_random_forest.py | Trained RF model + accuracy report |
| 3 | step3_prophet_forecasting.py | Forecasts + anomaly detection results |
| 4 | step4_alert_engine.py | Simulated 2024 clinic data + alerts |
| 5 | step5_export_dashboard_data.py | JSON files for React dashboard |

## Output folders created automatically
```
clean_data/     ← cleaned datasets + ML results
models/         ← saved ML models
outputs/        ← charts and plots
dashboard_data/ ← JSON files for React dashboard
```
