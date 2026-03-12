# ============================================================
# SENTINEL — Disease Outbreak Detection System
# Step 3: Prophet Forecasting + Anomaly Detection
# ============================================================
# For each disease + county combination:
#   1. Train Prophet on 1980–2018 data
#   2. Predict 2019–2023
#   3. Compare predictions vs actual
#   4. Flag anomalies where actual >> predicted
#
# HOW TO RUN:
#   pip install pandas prophet matplotlib joblib
#   python step3_prophet_forecasting.py
#
# MUST run step1 and step2 first
# Note: Prophet may take 5–15 minutes to run for all counties
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "clean_data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load clean IHME data ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: PROPHET FORECASTING + ANOMALY DETECTION")
print("=" * 60)

ihme_train = pd.read_csv(f"{DATA_DIR}/ihme_train.csv")
ihme_test  = pd.read_csv(f"{DATA_DIR}/ihme_test.csv")
ihme_all   = pd.read_csv(f"{DATA_DIR}/ihme_clean.csv")

diseases = sorted(ihme_all['disease'].unique())
counties = sorted(ihme_all['county'].unique())

print(f"Diseases: {diseases}")
print(f"Counties: {len(counties)}")
print(f"Training years: {ihme_train['year'].min()} - {ihme_train['year'].max()}")
print(f"Testing years:  {ihme_test['year'].min()} - {ihme_test['year'].max()}")

# ── Prophet helper functions ──────────────────────────────────────────────────

def train_prophet(df_train):
    """
    Train a Prophet model on yearly case count data.
    Prophet requires columns: ds (date) and y (value)
    """
    # Prophet needs a datetime column called 'ds'
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df_train['year'].astype(str) + '-01-01'),
        'y':  df_train['cases'].values
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,        # 95% confidence interval
        changepoint_prior_scale=0.1 # flexibility of trend changes
    )
    model.fit(prophet_df)
    return model


def predict_prophet(model, years):
    """
    Generate predictions for a list of years.
    Returns dataframe with predicted cases + confidence bounds.
    """
    future = pd.DataFrame({
        'ds': pd.to_datetime([f"{y}-01-01" for y in years])
    })
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def detect_anomaly(actual, predicted, upper_bound):
    """
    Flag a year as anomalous if:
    - Actual cases exceed the upper confidence bound (95%)
    - Z-score > 2
    """
    # Method 1: Exceeds upper bound
    exceeds_bound = actual > upper_bound

    # Method 2: Z-score
    z_score = (actual - predicted) / (predicted + 1e-9)

    is_anomaly = exceeds_bound or (z_score > 0.5)

    return {
        'is_anomaly': bool(is_anomaly),
        'z_score':    round(float(z_score), 3),
        'pct_above':  round(float((actual - predicted) / (predicted + 1e-9) * 100), 1)
    }


def risk_level(pct_above):
    """Convert percentage above predicted to risk level."""
    if pct_above >= 200:  return 'CRITICAL'
    if pct_above >= 100:  return 'HIGH'
    if pct_above >= 50:   return 'WARNING'
    if pct_above >= 20:   return 'WATCH'
    return 'NORMAL'

# ── Run Prophet for all diseases and counties ─────────────────────────────────
print("\nRunning Prophet forecasting...")
print("(This may take several minutes for all county-disease combinations)")
print()

results    = []   # stores test set predictions + anomaly flags
all_models = {}   # stores trained models

test_years  = sorted(ihme_test['year'].unique())
train_years = sorted(ihme_train['year'].unique())

total  = len(diseases) * len(counties)
done   = 0

for disease in diseases:
    all_models[disease] = {}

    for county in counties:
        done += 1
        if done % 50 == 0:
            print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)")

        # Get training data for this disease + county
        train_subset = ihme_train[
            (ihme_train['disease'] == disease) &
            (ihme_train['county']  == county)
        ].sort_values('year')

        # Need at least 5 years of data to train Prophet
        if len(train_subset) < 5:
            continue

        # Train Prophet
        try:
            model = train_prophet(train_subset)
            all_models[disease][county] = model
        except Exception as e:
            print(f"  ⚠️  Prophet failed for {disease}/{county}: {e}")
            continue

        # Predict test years (2019-2023)
        test_subset = ihme_test[
            (ihme_test['disease'] == disease) &
            (ihme_test['county']  == county)
        ].sort_values('year')

        if len(test_subset) == 0:
            continue

        forecast = predict_prophet(model, test_subset['year'].values)

        # Compare predictions vs actual
        for _, actual_row in test_subset.iterrows():
            year = actual_row['year']
            actual_cases = actual_row['cases']

            # Find matching forecast row
            fc_row = forecast[forecast['ds'].dt.year == year]
            if fc_row.empty:
                continue

            predicted    = max(0, fc_row['yhat'].values[0])
            pred_lower   = max(0, fc_row['yhat_lower'].values[0])
            pred_upper   = max(0, fc_row['yhat_upper'].values[0])

            anomaly_info = detect_anomaly(actual_cases, predicted, pred_upper)

            results.append({
                'county':          county,
                'disease':         disease,
                'year':            year,
                'actual_cases':    actual_cases,
                'predicted_cases': round(predicted),
                'pred_lower':      round(pred_lower),
                'pred_upper':      round(pred_upper),
                'is_anomaly':      anomaly_info['is_anomaly'],
                'z_score':         anomaly_info['z_score'],
                'pct_above':       anomaly_info['pct_above'],
                'risk_level':      risk_level(anomaly_info['pct_above'])
            })

# ── Convert results to dataframe ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
print(f"\n✅ Forecasting complete!")
print(f"Total predictions: {len(results_df)}")

# ── Model Performance Metrics ─────────────────────────────────────────────────
print("\n" + "─" * 60)
print("PROPHET MODEL PERFORMANCE (2019-2023 test set)")
print("─" * 60)

from sklearn.metrics import mean_absolute_error, mean_squared_error

for disease in diseases:
    d_results = results_df[results_df['disease'] == disease]
    if len(d_results) == 0:
        continue

    actual    = d_results['actual_cases'].values
    predicted = d_results['predicted_cases'].values

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100

    anomaly_count = d_results['is_anomaly'].sum()

    print(f"\n{disease}:")
    print(f"  MAE:      {mae:,.0f} cases")
    print(f"  RMSE:     {rmse:,.0f} cases")
    print(f"  MAPE:     {mape:.1f}%")
    print(f"  Anomalies detected: {anomaly_count} county-years flagged")

# ── Anomaly Summary ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("ANOMALY DETECTION SUMMARY")
print("─" * 60)

anomalies = results_df[results_df['is_anomaly'] == True].copy()
print(f"\nTotal anomalies detected: {len(anomalies)}")

by_risk = anomalies['risk_level'].value_counts()
print(f"\nBy risk level:")
for level in ['CRITICAL', 'HIGH', 'WARNING', 'WATCH']:
    count = by_risk.get(level, 0)
    print(f"  {level:<10}: {count}")

print(f"\nTop 10 worst anomalies:")
top10 = anomalies.nlargest(10, 'pct_above')[
    ['county', 'disease', 'year', 'actual_cases', 'predicted_cases', 'pct_above', 'risk_level']
]
print(top10.to_string(index=False))

# ── Plot: Forecast vs Actual for top diseases ─────────────────────────────────
print("\nGenerating forecast plots...")

# Plot for Nairobi county (most populous)
plot_county = 'Nairobi'

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, disease in enumerate(diseases):
    if i >= len(axes):
        break

    ax = axes[i]

    # All data (train + test)
    all_subset = ihme_all[
        (ihme_all['disease'] == disease) &
        (ihme_all['county']  == plot_county)
    ].sort_values('year')

    if len(all_subset) == 0:
        ax.set_title(f"{disease}\n(No data for {plot_county})")
        continue

    # Get predictions for this disease + county
    preds = results_df[
        (results_df['disease'] == disease) &
        (results_df['county']  == plot_county)
    ].sort_values('year')

    # Plot actual cases
    ax.plot(all_subset['year'], all_subset['cases'],
            color='steelblue', linewidth=2, label='Actual cases', marker='o', markersize=3)

    # Plot train/test divider
    ax.axvline(x=2018, color='gray', linestyle='--', alpha=0.7, label='Train/Test split')

    # Plot predictions
    if len(preds) > 0:
        ax.plot(preds['year'], preds['predicted_cases'],
                color='orange', linewidth=2, linestyle='--', label='Predicted')
        ax.fill_between(preds['year'], preds['pred_lower'], preds['pred_upper'],
                        alpha=0.2, color='orange', label='95% CI')

        # Highlight anomalies
        anom = preds[preds['is_anomaly'] == True]
        if len(anom) > 0:
            ax.scatter(anom['year'], anom['actual_cases'],
                      color='red', s=80, zorder=5, label='Anomaly')

    ax.set_title(f"{disease}", fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Cases', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f'SENTINEL — Disease Forecasting: {plot_county} County\nProphet Model (trained 1980–2018, tested 2019–2023)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/prophet_forecasts_{plot_county}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"📊 Forecast plots saved to {OUTPUT_DIR}/prophet_forecasts_{plot_county}.png")

# ── Save results ──────────────────────────────────────────────────────────────
results_df.to_csv(f"{DATA_DIR}/prophet_results.csv", index=False)
anomalies.to_csv(f"{DATA_DIR}/anomalies.csv", index=False)
joblib.dump(all_models, f"{MODEL_DIR}/prophet_models.pkl")

print(f"\n✅ Results saved to {DATA_DIR}/prophet_results.csv")
print(f"✅ Anomalies saved to {DATA_DIR}/anomalies.csv")
print(f"✅ Models saved to {MODEL_DIR}/prophet_models.pkl")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 COMPLETE ✅")
print("=" * 60)
print(f"""
Prophet trained for {len(diseases)} diseases × {len(counties)} counties
Total anomalies detected: {len(anomalies)}
Critical alerts:          {len(anomalies[anomalies['risk_level'] == 'CRITICAL'])}

Next step: Run step4_alert_engine.py
""")
