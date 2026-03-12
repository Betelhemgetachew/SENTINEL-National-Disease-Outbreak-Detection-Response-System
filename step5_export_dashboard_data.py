# ============================================================
# SENTINEL — Disease Outbreak Detection System
# Step 5: Export Dashboard Data
# ============================================================
# Exports all ML results as JSON files for the React dashboard.
#
# HOW TO RUN:
#   python step5_export_dashboard_data.py
#
# MUST run steps 1–4 first
# ============================================================

import pandas as pd
import numpy as np
import json
import os

DATA_DIR   = "clean_data"
OUTPUT_DIR = "outputs"
DASH_DIR   = "dashboard_data"
os.makedirs(DASH_DIR, exist_ok=True)

print("=" * 60)
print("STEP 5: EXPORTING DASHBOARD DATA")
print("=" * 60)

# ── Load all results ──────────────────────────────────────────────────────────
ihme          = pd.read_csv(f"{DATA_DIR}/ihme_clean.csv")
prophet_res   = pd.read_csv(f"{DATA_DIR}/prophet_results.csv")
anomalies     = pd.read_csv(f"{DATA_DIR}/anomalies.csv")
alerts        = pd.read_csv(f"{DATA_DIR}/alerts_2024.csv")
monthly       = pd.read_csv(f"{DATA_DIR}/monthly_counts_2024.csv")

# ── 1. National summary stats ─────────────────────────────────────────────────
summary = {
    'total_alerts':    int(len(alerts)),
    'critical_alerts': int(len(alerts[alerts['risk_level'] == 'CRITICAL'])),
    'high_alerts':     int(len(alerts[alerts['risk_level'] == 'HIGH'])),
    'counties_affected': int(alerts['county'].nunique()),
    'diseases_tracked':  int(ihme['disease'].nunique()),
    'last_updated':    '2024-12-01'
}

with open(f"{DASH_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print("✅ summary.json")

# ── 2. Disease trends (for time series charts) ────────────────────────────────
trends = []
for disease in ihme['disease'].unique():
    for county in ihme['county'].unique():
        subset = ihme[
            (ihme['disease'] == disease) &
            (ihme['county']  == county)
        ].sort_values('year')

        preds = prophet_res[
            (prophet_res['disease'] == disease) &
            (prophet_res['county']  == county)
        ].sort_values('year')

        if len(subset) == 0:
            continue

        series = []
        for _, row in subset.iterrows():
            pred_row = preds[preds['year'] == row['year']]
            point = {
                'year':   int(row['year']),
                'actual': int(row['cases']),
            }
            if len(pred_row) > 0:
                point['predicted']  = int(pred_row['predicted_cases'].values[0])
                point['pred_upper'] = int(pred_row['pred_upper'].values[0])
                point['pred_lower'] = int(pred_row['pred_lower'].values[0])
                point['is_anomaly'] = bool(pred_row['is_anomaly'].values[0])
                point['risk_level'] = str(pred_row['risk_level'].values[0])

            series.append(point)

        trends.append({
            'disease': disease,
            'county':  county,
            'series':  series
        })

with open(f"{DASH_DIR}/trends.json", 'w') as f:
    json.dump(trends, f, indent=2)
print(f"✅ trends.json ({len(trends)} disease-county combinations)")

# ── 3. County risk map data ───────────────────────────────────────────────────
county_risk = []
for county in alerts['county'].unique():
    c_alerts = alerts[alerts['county'] == county]
    worst = c_alerts.sort_values(
        'risk_level',
        key=lambda x: x.map({'CRITICAL':0,'HIGH':1,'WARNING':2,'WATCH':3}).fillna(4)
    ).iloc[0]

    county_risk.append({
        'county':       county,
        'risk_level':   worst['risk_level'],
        'top_disease':  worst['disease'],
        'total_alerts': int(len(c_alerts)),
        'max_pct_above': float(c_alerts['pct_above'].max())
    })

with open(f"{DASH_DIR}/county_risk.json", 'w') as f:
    json.dump(county_risk, f, indent=2)
print(f"✅ county_risk.json ({len(county_risk)} counties)")

# ── 4. Alerts feed ────────────────────────────────────────────────────────────
alerts_out = alerts[[
    'timestamp','month_name','county','disease',
    'actual_cases','baseline','pct_above','risk_level','alert_message'
]].to_dict(orient='records')

with open(f"{DASH_DIR}/alerts.json", 'w') as f:
    json.dump(alerts_out, f, indent=2)
print(f"✅ alerts.json ({len(alerts_out)} alerts)")

# ── 5. Monthly case counts ────────────────────────────────────────────────────
monthly_out = monthly.to_dict(orient='records')
with open(f"{DASH_DIR}/monthly_cases.json", 'w') as f:
    json.dump(monthly_out, f, indent=2)
print(f"✅ monthly_cases.json")

# ── 6. Response data ──────────────────────────────────────────────────────────
responses_path = f"{DATA_DIR}/responses_2024.csv"
if os.path.exists(responses_path):
    responses = pd.read_csv(responses_path)
    responses_out = responses.to_dict(orient='records')
    with open(f"{DASH_DIR}/responses.json", 'w') as f:
        json.dump(responses_out, f, indent=2)
    print(f"✅ responses.json ({len(responses_out)} response records)")
else:
    print("⚠️  responses_2024.csv not found — run step6_response_engine.py first")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 COMPLETE ✅")
print("=" * 60)
print(f"""
Dashboard data exported to '{DASH_DIR}/':
  📄 summary.json       — national stats
  📄 trends.json        — disease time series per county
  📄 county_risk.json   — risk level per county
  📄 alerts.json        — all fired alerts
  📄 monthly_cases.json — 2024 monthly case counts

Copy the '{DASH_DIR}/' folder into your React app's public/ folder.
Then run the React dashboard to visualize everything.

ALL STEPS COMPLETE 🎉
""")
