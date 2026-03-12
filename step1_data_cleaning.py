# ============================================================
# SENTINEL — Disease Outbreak Detection System
# Step 1: Data Loading & Cleaning
# ============================================================
# Run this first before any other step.
# This script loads both datasets, cleans them, and saves
# clean versions ready for ML models.
#
# HOW TO RUN:
#   pip install pandas numpy
#   python step1_data_cleaning.py
# ============================================================

import pandas as pd
import numpy as np
import os

# ── File paths ────────────────────────────────────────────────────────────────
# Update these paths to where your files are located
IHME_PATH     = "IHME-GBD_2023_DATA-dde85905-1.csv"
TRAINING_PATH = "Training_extended.csv"
TESTING_PATH  = "Testing_extended.csv"

# Output paths (cleaned files saved here)
OUTPUT_DIR    = "clean_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── DATASET 1: IHME Kenya County Disease Data ─────────────────────────────────
print("=" * 60)
print("LOADING IHME DATASET...")
print("=" * 60)

ihme_raw = pd.read_csv(IHME_PATH)

print(f"Raw shape: {ihme_raw.shape}")
print(f"Columns: {list(ihme_raw.columns)}")
print(f"\nUnique measures: {ihme_raw['measure_name'].unique()}")
print(f"Unique metrics:  {ihme_raw['metric_name'].unique()}")

# ── Step 1a: Keep only Incidence (case counts) ────────────────────────────────
ihme = ihme_raw[
    (ihme_raw['measure_name'] == 'Incidence') &
    (ihme_raw['metric_name']  == 'Number')
].copy()

print(f"\nAfter filtering for Incidence + Number: {ihme.shape}")

# ── Step 1b: Select and rename relevant columns ───────────────────────────────
ihme = ihme[[
    'location_name',   # county
    'cause_name',      # disease
    'year',
    'val',             # case count
    'upper',           # confidence interval upper
    'lower'            # confidence interval lower
]].rename(columns={
    'location_name': 'county',
    'cause_name':    'disease',
    'val':           'cases',
    'upper':         'cases_upper',
    'lower':         'cases_lower'
})

# ── Step 1c: Round case counts to whole numbers ───────────────────────────────
ihme['cases']       = ihme['cases'].round().astype(int)
ihme['cases_upper'] = ihme['cases_upper'].round().astype(int)
ihme['cases_lower'] = ihme['cases_lower'].round().astype(int)

# ── Step 1d: Check for missing values ────────────────────────────────────────
print(f"\nMissing values:\n{ihme.isnull().sum()}")
ihme = ihme.dropna()

# ── Step 1e: Standardize disease names ───────────────────────────────────────
disease_map = {
    'Typhoid and paratyphoid': 'Typhoid',
    'Diarrheal diseases':      'Diarrheal diseases',
    'Tuberculosis':            'Tuberculosis',
    'Malaria':                 'Malaria',
    'Measles':                 'Measles',
    'Meningitis':              'Meningitis',
    'Dengue':                  'Dengue'
}
ihme['disease'] = ihme['disease'].map(disease_map)
ihme = ihme.dropna(subset=['disease'])  # drop 'All causes' rows

# ── Step 1f: Train / Test split ───────────────────────────────────────────────
# Train: 1980–2018 → Prophet learns the baseline
# Test:  2019–2023 → Evaluate Prophet accuracy
ihme_train = ihme[ihme['year'] <= 2018].copy()
ihme_test  = ihme[ihme['year'] >= 2019].copy()

print(f"\nIHME Train set (1980-2018): {ihme_train.shape}")
print(f"IHME Test  set (2019-2023): {ihme_test.shape}")

# ── Step 1g: Summary ─────────────────────────────────────────────────────────
print(f"\nDiseases: {sorted(ihme['disease'].unique())}")
print(f"Counties: {len(ihme['county'].unique())} counties")
print(f"Years:    {ihme['year'].min()} - {ihme['year'].max()}")
print(f"\nSample rows:")
print(ihme.head(10).to_string(index=False))

# ── Save IHME clean files ─────────────────────────────────────────────────────
ihme.to_csv(f"{OUTPUT_DIR}/ihme_clean.csv", index=False)
ihme_train.to_csv(f"{OUTPUT_DIR}/ihme_train.csv", index=False)
ihme_test.to_csv(f"{OUTPUT_DIR}/ihme_test.csv", index=False)
print(f"\n✅ IHME clean data saved to {OUTPUT_DIR}/")

# ── DATASET 2: Symptom Dataset ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOADING SYMPTOM DATASET...")
print("=" * 60)

train_df = pd.read_csv(TRAINING_PATH)
test_df  = pd.read_csv(TESTING_PATH)

print(f"Training shape: {train_df.shape}")
print(f"Testing shape:  {test_df.shape}")

# ── Step 2a: Clean column names ───────────────────────────────────────────────
train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()

# ── Step 2b: Remove empty columns ────────────────────────────────────────────
train_df = train_df.loc[:, train_df.columns != '']
test_df  = test_df.loc[:, test_df.columns != '']

# ── Step 2c: Clean prognosis column ──────────────────────────────────────────
train_df['prognosis'] = train_df['prognosis'].str.strip()
test_df['prognosis']  = test_df['prognosis'].str.strip()

# ── Step 2d: Drop rows with missing disease labels ────────────────────────────
train_df = train_df.dropna(subset=['prognosis'])
test_df  = test_df.dropna(subset=['prognosis'])

# ── Step 2e: Convert symptom columns to integers (0 or 1) ────────────────────
symptom_cols = [c for c in train_df.columns if c != 'prognosis']
train_df[symptom_cols] = train_df[symptom_cols].fillna(0).astype(int)
test_df[symptom_cols]  = test_df[symptom_cols].fillna(0).astype(int)

# ── Step 2f: Summary ──────────────────────────────────────────────────────────
print(f"\nTotal symptoms: {len(symptom_cols)}")
print(f"Total diseases: {train_df['prognosis'].nunique()}")
print(f"\nDisease distribution (training):")
print(train_df['prognosis'].value_counts().to_string())

# ── Step 2g: Check our 7 target diseases are present ─────────────────────────
target_diseases = [
    'Malaria', 'Typhoid', 'Dengue', 'Tuberculosis',
    'Measles', 'Meningitis', 'Diarrheal diseases'
]
print(f"\n{'─'*40}")
print("TARGET DISEASE CHECK:")
for d in target_diseases:
    count = len(train_df[train_df['prognosis'] == d])
    status = '✅' if count > 0 else '❌ MISSING'
    print(f"  {status} {d}: {count} rows")

# ── Save symptom clean files ──────────────────────────────────────────────────
train_df.to_csv(f"{OUTPUT_DIR}/symptoms_train.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/symptoms_test.csv", index=False)
print(f"\n✅ Symptom data saved to {OUTPUT_DIR}/")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 COMPLETE ✅")
print("=" * 60)
print(f"""
Files saved in '{OUTPUT_DIR}/':
  📄 ihme_clean.csv      — Full IHME data (all years)
  📄 ihme_train.csv      — IHME 1980-2018 (Prophet training)
  📄 ihme_test.csv       — IHME 2019-2023 (Prophet testing)
  📄 symptoms_train.csv  — Symptom training data
  📄 symptoms_test.csv   — Symptom testing data

Next step: Run step2_random_forest.py
""")
