# ============================================================
# SENTINEL — Disease Outbreak Detection System
# Step 2: Random Forest Disease Classifier
# ============================================================
# Trains a Random Forest model to predict disease from symptoms.
# Input:  Patient symptoms (0/1 for each symptom)
# Output: Predicted disease name
#
# HOW TO RUN:
#   pip install pandas scikit-learn joblib matplotlib seaborn
#   python step2_random_forest.py
#
# MUST run step1_data_cleaning.py first
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "clean_data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load clean symptom data ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: RANDOM FOREST CLASSIFIER")
print("=" * 60)

train_df = pd.read_csv(f"{DATA_DIR}/symptoms_train.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/symptoms_test.csv")

print(f"Training rows: {len(train_df)}")
print(f"Testing rows:  {len(test_df)}")

# ── Separate features (X) and labels (y) ──────────────────────────────────────
symptom_cols = [c for c in train_df.columns if c != 'prognosis']

X_train = train_df[symptom_cols].values
y_train = train_df['prognosis'].values

X_test  = test_df[symptom_cols].values
y_test  = test_df['prognosis'].values

print(f"\nFeatures (symptoms): {len(symptom_cols)}")
print(f"Classes (diseases):  {len(np.unique(y_train))}")

# ── Encode disease labels to numbers ─────────────────────────────────────────
le = LabelEncoder()
le.fit(y_train)

y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

# ── Train Random Forest ───────────────────────────────────────────────────────
print("\nTraining Random Forest...")
print("  → n_estimators = 200 trees")
print("  → max_depth    = 15")
print("  → This may take 10–30 seconds...")

rf_model = RandomForestClassifier(
    n_estimators=200,    # 200 decision trees
    max_depth=15,        # max depth per tree
    min_samples_split=2,
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

rf_model.fit(X_train, y_train_enc)
print("✅ Training complete!")

# ── Evaluate on test set ──────────────────────────────────────────────────────
print("\n" + "─" * 40)
print("MODEL EVALUATION ON TEST SET")
print("─" * 40)

y_pred_enc = rf_model.predict(X_test)
y_pred     = le.inverse_transform(y_pred_enc)

accuracy = accuracy_score(y_test_enc, y_pred_enc)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

# Detailed report for our 7 target diseases only
target_diseases = [
    'Malaria', 'Typhoid', 'Dengue', 'Tuberculosis',
    'Measles', 'Meningitis', 'Diarrheal diseases'
]

print("\nClassification Report (Target Diseases):")
target_mask = np.isin(y_test, target_diseases)
if target_mask.sum() > 0:
    print(classification_report(
        y_test[target_mask],
        y_pred[target_mask],
        zero_division=0
    ))

# Full report
print("\nFull Classification Report:")
print(classification_report(
    le.inverse_transform(y_test_enc),
    y_pred,
    zero_division=0
))

# ── Feature importance ────────────────────────────────────────────────────────
print("─" * 40)
print("TOP 20 MOST IMPORTANT SYMPTOMS:")
print("─" * 40)
importances = pd.Series(rf_model.feature_importances_, index=symptom_cols)
top20 = importances.nlargest(20)
for symptom, score in top20.items():
    bar = "█" * int(score * 500)
    print(f"  {symptom:<35} {score:.4f}  {bar}")

# ── Plot feature importance ───────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
top20.sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 20 Most Important Symptoms — Random Forest', fontsize=14)
plt.xlabel('Feature Importance Score')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
plt.close()
print(f"\n📊 Feature importance chart saved to {OUTPUT_DIR}/feature_importance.png")

# ── Confusion matrix for target diseases ─────────────────────────────────────
target_test_mask  = np.isin(y_test, target_diseases)
target_pred_mask  = np.isin(y_pred, target_diseases)
combined_mask     = target_test_mask

if combined_mask.sum() > 0:
    cm = confusion_matrix(
        y_test[combined_mask],
        y_pred[combined_mask],
        labels=target_diseases
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_diseases)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix — Target Outbreak Diseases', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

# ── Save model and encoder ────────────────────────────────────────────────────
joblib.dump(rf_model,    f"{MODEL_DIR}/random_forest_model.pkl")
joblib.dump(le,          f"{MODEL_DIR}/label_encoder.pkl")
joblib.dump(symptom_cols,f"{MODEL_DIR}/symptom_columns.pkl")

print(f"\n✅ Model saved to {MODEL_DIR}/random_forest_model.pkl")
print(f"✅ Label encoder saved to {MODEL_DIR}/label_encoder.pkl")

# ── Demo: predict disease from symptoms ───────────────────────────────────────
print("\n" + "=" * 60)
print("DEMO: PREDICT DISEASE FROM SYMPTOMS")
print("=" * 60)

def predict_disease(symptoms_present: list) -> dict:
    """
    Given a list of symptom names, predict the disease.
    
    Example:
        predict_disease(['high_fever', 'diarrhoea', 'vomiting', 'dehydration'])
    """
    input_vector = np.zeros(len(symptom_cols))
    for s in symptoms_present:
        s = s.strip()
        if s in symptom_cols:
            input_vector[symptom_cols.index(s)] = 1
        else:
            print(f"  ⚠️  Unknown symptom: '{s}' — skipped")

    proba    = rf_model.predict_proba([input_vector])[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    return {
        'predicted_disease': le.inverse_transform([top3_idx[0]])[0],
        'confidence':        round(proba[top3_idx[0]] * 100, 1),
        'top3': [
            {
                'disease':    le.inverse_transform([i])[0],
                'confidence': round(proba[i] * 100, 1)
            }
            for i in top3_idx
        ]
    }

# Test cases
test_cases = [
    {
        'name': 'Patient A — Malaria symptoms',
        'symptoms': ['high_fever', 'chills', 'sweating', 'headache', 'nausea', 'muscle_pain']
    },
    {
        'name': 'Patient B — Typhoid symptoms',
        'symptoms': ['high_fever', 'fatigue', 'stomach_pain', 'constipation', 'toxic_look_(typhos)']
    },
    {
        'name': 'Patient C — Diarrheal/Cholera symptoms',
        'symptoms': ['diarrhoea', 'vomiting', 'dehydration', 'stomach_pain', 'nausea']
    },
    {
        'name': 'Patient D — Meningitis symptoms',
        'symptoms': ['stiff_neck', 'high_fever', 'headache', 'vomiting', 'loss_of_balance']
    },
    {
        'name': 'Patient E — Measles symptoms',
        'symptoms': ['high_fever', 'skin_rash', 'cough', 'runny_nose', 'redness_of_eyes']
    }
]

for case in test_cases:
    result = predict_disease(case['symptoms'])
    print(f"\n{case['name']}")
    print(f"  Symptoms: {', '.join(case['symptoms'])}")
    print(f"  ➤ Predicted: {result['predicted_disease']} ({result['confidence']}% confidence)")
    print(f"  ➤ Top 3:")
    for r in result['top3']:
        print(f"      {r['disease']:<35} {r['confidence']}%")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 COMPLETE ✅")
print("=" * 60)
print(f"""
Model Performance:
  Overall Accuracy: {accuracy * 100:.2f}%

Files saved:
  🤖 {MODEL_DIR}/random_forest_model.pkl — trained model
  🏷️  {MODEL_DIR}/label_encoder.pkl       — disease label encoder
  📋 {MODEL_DIR}/symptom_columns.pkl     — symptom column list
  📊 {OUTPUT_DIR}/feature_importance.png — top symptoms chart
  📊 {OUTPUT_DIR}/confusion_matrix.png   — confusion matrix

Next step: Run step3_prophet_forecasting.py
""")
